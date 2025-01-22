import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import numpy as np
import re
from tqdm import tqdm
import keras
import tensorflow as tf
from Tokenizer import Tokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

BASE_DIR = 'C:/Users/Admin/Documents/PYCHARM/flickr8k'
WORKING_DIR = 'C:/Users/Admin/Documents/PYCHARM/YsaProje'
directory = os.path.join(BASE_DIR, 'Images')


base_model = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = keras.layers.GlobalAveragePooling2D()(base_model.output)
model = keras.models.Model(inputs=base_model.inputs, outputs=x)
feature_extractor = keras.models.Model(inputs=base_model.inputs, outputs=x)


features = {}
for img_name in tqdm(os.listdir(directory)):
    img_path = os.path.join(directory, img_name)
    image = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    feature = feature_extractor.predict(image, verbose=0)
    image_id = img_name.split('.')[0]
    features[image_id] = feature

pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))

with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)

with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()

mapping = {}
for line in captions_doc.split('\n'):
    tokens = line.split(',')
    if len(tokens) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]
    caption = " ".join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = re.sub(r'[^a-z\s]', '', caption)
            caption = re.sub(r'\s+', ' ', caption).strip()
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            captions[i] = caption

clean(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in all_captions)

with open(os.path.join(WORKING_DIR, 'max_length.pkl'), 'wb') as f:
    pickle.dump(max_length, f)

with open(os.path.join(WORKING_DIR, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)

image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]

def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size):
    for key in data_keys:
        if key not in features:
            continue
        captions = mapping[key]
        for caption in captions:
            seq = tokenizer.texts_to_sequences([caption])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                yield (features[key][0], in_seq), out_seq

output_signature = (
    (tf.TensorSpec(shape=(1280,), dtype=tf.float32), tf.TensorSpec(shape=(max_length,), dtype=tf.float32)),
    tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
)

dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train, mapping, features, tokenizer, max_length, vocab_size),
    output_signature=output_signature
)

embedding_index = {}
glove_file = os.path.join(BASE_DIR, 'glove.6B.300d.txt')

with open(glove_file, encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefficients

embedding_dim = 300

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.random.normal(size=(embedding_dim,))

dataset = dataset.batch(8).prefetch(tf.data.AUTOTUNE)

inputs1 = keras.layers.Input(shape=(1280,), name="image")
fe1 = keras.layers.Dropout(0.5)(inputs1)
fe2 = keras.layers.Dense(256, activation='relu')(fe1)

inputs2 = keras.layers.Input(shape=(max_length,), name="text")
se1 = keras.layers.Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    weights=[embedding_matrix],
    input_length=max_length,
    trainable=True
)(inputs2)

se2 = keras.layers.Dropout(0.5)(se1)
se3 = keras.layers.LSTM(256, return_sequences=True)(se2)

attention_units = 256

multi_head_attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=256)
context_vector = multi_head_attention(query=se3, value=se3, key=se3)

fe2_expanded = keras.layers.Lambda(
    lambda x: tf.expand_dims(x, axis=1),
    output_shape=(1, 256)
)(fe2)

fe2_repeated = keras.layers.Lambda(
    lambda args: tf.tile(args[0], [1, tf.shape(args[1])[1], 1]),
    output_shape=lambda input_shapes: (input_shapes[0][0], None, input_shapes[0][2])
)([fe2_expanded, se3])

decoder1 = keras.layers.Concatenate()([context_vector, fe2_repeated])
decoder2 = keras.layers.Dense(256, activation='relu')(decoder1)

outputs = keras.layers.Dense(vocab_size, activation='softmax')(decoder2)
outputs = keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1))(outputs)

model = keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0005))

keras.utils.plot_model(model, show_shapes=True)

steps_per_epoch = len(train) // 8

for i in range(30):
    model.fit(dataset, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)

model.save(os.path.join(WORKING_DIR, 'mobilenetv2_model.h5'))

def predict_caption(model, feature, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

actual, predicted = list(), list()
for key in tqdm(test):
    captions = mapping[key]
    y_pred = predict_caption(model, features[key], tokenizer, max_length)
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    actual.append(actual_captions)
    predicted.append(y_pred)

smoothing_function = SmoothingFunction().method1
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smoothing_function))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function))

while True:

    image_name = input("Lütfen tahmin yapılacak görüntü adını girin (çıkmak için 'q'): ")
    if image_name.lower() == 'q':
        print("Program sonlandırılıyor...")
        break

    if not image_name.endswith('.jpg'):
        image_name += '.jpg'

    image_path = os.path.join('C:/Users/Admin/Documents/PYCHARM/flickr8k/Images', image_name)

    if not os.path.exists(image_path):
        print(f"Görüntü bulunamadı: {image_path}")
        continue

    image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    feature = feature_extractor.predict(image, verbose=0)

    caption = predict_caption(model, feature, tokenizer, max_length)
    print(caption)


