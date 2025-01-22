ImageCaptioning
This repository implements an advanced image captioning model using deep learning techniques. The project aims to generate meaningful captions for input images by leveraging a combination of feature extraction, sequence modeling, and attention mechanisms.

**Features**

**Feature Extraction:** Utilizes the pre-trained MobileNetV2 model to extract rich feature representations of images.

**Caption Generation:** Employs an LSTM-based sequence generator with an embedding layer initialized using GloVe embeddings.

**Attention Mechanism:** Incorporates a multi-head attention layer for better context modeling during caption generation.

**BLEU Score Evaluation:** Evaluates the generated captions using BLEU metrics to assess quality and relevance.

**How It Works**

**Feature Extraction**

• The MobileNetV2 model is used to extract features from images.
• Features are stored in a serialized file (features.pkl) for later use.

**Text Preprocessing:**

Captions are cleaned and tokenized.
A tokenizer is created to map words to indices and vice versa.
GloVe embeddings are used to initialize the embedding layer for better semantic understanding.

**Model Architecture:**

The model includes;
• An image encoder for visual feature extraction.
• A text decoder with LSTM layers to generate captions.
• A multi-head attention mechanism to enhance context understanding between image and text.
Outputs predictions using a dense layer with a softmax activation.

**Training:**

• Training data is split into train and test sets.
• A data generator provides batches of sequences and features for training.
• The model is trained using categorical cross-entropy loss with the Adam optimizer.

**Evaluation:**

• BLEU-1 and BLEU-2 scores are computed to measure the quality of generated captions against actual captions.

The project utilizes the Flickr8k dataset for training.
Pre-trained GloVe embeddings provide semantic initialization for the embedding layer.

**ImageCaptioning.py:** Main script for training and inference.
**features.pkl:** Serialized file containing image features.
**tokenizer.pkl:** Tokenizer for text preprocessing.
**max_length.pkl:** Maximum sequence length used in training.
**mobilenetv2_model.h5:** Trained model weights.
