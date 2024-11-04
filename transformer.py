import argparse
import json
import logging
import random as python_random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Bidirectional
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.initializers import Constant
import run_models

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

def read_embeddings(embeddings_file):
    """Read in word embeddings from file and save as numpy array"""
    embeddings = {}

    with open(embeddings_file) as f:
        for line in f:
            line = line.rstrip().split()
            word = line[0]
            vector = np.asarray(line[1:], "float32")
            embeddings[word] = vector

    return embeddings

def get_emb_matrix(voc, emb):
    """Get embedding matrix given vocab and the embeddings"""
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def compile_transformer(lm, args):
    """
    Compile transformer model.
    """
    learning_rate = args.learning_rate
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=1)
    # loss_function = CategoricalCrossentropy(from_logits=True)
    loss_function = BinaryCrossentropy(from_logits=True)
    optim = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])

    return model


def predict_transformers(args, model, tokens_dev, Y_dev_bin, ident):
    """
    Create prediction for the transformer model.
    """
    # Get predictions
    Y_pred = model.predict(tokens_dev)["logits"]
    # Finally, convert to numerical labels to get scores with sklearn
    Y_test = Y_dev_bin
    Y_pred = (Y_pred >= 0.5).astype(int)
    if args.log:
        # If you have gold data, you can calculate accuracy
        run_models.log_and_print("Accuracy on own {1} set: {0}".format(round(accuracy_score(Y_test, Y_pred), 3), ident))
        run_models.log_and_print("f1 score on own {1} set: {0}".format(round(f1_score(Y_test, Y_pred, average="macro"), 3), ident))
        run_models.log_and_print(classification_report(Y_test, Y_pred))
    else:
        print("Accuracy on own {1} set: {0}".format(round(accuracy_score(Y_test, Y_pred), 3), ident))
        print("f1 score on own {1} set: {0}".format(round(f1_score(Y_test, Y_pred, average="macro"), 3), ident))
        print(classification_report(Y_test, Y_pred))


def run(train, dev, test, args):

    # opening data
    X_train, y_train = train["text"].to_list(), train["off"]
    X_dev, y_dev = dev["text"].to_list(), dev["off"]
    embeddings = read_embeddings("/content/gdrive/MyDrive/lfd_final/data/embeddings/glove.twitter.27B.200d.txt")

    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    #emb_matrix = get_emb_matrix(voc, embeddings)   

    # # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    y_train_bin = encoder.fit_transform(y_train)  # Use encoder.classes_ to find mapping back
    Y_dev_bin = encoder.fit_transform(y_dev)

    if args.transformer == "electra":
        lm = "google/electra-small-discriminator"
    elif args.transformer == "roberta":
        lm = "FacebookAI/roberta-base"
    elif args.transformer == "deberta":
        lm = "microsoft/deberta-v3-base"
    else:
        lm = "distilbert/distilbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(lm)
    transformer_tokens_train = tokenizer(X_train, padding=True, max_length=100,
                                            truncation=True, return_tensors="np").data
    transformer_tokens_dev = tokenizer(X_dev, padding=True, max_length=100,
                                        truncation=True, return_tensors="np").data

    model = compile_transformer(lm, args)
    model.fit(transformer_tokens_train, y_train_bin, verbose=1, epochs=args.epochs,
                batch_size=args.batch_size, validation_data=(transformer_tokens_dev, Y_dev_bin))
    predict_transformers(args, model, transformer_tokens_dev, Y_dev_bin, "dev")

 # Do predictions on specified test set
    if args.test:
        X_test, Y_test = test["text"].to_list(), test["off"].to_list()
        Y_test_bin = encoder.fit_transform(Y_test)
        transformer_tokens_test = tokenizer(X_test, padding=True, max_length=100,
                                                truncation=True, return_tensors="np").data
        # Finally do the predictions
        predict_transformers(args, model, transformer_tokens_test, Y_test_bin, "test")

