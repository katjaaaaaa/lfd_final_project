import argparse
import json
import logging
import random as python_random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Bidirectional
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.initializers import Constant
import run_models

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer


# def read_embeddings(embeddings_file):
#     """Read in word embeddings from file and save as numpy array"""
#     with open(embeddings_file) as f1:
#         emb_list = f1.readlines()

#     embeddings = dict()

#     for line in emb_list:
#         line_list = line.rstrip().split()
#         embeddings[line_list[0]] = np.array(line_list[1:], "float32")

#     return embeddings
#     #return {word: np.array(embeddings) for word in embeddings}


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


def create_model(Y_train, emb_matrix, args):
    """Create the Keras model to use"""

    # Define hyperparameter settings
    learning_rate = args.learning_rate
    loss_function = args.loss_function
    momentum = args.momentum
    activation = args.activation

    # Define the optimizer
    optim = SGD(learning_rate=learning_rate,
                momentum=momentum, nesterov=False)
    if args.optimizer == "adam":
        optim = Adam(learning_rate=learning_rate)

    # Take embedding dim and size from emb_matrix
    embedding_dim = len(emb_matrix[0])
    num_tokens = len(emb_matrix)
    num_labels = len(set(Y_train))
    # Now build the model
    model = Sequential()
    model.add(Embedding(num_tokens, embedding_dim, embeddings_initializer=Constant(emb_matrix), trainable=False))

    # Add bidirectional LSTM layer
    if args.bidirectional and not args.extra_layer:
        model.add(Bidirectional(LSTM(embedding_dim)))

    # Adds the bidirectional LSTM layers and the possibility to add extra layers, 1 or 2.
    elif args.bidirectional and args.extra_layer:
        model.add(Bidirectional(LSTM(embedding_dim, return_sequences=True)))

        # Adds dropout layer for first layer if asked
        if args.dropout:
            model.add(Dropout(args.dropout))
        if args.extra_layer > 1:
            for i in range(args.extra_layer - 1):
                model.add(Bidirectional(LSTM(embedding_dim, return_sequences=True)))
                # Adds dropout layer for each layer if asked
                if args.dropout:
                    model.add(Dropout(args.dropout))

        model.add(Bidirectional(LSTM(embedding_dim)))

    # Adds the LSTM layers and the possibility to add extra layers, 1 or 2.
    elif args.extra_layer and not args.bidirectional:
        model.add(LSTM(embedding_dim, return_sequences=True))

        # Adds a dropout layer for the first layer if asked
        if args.dropout:
            model.add(Dropout(args.dropout))

        if args.extra_layer > 1:
            for i in range(args.extra_layer - 1):
                model.add(LSTM(embedding_dim, return_sequences=True))

                # Adds dropout layer for each layer if asked
                if args.dropout:
                    model.add(Dropout(args.dropout))
        model.add(LSTM(embedding_dim))

    # Adds the LSTM layer
    else:
        model.add(LSTM(embedding_dim))

    # Adds the last dropout layer if asked
    if args.dropout and args.extra_layer < 2:
        model.add(Dropout(args.dropout))

    # Ultimately, end with dense layer with the activation function
    model.add(Dense(1, activation=activation))

    # Compile model using our settings, check for accuracy
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])
    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev, args):
    """Train the LSTM model."""
    verbose = 1
    batch_size = args.batch_size
    epochs = args.epochs
    # Early stopping: stop training when there are three consecutive epochs without improving
    # It"s also possible to monitor the training loss with monitor="loss"
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stop)
    # Finally fit the model to our data
    class_weights = { 0 : args.class_weights[0], 1 : args.class_weights[1]}
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs, callbacks=[callback], batch_size=batch_size,
              class_weight=class_weights, validation_data=(X_dev, Y_dev))
    # Print final accuracy for the model (clearer overview)
    test_set_predict(args, model, X_dev, Y_dev, "dev")
    return model


def test_set_predict(args, model, X_test, Y_test, ident):
    """Do predictions and measure accuracy on our own test set (that we split off train)"""
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)

    # Finally, convert to numerical labels to get scores with sklearn
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
    embeddings = read_embeddings("./data/embeddings/glove.twitter.27B.25d.txt")

    # Transform words to indices using a vectorizer
    vectorizer = TextVectorization(standardize=None, output_sequence_length=50)
    # Use train and dev to create vocab - could also do just train
    text_ds = tf.data.Dataset.from_tensor_slices(X_train + X_dev)
    vectorizer.adapt(text_ds)
    # Dictionary mapping words to idx
    voc = vectorizer.get_vocabulary()
    emb_matrix = get_emb_matrix(voc, embeddings)   

    # # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    y_train_bin = encoder.fit_transform(y_train)  # Use encoder.classes_ to find mapping back
    y_dev_bin = encoder.fit_transform(y_dev)

    print(np.unique(y_train_bin, return_counts=True ))
    print(np.unique(y_dev_bin, return_counts=True ))
    # if args.transofmer:
        # ...
    # else:

    # Create model
    model = create_model(y_train, emb_matrix, args)

    # Transform input to vectorized input
    X_train_vect = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_dev_vect = vectorizer(np.array([[s] for s in X_dev])).numpy()

    # Train the model
    model = train_model(model, X_train_vect, y_train_bin, X_dev_vect, y_dev_bin, args)

    # Do predictions on specified test set
    if args.test:
        X_test, y_test = test["text"].to_list(), test["off"].to_list()
        X_test_vect = vectorizer(np.array([[s] for s in X_test])).numpy()
        test_set_predict(args, model, X_test_vect, y_test, "test")
