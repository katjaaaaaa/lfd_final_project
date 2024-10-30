# bad ass words https://github.com/zacanger/profane-words/blob/master/words.json

import logging
import argparse
import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import json
import random
import lstm
import transformer


random.seed(5)

# Setup logging configuration
logging.basicConfig(filename='logs.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')


# Custom function to log and print
def log_and_print(message, printed=True):
    """Logs a message and prints it to the console."""
    logging.info(message)
    if printed:
        print(message)


def create_arg_parser():
    parser = argparse.ArgumentParser()

    # GENERAL STUFF
    parser.add_argument("-d","--data", default=False, action="store_true",
                        help="Provides information about the datasets")
    parser.add_argument("-log", "--log", default=False, action="store_true",
                        help="Logs the output and arguments")
    parser.add_argument("-md", "--make_dev", default=False, action="store_true", # change so it catches the file name as well
                        help="Create a new dev file")
    parser.add_argument("-t", "--test", default="False", action="store_true",
                        help="Run the model on the test set")
    parser.add_argument("-mt", "--make_test", default=False, action="store_true", # change so it catches the file name as well
                        help="Create a new test file")
    parser.add_argument("-m", "--model", default="nb", choices=["nb", "lstm", "llm"],
                        help="Defines the model to make the predictions")
    
    # BASELINE STUFF
    parser.add_argument("-vec", "--vectorizer", choices=["bow", "tfidf", "both"],
                        default="bow", help="Select vectorizer: bow (bag of words), tfidf or both")
    parser.add_argument("-ng", "--ngram_range", nargs=2, type=int, default=(1, 1),
                        help="Set the ngram range, give two integers separated by space")
    parser.add_argument("-lemma", "--lemmas", action="store_true",
                        help="Lemmatizes the tokenized data.")
    parser.add_argument("-w", "--weights", default=False, action="store_true",
                        help="Adds weights to the baseline classifier")

    # LSTM STUFF
    parser.add_argument("-bi", "--bidirectional", action="store_true",
                        help="If added, use a bidirectional LSTM")
    parser.add_argument("-lr", "--learning_rate", default=0.01, type=float,
                    help="Set the learning rate")
    parser.add_argument("-ww", "--class_weights", nargs=2, type=int, default=[1, 1],
                        help="Sets the class weights to balance the data")
    parser.add_argument("-l", "--loss_function", default="binary_crossentropy", type=str,
                        help="Set the loss function")
    parser.add_argument("-a", "--activation", default="sigmoid", type=str,
                        help="Set the activation")
    parser.add_argument("-ah", "--activation_hidden", default="sigmoid", type=str,
                        help="Set the activation for the hidden layer")
    parser.add_argument("-bs", "--batch_size", default=16, type=int,
                        help="Set the batch size")
    parser.add_argument("-ep", "--epochs", default=50, type=int,
                        help="Set the number of epochs")
    parser.add_argument("-mo", "--momentum", default=0.9, type=float,
                        help="Controls the influence of previous epoch on the next weight update")
    parser.add_argument("-es", "--early_stop", default=3, type=int,
                        help="Set the patience of early stop")
    parser.add_argument("-o", "--optimizer", default="adam", choices=["sgd", "adam"],
                        help="Select optimizer (SGD, ADAM)")
    parser.add_argument("-dr", "--dropout", default=None, type=float,
                        help="Set a dropout layer")
    parser.add_argument("-ex", "--extra_layer", default=0, type=int, choices=[1, 2],
                        help="Set an amount of extra layers, max 2 extra layers, keeps the same settings as the base layer.")

    # TRANSFORMERS STUFF
    parser.add_argument("-tr", "--transformer", default="roberta", choices=["distilbert", "roberta", "electra", "deberta"])



    args = parser.parse_args()
    return args


def select_classifier(args):

    '''
    Select the model and initialize it with the given arguments
    '''

    if args.model == "nb":
         model = MultinomialNB(alpha=0.21)

    return model

def data_preprocess(data, off=False):

    data_new = data.copy(deep=True)

    # replace necessary words
    data_new["text"] = data_new["text"].str.replace('@USER ', '')
    data_new["text"] = data_new["text"].str.replace('URL', '')

    data_new["text"] = data_new["text"].apply(lambda x: x.lower())

    if off:
        with open('./data/words.json') as f:
            off_words = json.load(f)
        print("Start preprocessing data")
        for word in off_words:
            data_new["text"] = data_new["text"].str.replace(rf"\b{word}\b", "offword", regex=True)
        #print(data_new[data_new["text"].str.contains("offword", na=False)].head)
        print("Data preprocessed")

    data_new["text"] = data_new["text"].str.replace(r'[^\w\s]', '', regex=True)


    #nltk.download('punkt_tab')
    # if args.ntopwords:
    #     tt = nltk.tokenize.TweetTokenizer()
    #     tokens = sum([tt.tokenize(line) for line in data_new["text"]], [])
    #     tokens_dict = nltk.FreqDist(w for w in tokens)
    #     for k in tokens_dict.keys():
    #         if k != "offword":
    #             data_new["text"] = data_new["text"].str.replace(rf"\b{k}\b", "", regex=True)

    return data_new


def data_count(train, dev, test):

    '''Provides information about the datasets'''

    data = ["Train data", "Dev data", "Test data"]

    for i, df in enumerate([train, dev, test]):
        print(f"{data[i]}:")
        print(f"Total: {len(df.index)}")
        print(f"Unique: {df['off'].value_counts()}")


def select_vectorizer(args):
    """
    Initialize the vectorizer based on the given arguments.
    """
    # Initialize vectorizers with selected arguments.
    tf_idf = TfidfVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=tuple(args.ngram_range), token_pattern=None)
    if args.lemmas:
        bow = CountVectorizer(analyzer="word",preprocessor=identity, tokenizer=identity, ngram_range=tuple(args.ngram_range), token_pattern=None)
    else:
        bow = CountVectorizer(analyzer="char", ngram_range=tuple(args.ngram_range))

    union = FeatureUnion([("count", bow), ("tf", tf_idf)])

    if args.vectorizer == "tfidf":
        return tf_idf
    elif args.vectorizer == "bow":
        # Bag of Words vectorizer
        return bow
    elif args.vectorizer == "both":
        return union


def identity(inp):
    """
    Returns the input, or the lemmatized input if the user has chosen
    lemmatization.
    """
    if args.lemmas:
        lemmatizer = WordNetLemmatizer()
        lemma_list = [lemmatizer.lemmatize(word) for word in inp]

        return lemma_list

    return inp



def run_model(args, train, test):

    vectorizer = select_vectorizer(args)
    model = select_classifier(args)

    X_train = vectorizer.fit_transform(train["text"])
    X_test = vectorizer.transform(test["text"])

    y_train = train["off"]
    y_test = test["off"]

    if args.weights:
        c_w = class_weight.compute_class_weight(class_weight={"NOT" : 1, "OFF": 2}, classes=np.unique(y_train), y=y_train)
        c_w = {"NOT" : c_w[0], "OFF" : c_w[1]}
        model.fit(X_train, y_train, sample_weight=[c_w[i] for i in y_train.to_numpy()])
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if args.log:
        log_and_print(classification_report(y_test, y_pred))
    else:
        print(classification_report(y_test, y_pred))



def prepare_data(args, train, test, mode="dev"):

    full_train = data_preprocess(train)
    full_test = data_preprocess(test)

    # Preprocess all the off_data and store in the files
    if args.make_dev or args.make_test:
        off_train = data_preprocess(train, True)
        off_test = data_preprocess(test, True)
        off_train.to_csv('./data/off_train.tsv', sep='\t', index=False)
        if mode == "test":
            off_test.to_csv('./data/off_test.tsv', sep='\t', index=False)
        else:
            off_test.to_csv('./data/off_dev.tsv', sep='\t', index=False)
    else:
        # Load already created datasets
        off_train = pd.read_csv("./data/off_train.tsv", sep='\t')

        if mode == "test":
            off_test = pd.read_csv("./data/off_test.tsv", sep='\t')
        else:
            off_test = pd.read_csv("./data/off_dev.tsv", sep='\t')

    return full_train, full_test, off_train, off_test


if __name__ == "__main__":
    args = create_arg_parser()

    colnames = ["text", "off"]
    df_train = pd.read_csv("./data/train.tsv", sep='\t', header=None, names=colnames)
    df_dev = pd.read_csv("./data/dev.tsv", sep='\t', header=None, names=colnames)
    df_test = pd.read_csv("./data/test.tsv", sep='\t', header=None, names=colnames)

    if args.data:
        data_count(df_train, df_dev, df_test)
        exit()


    # run the baseline
    if args.model == "nb":
        if args.test:
            full_train, full_test, off_train, off_test = prepare_data(args, df_train, df_test, "test")
        else:
            full_train, full_test, off_train, off_test = prepare_data(args, df_train, df_dev)

        print("FULL DATA")
        run_model(args, full_train, full_test)
        print("NO OFF DATA")
        run_model(args, off_train, off_test)

    # run lstm
    elif args.model == "lstm":
        full_train, full_test, off_train, off_test = prepare_data(args, df_train, df_test, "test")
        _, full_dev, _, off_dev = prepare_data(args, df_train, df_dev)

        print("FULL DATA")
        lstm.run(full_train, full_dev, full_test, args)
    
        print("NO OFF DATA")
        lstm.run(off_train, off_dev, off_test, args)

    elif args.model == "llm":
        full_train, full_test, off_train, off_test = prepare_data(args, df_train, df_test, "test")
        _, full_dev, _, off_dev = prepare_data(args, df_train, df_dev)

        print("FULL DATA")
        transformer.run(full_train, full_dev, full_test, args)
    
        print("NO OFF DATA")
        transformer.run(off_train, off_dev, off_test, args)


    if args.log:
        all_args = " \\\n".join([f" --{key}={value}" for key, value in vars(args).items() if value])
        log_and_print(f"Used settings:\n{all_args}", False)