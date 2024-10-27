import logging
import argparse
import pandas as pd
from itertools import combinations
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


# Setup logging configuration
logging.basicConfig(filename='results.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')


# Custom function to log and print
def log_and_print(message, printed=True):
    """Logs a message and prints it to the console."""
    logging.info(message)
    if printed:
        print(message)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data", default=False, action="store_true",
                        help="Provides information about the datasets")
    parser.add_argument("-t", "--test", default=False, action="store_true", # change so it catches the file name as well
                        help="Runs the model on the test set")
    parser.add_argument("-m", "--model", default="nb", choices=["nb", "x", "lstm", "llm"],
                        help="Defines the model to make the predictions")
    parser.add_argument("-vec", "--vectorizer", choices=["bow", "tfidf", "both"],
                        default="bow", help="Select vectorizer: bow (bag of words), tfidf or both")
    args = parser.parse_args()

    return args


def select_classifier(args):

    '''
    Select the model and initialize it with the given arguments
    '''

    if args.model == "nb":
         model = MultinomialNB(alpha=0.21)

    return model

def data_preprocess(data):

    # replace necessary words
    data["text"] = data["text"].str.replace('@USER', 'usersign')
    data["text"] = data["text"].str.replace(r"#\w+", "hashtagsign", regex=True)

    # remove all characters except for ascii supported letters and numbers
    filter_char = lambda c: ord(c) < 256
    #data["text"] = data["text"].apply(lambda s: ''.join(filter(filter_char, s)))
    data["text"] = data["text"].apply(lambda x: x.lower())
    data["text"] = data["text"].str.replace(r'[^\w\s]', '', regex=True)

    return data


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
    #tf_idf = TfidfVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=tuple(arguments.ngram_range))
    #bow = CountVectorizer(preprocessor=identity, tokenizer=identity, ngram_range=tuple(arguments.ngram_range))
    bow = CountVectorizer()
    #union = FeatureUnion([("count", bow), ("tf", tf_idf)])

    if args.vectorizer == "tfidf":
        pass
        # return tf_idf
    elif args.vectorizer == "bow":
        # Bag of Words vectorizer
        return bow
    elif args.vectorizer == "both":
        pass
        # return union


def run_model(args, train, test):

    vectorizer = select_vectorizer(args)
    model = select_classifier(args)

    X_train = vectorizer.fit_transform(train["text"])
    X_dev = vectorizer.transform(test["text"])

    y_train = train["off"]
    y_dev = test["off"]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_dev)

    print(classification_report(y_dev, y_pred))


def prepare_for_triple(args, train, test):

    train_full, test_full = train, test
    train_hash, test_hash = train, test
    train_user, test_user = train, test

    # data with hashtags only
    train_hash = train[train["text"].str.contains(r"\bhashtagsign\b", case=False, na=False)]
    test_hash = test[test["text"].str.contains(r"\bhashtagsign\b", case=False, na=False)]
    train_hash.loc[:, "text"] = train_hash["text"].str.replace('usersign', '')
    test_hash.loc[:, "text"] = test_hash["text"].str.replace('usersign', '')

    # data with user mentions only
    train_user = train[train["text"].str.contains(r"\busersign\b", case=False, na=True)]
    test_user = test[test["text"].str.contains(r"\busersign\b", case=False, na=True)]
    train_user.loc[:, "text"] = train_user["text"].str.replace('hashtagsign', '')
    test_user.loc[:, "text"] = test_user["text"].str.replace('hashtagsign', '')

    data_list = [train_full, test_full, train_hash, test_hash, train_user, test_user]

    print("FULL DATA")
    run_model(args, train_full, test_full)
    print("HASH ONLY DATA")
    run_model(args, train_hash, test_hash)
    print("USER ONLY DATA")
    run_model(args, train_user, test_user)

def main():
    args = create_arg_parser()

    colnames = ["text", "off"]
    df_train = pd.read_csv("./data/train.tsv", sep='\t', header=None, names=colnames)
    df_dev = pd.read_csv("./data/dev.tsv", sep='\t', header=None, names=colnames)
    df_test = pd.read_csv("./data/test.tsv", sep='\t', header=None, names=colnames)

    if args.data:
        data_count(df_train, df_dev, df_test)
        exit()

    df_train = data_preprocess(df_train)
    df_dev = data_preprocess(df_dev)
    df_test = data_preprocess(df_test)

    if args.test:
        pass
        #run_model(args, df_train, df_test)
    else:
        prepare_for_triple(args, df_train, df_dev)
        #run_model(args, df_train, df_dev)



if __name__ == "__main__":
    main()