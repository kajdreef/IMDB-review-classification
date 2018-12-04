import os
import numpy as np

from data_loader import load_vocab_dict, load_train_data, load_test_data
from nltk.corpus import stopwords
from pprint import pprint

from util import curry, output_to_csv
from preprocessing import clean_text, remove_stop_words, negate_handling, lemmatizing
from features_extractor import Extractor, extract_tf, extract_tf_idf, extract_sentiment
from classify import classifier, compute_auc
 

def load_data(train_data_path='./aclImdb/train/', test_data_path='./aclImdb/test/'):
    # Load data
    print("Load Data...")
    Xtr_text, Ytr, Xva_text, Yva = load_train_data(train_data_path, 0.1)
    Xte_text, Yte = load_test_data(test_data_path)

    # Combine training and validation data:
    Xtr_text = np.append(Xtr_text, Xva_text)
    Ytr = np.append(Ytr, Yva)
    print("Done loading data!\n")

    return Xtr_text, Ytr, Xte_text, Yte


def extract_features(Xtr_text, Xte_text, stop_words=None, ngram_range=(1, 1), min_df=0.0):
    # Extract features (Yes, this is based on The One programming style haha)
    print("Extract features...")

    print("Obtaining classic data...")
    # Connect the preprocessing functions
    extractor = Extractor(Xtr_text, Xte_text)\
        .bind(curry(remove_stop_words))\
        .bind(curry(negate_handling))\
        .bind(curry(clean_text))\
        .bind(curry(lemmatizing))

    # Add the feature extractor functions
    extractor\
        .bind(extract_tf(ngram_range=ngram_range, min_df=min_df))\
        .bind(extract_tf_idf)

    # Extract the features
    Xtr, Xte = extractor.get_features()

    print("Xtr shape: {}".format(Xtr.shape))
    print("Xte shape: {}".format(Xte.shape))
    print("Done extracting features!")
    return Xtr, Xte


def run_classifier(name, Xtr, Ytr, Xte, Yte, init={}):
    learner = classifier(name, init=init)
    learner.train(Xtr, Ytr)

    Yte_hat = learner.predict(Xte)
    auc = compute_auc(Yte, Yte_hat)
    print("{} - AUC: {}".format(name, auc))
    return [name, auc]

if __name__ == '__main__':
    print("Classify IMDB data")

    if not os.path.exists("./aclImdb/"):
        print("IMDB data set is missing")
        exit(-1)
    
    #--------------------- Parameters----------------------------
    stop_words = set(stopwords.words('english'))
    ngram_range = (1, 3)  # bigrams
    min_df = 0.0005
    
    auc_out = []
    classifiers_config = [
        ("Random Forest", {}),
        ("logistic", {}),
        ("Linear SVM SGD", {}),
        ("Logistic SGD", {}),
        ("KMeans", {
            'n_clusters': 2, 'init': 'k-means++', 'random_state': 0}),
        ("kNN", {}),
        ("MLP", {})
    ]
    #------------------------------------------------------------
    Xtr_text, Ytr, Xte_text, Yte = load_data()

    Xtr, Xte =  extract_features(Xtr_text, Xte_text,
                    ngram_range=ngram_range,
                    min_df=min_df
                )

    # Run different Classifiers and determine the AUC 
    print("Start running the different classifiers...")
    for (classifier_name, init) in classifiers_config:
        result = run_classifier(classifier_name, Xtr, Ytr, Xte, Yte, init=init)
        auc_out.append(result)
    
    output_csv_loc = "./data/classifier_auc.csv"
    print("Output the results to: {}".format(output_csv_loc))
    head = ["Classifier", "AUC"]
    output_to_csv(output_csv_loc, head, auc_out)
