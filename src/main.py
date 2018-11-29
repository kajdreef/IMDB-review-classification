import os
import numpy as np

from data_loader import load_vocab_dict, load_train_data, load_test_data
from nltk.corpus import stopwords
from pprint import pprint

from features_extractor import Extractor, clean_text, extract_tf, extract_tf_idf
from classify import classifier, compute_auc

def run_classifier(name, Xtr, Ytr, Xte, Yte, init={}):
    print("Start classification...")
    learner = classifier(name, init=init)
    learner.train(Xtr, Ytr)

    Yte_hat = learner.predict(Xte)
    print("{} - AUC: {}".format(name, compute_auc(Yte, Yte_hat)))
    print("Done!")

if __name__ == '__main__':
    print("Classify IMDB data")

    if not os.path.exists("./aclImdb/"):
        print("IMDB data set is missing")
        exit(-1)
    
    # Load data
    print("Load Data...")
    Xtr_text, Ytr, Xva_text, Yva = load_train_data('./aclImdb/test/', 0.1)
    Xte_text, Yte = load_test_data('./aclImdb/train/')

    # Combine training and validation data:
    Xtr_text = np.append(Xtr_text, Xva_text)
    Ytr = np.append(Ytr, Yva)
    print("Done loading data!\n")

    # Only use to speed up to program for testing --------------
    # subsample = 1000
    # Xtr_text = Xtr_text[:subsample]
    # Xte_text = Xte_text[:subsample]
    # Yte = Yte[:subsample]
    # Ytr = Ytr[:subsample]
    #-----------------------------------------------------------

    # Extract features (Yes, this is based on The One programming style haha)
    print("Extract features...")
    
    stop_words = set(stopwords.words('english'))
    ngram_range = (2,2) # bigrams

    Xtr, Xte = Extractor(Xtr_text, Xte_text)\
        .bind(clean_text)\
        .bind(extract_tf(stop_words=stop_words, ngram_range=(1, 1)))\
        .bind(extract_tf_idf)\
        .get_features()
    
    print("Xtr shape: {}".format(Xtr.shape))
    print("Xte shape: {}".format(Xte.shape))
    print("Done extracting features!")

    # Run different Classifiers and determine the AUC (Bayes has been left out for now)
    run_classifier("random_forest", Xtr, Ytr, Xte, Yte)
    run_classifier("logistic", Xtr, Ytr, Xte, Yte)
    # run_classifier("kmeans", Xtr, Ytr, Xte, Yte, init={'n_clusters':2, 'init':'k-means++', 'random_state':0})
    # run_classifier("knn", Xtr, Ytr, Xte, Yte)
    # run_classifier("MLP", Xtr, Ytr, Xte, Yte)
