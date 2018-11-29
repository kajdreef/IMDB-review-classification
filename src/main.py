import os
import numpy as np

from data_loader import load_vocab_dict, load_train_data, load_test_data
from features_extractor import Extractor, clean_text, extract_tf, extract_tf_idf

from classify import classifier, compute_auc

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

    # Extract features (Yes, this is based on The One programming style haha)
    print("Extract features...")
    Xtr, Xte = Extractor(Xtr_text, Xte_text)\
        .bind(clean_text)\
        .bind(extract_tf)\
        .bind(extract_tf_idf)\
        .get_features()
    
    print("Xtr shape: {}".format(Xtr.shape))
    print("Xte shape: {}".format(Xte.shape))
    print("Done extracting features!")

    # classify
    print("Start classification...")
    learner = classifier("random_forest")
    learner.train(Xtr, Ytr)
    
    Yte_hat = learner.predict(Xte)
    print("AUC: {}".format(compute_auc(Yte, Yte_hat)))
    print("Done!")
