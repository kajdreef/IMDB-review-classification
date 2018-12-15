import os
import numpy as np
import argparse

from data_loader import load_vocab_dict, load_train_data, load_test_data
from pprint import pprint
from sklearn.metrics import f1_score, precision_score, recall_score
from util import curry, output_to_csv, output_features_to_file, load_features_from_file
from preprocessing import clean_text, remove_stop_words, negation_handling, lemmatizing, emoji_tagging
from features_extractor import Extractor, extract_tf, extract_tf_idf, extract_sentiment
from feature_selection import k_best
from classify import classifier, compute_auc
from cross_validation import cross_validation
 

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


def extract_features(Xtr_text, Xte_text, stop_words=None, ngram_range=(1, 1), min_df=0.0, max_df=1.0, max_features=None):
    # Extract features (Yes, this is based on The One programming style haha)
    print("Extract features...")

    print("Obtaining classic data...")
    # Connect the preprocessing functions
    extractor = Extractor(Xtr_text, Xte_text)\
        .bind(curry(emoji_tagging))\
        .bind(curry(remove_stop_words))\
        .bind(curry(lemmatizing))\
        .bind(curry(negation_handling))\
        .bind(curry(clean_text))
        

    # Add the feature extractor functions
    extractor\
        .bind(extract_tf(ngram_range=ngram_range, min_df=min_df, max_df=max_df))\
        .bind(extract_tf_idf)\
        .bind(k_best(Ytr, Yte, max_features))

    # Extract the features
    Xtr, Xte = extractor.get_features()

    print("Xtr shape: {}".format(Xtr.shape))
    print("Xte shape: {}".format(Xte.shape))
    print("Done extracting features!")
    return Xtr, Xte


def run_classifier(name, Xtr, Ytr, Xte, Yte, init={}, ensemble_run=False):
    learner = classifier(name, init=init)
    l = learner.get_learner()
    print("learner initialized......")
    score = cross_validation(l, Xtr, Ytr)
    print("cross_validation complete.....")
    learner.train(Xtr, Ytr)
    print("learner training complete.....")
    Yte_hat = learner.predict(Xte)
    print(Yte_hat)
    mean_cv_score = np.mean(score['test_score'])
    auc = compute_auc(Yte, Yte_hat)
    print("{} - AUC: {}".format(name, auc))

    f1 = f1_score(Yte, Yte_hat)
    precision = precision_score(Yte, Yte_hat)
    recall = recall_score(Yte, Yte_hat)

    print("{} - Mean 10-fold CV: {}".format(name, mean_cv_score))
    print("{} - precision: {}".format(name, precision))
    print("{} - recall: {}".format(name, recall))
    print("{} - f1-score: {}".format(name, f1))
    print("{} - AUC: {}".format(name, auc))

    return [name, auc, mean_cv_score, f1, precision, recall] if not ensemble_run else [Yte_hat, mean_cv_score]


def ensembles(ensemble_predictions, name="Ensembles"):
    percent_lower, percent_higher = 0.35, 0.65
    percent1, percent2, percent3 = 0.23, 0.40, 0.37

    if(len(ensemble_predictions)==2): 
        Yhat_final = percent_lower*ensemble_predictions[0][0] + percent_higher*ensemble_predictions[1][0]
        mean_cv_score = percent_lower*ensemble_predictions[0][1] + percent_higher*ensemble_predictions[1][1]
    elif(len(ensemble_predictions)==3):
        Yhat_final = percent1*ensemble_predictions[0][0] \
                    + percent2*ensemble_predictions[1][0] \
                    + percent3*ensemble_predictions[2][0]
        mean_cv_score = percent1*ensemble_predictions[0][1] \
                        + percent2*ensemble_predictions[1][1] \
                        +percent3*ensemble_predictions[2][1]

    auc = compute_auc(Yte, Yhat_final)
    print("######### {} - AUC: {}".format(name, auc))

    print("{} - Mean 10-fold CV: {}".format(name, mean_cv_score))

    precision = precision_score(Yte, [ round(y) for y in Yhat_final ])
    print("{} - precision: {}".format(name, precision))

    recall = recall_score(Yte, [ round(y) for y in Yhat_final ])
    print("{} - recall: {}".format(name, recall))

    f1 = f1_score(Yte, [ round(y) for y in Yhat_final ])
    print("{} - f1-score: {}".format(name, f1))


if __name__ == '__main__':
    print("Classify IMDB data")

    if not os.path.exists("./aclImdb/"):
        print("IMDB data set is missing")
        exit(-1)
    
    # Command line argument parser options
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-new-features", action='store_true', default=False, help="Assumes features have already been extracted, so it will skip the feature extracing step.")
    parser.add_argument("--with-ensemble", action='store_true', default=False, help="Assumes features have already been extracted, so it will skip the feature extracing step.")

    args = parser.parse_args()

    #--------------------- Parameters----------------------------
    ngram_range = (1, 1)  # unigrams
    min_df = 2 # Filter words out that only occur in one document
    max_df = 0.5 # filter all words out that occur in more than half the documents
    n_features = 10000 # Max number of features

    auc_out = []
    classifiers_config = [
        ("Random Forest", {'n_estimators':100}),
        ("Logistic", {'solver':'lbfgs'}),
        ("Linear SVM SGD", {'max_iter': 1000, 'tol': 1e-3}),
        # ("Logistic SGD", {}),
        # ("KMeans", {
        #     'n_clusters': 2, 'init': 'k-means++', 'random_state': 0}),
        # ("kNN", {}),
        # ("MLP", {}),
        # ("Bagging", {'n_clusters': 2, 'init': 'k-means++', 'random_state': 0}), -- KMeans
        # ("Bagging", {'random_state': 0, 'max_features': "sqrt", 'min_samples_split': 600}) #-- DecisionTrees
        # ("DecisionTrees", {'random_state': 0, 'max_features': "sqrt", 'min_samples_split': 600}),
    ]
    #------------------------------------------------------------
    Xtr, Ytr, Xte, Yte = None, None, None, None

    if not args.no_new_features:
        Xtr_text, Ytr, Xte_text, Yte = load_data()

        Xtr, Xte =  extract_features(Xtr_text, Xte_text,
                        ngram_range=ngram_range,
                        min_df=min_df,
                        max_df=max_df,
                        max_features=n_features
                    )
        output_features_to_file(Xtr, Ytr, Xte, Yte)
    else: 
        Xtr, Ytr, Xte, Yte = load_features_from_file()


    # Run different Classifiers and determine the AUC 
    print("Start running the different classifiers...")
    ensemble_predictions = []
    for (classifier_name, init) in classifiers_config:
        result = run_classifier(classifier_name, Xtr, Ytr, Xte, Yte, init=init, ensemble_run=args.with_ensemble)
        if args.with_ensemble:
            ensemble_predictions.append(result)
        auc_out.append(result)

    if args.with_ensemble:
        ensembles(ensemble_predictions)

    output_csv_loc = "./data/classifier_auc.csv"
    print("Output the results to: {}".format(output_csv_loc))
    head = ["Classifier", "AUC", "Mean 10-fold CV", "F1-score", "Precision", "Recall"]
    output_to_csv(output_csv_loc, head, auc_out)

