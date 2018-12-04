# CS273A-IMDB-review-classification

## Requirements

+ Python 3
+ Numpy
+ Sklearn
+ nltk

Run the following commands from the root folder:

```sh
pip3 install -r requirements.txt
python3 setup.py
```

## Running program

At the moment it uses Logistic classification algorithm, with tf-idf as features.
That results in an AUC of 0.89.

The program can be run using the following command from the command line:
`python3 src/main.py`

The Jupyter notebook is out of date, and I am currently only using to run small tests.

## Preprocessing
- Clean text
- Remove stop words
- negation handling
- lemmatizing
- stemming (Not used in favor for lemmatizing)

## Features
- TF-IDF
- Extract sentiment using VADER*

(*TF-IDF showed better performance so far)

## Classifiers
Logistic regression, SVM, and MLP showed best performance, but other classfiers are available as well:

- Logistic
- SVM using SGD for training (performs similar to Logistic)
- Random Forest
- kmeans
- knn
- MLP