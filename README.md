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

After the initial run the features will be stored in the data folder. After that we can rerun the program with the following flag:

`python3 src/main.py --no-new-features`

This will not recompute the features, giving us the option to optimize the classification part without having to recompute the features each time.

## Preprocessing
- Clean text
- Remove stop words
- negation handling (Still need work, Clean text and negation handling are giving problems together &_NEG can occur and clean text removes &)
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
- Linear SVM SGD
- Random Forest
- kmeans
- knn
- MLP