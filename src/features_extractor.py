from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

class Extractor:
    """ Just and helper class to make pipelining the extractors nicer looking
    """
    def __init__(self, Xtr, Xte):
        self._xtr = Xtr
        self._xte = Xte


    def bind(self, func):
        self._xtr, self._xte = func(self._xtr, self._xte)
        return self
    
    def get_features(self):
        return self._xtr, self._xte

def extract_tf(ngram_range=(1, 1), min_df=None):
    def extract(Xtr, Xte):
        """ Extract the TF from a string
        args:
            X: an array of strings
        Returns:
            Xtf: a matrix where the rows are the reviews and columns the tf for each feature.
        """
        cv = CountVectorizer(
            binary=True,
            ngram_range=ngram_range,
            min_df=min_df
        )
        cv.fit(Xtr)
        Xtr = cv.transform(Xtr)
        Xte = cv.transform(Xte)
        return Xtr, Xte
    return extract


def extract_tf_idf(Xtr, Xte):
    """ Extract the tf-idf from a tf matrix
    args:
        X: term frequency matrix
    Returns:
        Xtf_idf: A matrix containing the tf-idf
    """
    tf_transformer = TfidfTransformer()
    tf_transformer.fit(Xtr)
    Xtr = tf_transformer.transform(Xtr)
    Xte = tf_transformer.transform(Xte)
    return Xtr, Xte


def extract_sentiment(Xtr, Xte):
    s_analyszer = SentimentIntensityAnalyzer()
    review_sentiment_tr = []
    review_sentiment_te = []

    for r in Xtr:
        ss = s_analyszer.polarity_scores(r)
        review_sentiment_tr.append(
            [ss['neg'], ss['neu'], ss['pos'], ss['compound']])

    for r in Xte:
        ss = s_analyszer.polarity_scores(r)
        review_sentiment_te.append(
            [ss['neg'], ss['neu'], ss['pos'], ss['compound']])

    return [np.asarray(review_sentiment_tr), np.asarray(review_sentiment_te)]
