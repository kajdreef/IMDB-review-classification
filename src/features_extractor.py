from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

from nltk import bigrams
from nltk.corpus import stopwords

from preprocessing import preprocess_reviews

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

def clean_text(Xtr, Xte):
    return preprocess_reviews(Xtr), preprocess_reviews(Xte)


def extract_tf(stop_words=None, ngram_range=(1,1)):
    def extract(Xtr, Xte):
        """ Extract the TF from a string
        args:
            X: an array of strings
        Returns:
            Xtf: a matrix where the rows are the reviews and columns the tf for each feature.
        """
        cv = CountVectorizer(
            binary=True,
            stop_words=stop_words,
            ngram_range=ngram_range
        
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
