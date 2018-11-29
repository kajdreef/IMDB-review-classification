import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# copied from: https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
__REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
__REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def __preprocess_review(review):
    review = __REPLACE_NO_SPACE.sub("", str(review))
    review = __REPLACE_WITH_SPACE.sub(" ", str(review))
    review = __stemming(review)
    return review

def preprocess_reviews(reviews):
    return [__preprocess_review(review) for review in reviews]


def __stemming(review):
    """
    Takes a word and returns the Stem of the word.
    """
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in review.split(' '):
        stemmed_words.append(stemmer.stem(word))

    return ' '.join(stemmed_words)
