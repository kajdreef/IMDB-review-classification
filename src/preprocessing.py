import re
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import mark_negation
from nltk import sent_tokenize, word_tokenize

# copied from: https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
__REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
__REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def clean_text(review):
    ## Remove random puncuation
    review = __REPLACE_NO_SPACE.sub("", str(review))
    review = __REPLACE_WITH_SPACE.sub(" ", str(review))
    return review

def remove_stop_words(review):
    stop_words = set(stopwords.words('english'))
    out = []

    # Split review up in sentences
    for sentence in sent_tokenize(review):
        tokens = word_tokenize(sentence)

        # Check each sentence and remove stop words
        filtered_tokens = []
        for word in tokens:
            if word.lower() not in stop_words:
                filtered_tokens.append(word)

        out.append(" ".join(filtered_tokens))

    return " ".join(out)


def negate_handling(review):
    """
    args:
        review: input data is a strings
    returns:
        out: string with negation handled
    """
    out = []
    for sentence in sent_tokenize(review):
        tokens = word_tokenize(sentence)
        tokens = ['.' if t == ',' else t for t in tokens]
        out.append(" ".join(mark_negation(tokens, double_neg_flip=True)))

    return " ".join(out)

def stemming(review):
    """
    Takes a word and returns the Stem of the word.
    """
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in review.split(' '):
        stemmed_words.append(stemmer.stem(word))

    return ' '.join(stemmed_words)

def lemmatizing(review):
    """
    Takes a word and returns the lemmatization of the word.
    """
    lemmatization = WordNetLemmatizer()

    lemmatized_words = []
    for word in review.split(' '):
        lemmatized_words.append(lemmatization.lemmatize(word))

    return ' '.join(lemmatized_words)
