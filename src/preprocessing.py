import re
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import mark_negation
from nltk import sent_tokenize, word_tokenize

# Smileys were ordered in Happy/Sad based on imdb.vocab/imdbEr.txt
__HAPPY_SMILEYS = re.compile(
    r'(\:\-\))|(\:\))|(\(\:)|(\(\=)|(\:\-P)|(\:P)|(\:\-D)|(\:D)|(\:\>)|(\:\-\>)|(\<\:\-\))|(\=\^\.\^\=)(\=\-\))|(\=\))|(\=\-P)|(\=\-D)|(\=D)|(\=\])|(\:d)')
__SAD_SMILEY = re.compile(
    r'(\:\-\()|(\:\()|(\:\<)|(\:\-\<)|(\<\:\-\()|(\:\-\@)|(\:\@)|(\:\-\)|(\:\-\!)|\=\-\()|(\=\()|(\=\<)|(\=\-\<)|(\=d)|(\=P)|(\)\:)')


def emoji_tagging(review):
    """
    Replaces emoji's in a String with HAPPY_SMILEY or SAD_SMILEY. 
    Make sure emoji tagging is done before cleaning text else smileys will be seen as separate characters and be removed.
    Input:
        review: a string
    Output:
        review: A string where the emoji's (e.g., :D) are replaced with HAPPY_SMILEY or SAD_SMILEY.
    """
    review = __HAPPY_SMILEYS.sub(" HAPPY_SMILEY ", str(review))
    review = __SAD_SMILEY.sub(" SAD_SMILEY ", str(review))
    return review



# Inspired, but changed for regular expressions (https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184)
# Remove single characters (a-z)
__REPLACE_WITH_SPACE_SINGLE_CHAR = re.compile(
    r'(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\`)|(\()|(\))|(\[)|(\])|(\*)|(\')|(\+)|(\$)|(\#)|(\@)|(\%)')
__REPLACE_WITH_SPACE = re.compile(
    r'<\s*br[^>]*>|(\-)|(\/)|(^| )[a-z]( |$)|(^|[^a-zA-Z])[0-9]+([^a-zA-Z]|$)|(^| )\_NEG( |$)')


def clean_text(review):
    """
    Remove punctuation from sentences.
    """
    review = __REPLACE_WITH_SPACE_SINGLE_CHAR.sub(" ", str(review))
    review = __REPLACE_WITH_SPACE.sub(" ", str(review))
    return review

def remove_stop_words(review):
    stop_words = set(stopwords.words('english'))
    stop_words.remove("not")
    stop_words.add("the")
    stop_words.add("it")
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


def negation_handling(review):
    """
    args:
        review: input data is a strings
    returns:
        out: string with negation handled
    """
    # Negation limitation based on: http://www.jcomputers.us/vol12/jcp1205-11.pdf
    punctuation = ['.', ',', ':', ';', '!', '?', '\'', '\"', '(', ')']
    out = []
    for sentence in sent_tokenize(review):
        tokens = word_tokenize(sentence)
        tokens = ['.' if t in punctuation else t for t in tokens]
        tokens_neg = mark_negation(tokens, double_neg_flip=True)
        out.append(" ".join(tokens_neg))

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
