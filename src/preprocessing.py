import re

# copied from: https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
__REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
__REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def __preprocess_review(review):
    review = __REPLACE_NO_SPACE.sub("", str(review))
    review = __REPLACE_WITH_SPACE.sub(" ", str(review))
    
    return review

def preprocess_reviews(reviews):
    return [__preprocess_review(review) for review in reviews]

def negate_handling():
    pass


