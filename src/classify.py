from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

class classifier:
    learner = None

    def __init__(self, alg=None, init={}):
        if alg == "Random Forest":
            self.learner = RandomForestClassifier(**init)
        elif alg == "Logistic":
            self.learner = LogisticRegression(**init)
        elif alg == "kNN":
            self.learner = KNeighborsClassifier(**init)
        elif alg == "MLP":
            self.learner = MLPClassifier(**init)
        elif alg == "KMeans":
            self.learner = KMeans(**init)
        elif alg == "Linear SVM SGD":
            self.learner = SGDClassifier(loss='hinge', penalty='l1')
        elif alg == "Logistic SGD":
            self.learner = SGDClassifier(loss='log', penalty='l1')
        elif alg == "Bagging":
            self.learner = BaggingClassifier(DecisionTreeRegressor(**init), max_samples=0.5, max_features=0.5)
        elif alg == "DecisionTrees":
            self.learner = DecisionTreeRegressor(**init)
        else:
            raise("Classifier not yet implemented")

    def train(self, Xtr, Ytr=None):
        print("training the model.....")
        self.learner.fit(Xtr, Ytr)
    
    def predict(self, Xte):
        print("making predictions.....")
        Yhat = self.learner.predict(Xte)
        return Yhat
    
    def get_learner(self):
        return self.learner


def compute_auc(Y, Yhat):
    print("computing AUC.....")
    fp, tp, _ = metrics.roc_curve(Y, Yhat)
    return metrics.auc(fp, tp)
    
