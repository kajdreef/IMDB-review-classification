from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


class classifier:
    learner = None

    def __init__(self, alg=None, init={}):
        if alg == "random_forest":
            self.learner = RandomForestClassifier(**init)
        elif alg == "logistic":
            self.learner = LogisticRegression(**init)
        elif alg == "knn":
            self.learner = KNeighborsClassifier(**init)
        elif alg == "MLP":
            self.learner = MLPClassifier(**init)
        elif alg == "kmeans":
            self.learner = KMeans(**init)
        else:
            raise("Classifier not yet implemented")

    def train(self, Xtr, Ytr=None):
        self.learner.fit(Xtr, Ytr)
    
    def predict(self, Xte):
        Yhat = self.learner.predict(Xte)
        return Yhat

def compute_auc(Y, Yhat):
    fp, tp, _ = metrics.roc_curve(Y, Yhat)
    return metrics.auc(fp, tp)
    
