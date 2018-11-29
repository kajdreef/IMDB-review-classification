from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


class classifier:
    learner = None

    def __init__(self, alg=None):
        if alg == "random_forest":
            self.learner = RandomForestClassifier()
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
    