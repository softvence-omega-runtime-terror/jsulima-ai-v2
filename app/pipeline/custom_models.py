import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ManualVotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators
        
    def predict_proba(self, X):
        probs = []
        for name, clf in self.estimators:
            probs.append(clf.predict_proba(X))
        return np.mean(probs, axis=0)
        
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
