from sklearn.base import BaseEstimator
import numpy as np

class CustomBernoulliNaiveBayes(BaseEstimator):  
    """A custom Bernoulli Naive Bayes implementation for COMP551 Mini-Project 2"""

    def __init__(self, laplaceSmoothing=True):
        """
        Initializing the custom, from scratch Bernoulli Naive Bayes
        """
        self.laplaceSmoothing = laplaceSmoothing
        
    def fit(self, X, y):
        """
        Fits Bernoulli Naive Bayes model to training data, X, with target values, y
        """
        
        ## number of 0/1 examples in training
        Ny1 = np.sum(y)
        Ny0 = y.shape[0] - Ny1
        
        ## percentage of 0/1 examples in training
        self.theta0 = Ny0/y.shape[0]
        self.theta1 = Ny1/y.shape[0]
               
        ## counts for each feature
        Nj1 = X.T.dot(y.reshape([-1,1]))  
        Nj0 = X.T.dot(1-y.reshape([-1,1]))
        
        if self.laplaceSmoothing:
            self.T1 = (Nj1 + 1)/(Ny1 + 2)
            self.T0 = (Nj0 + 1)/(Ny0 + 2)
        else:
            self.T1 = Nj1/Ny1
            self.T0 = Nj0/Ny0
        
        return self

    def predict(self, X):
        
        ## ensure that the model has been trained before predicting
        try:
            getattr(self, "T0")
            getattr(self, "T1")
            getattr(self, "theta0")
            getattr(self, "theta1")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        ## calculate probablity function, delta (lecture 9, slide 5)
        delta = (  np.log(self.theta1/(1-self.theta1)) 
                 + X.dot(np.log(self.T1/self.T0))
                 + (1-X.todense()).dot(np.log((1-self.T1)/(1-self.T0))) )
    
        
        ## make prediction from learned weights
        pred = np.zeros(delta.shape).astype('int')
        pred[delta > 0] = 1

        return pred.reshape([-1,])

    def score(self, X, y):
        ## ensure features and target are binary
        assert np.array_equal(y, y.astype(bool))
        
        ## predict output from learned weights
        pred = self.predict(X)
        
        ## compare to true targets
        diff = np.equal(pred, y)
        
        ## sum how many examples are correctly predicted
        score = np.sum(diff)/y.shape[0] 
        return score