import numpy as np
class LogisticRegression():
    
    def __init__(self,X_train,y_train,X_test,y_test, lr=0.001, n_iters=1000, cv=5):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cv = cv
        
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def fit(self):
        n_samples, n_features = self.X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        fold_size = n_samples // self.cv
        accuracy_scores = []
        
        for i in range(self.cv):
            X_test_fold = self.X_train[i*fold_size:(i+1)*fold_size]
            y_test_fold = self.y_train[i*fold_size:(i+1)*fold_size]
            X_train_fold = np.concatenate([self.X_train[:i*fold_size], self.X_train[(i+1)*fold_size:]])
            y_train_fold = np.concatenate([self.y_train[:i*fold_size], self.y_train[(i+1)*fold_size:]])

            for _ in range(self.n_iters):
                linear_pred = np.dot(X_train_fold, self.weights) + self.bias
                predictions = self.sigmoid(linear_pred)

                dw = (1/X_train_fold.shape[0]) * np.dot(X_train_fold.T, (predictions - y_train_fold))
                db = (1/X_train_fold.shape[0]) * np.sum(predictions-y_train_fold)

                self.weights = self.weights - self.lr*dw
                self.bias = self.bias - self.lr*db

            y_pred = self.predict(X_test_fold)
            accuracy_fold = np.mean(y_test_fold == y_pred)
            accuracy_scores.append(accuracy_fold)

        return np.mean(accuracy_scores)
    
    def score(self):
        y_pred = self.predict(self.X_test)
        accuracy = np.mean(self.y_test == y_pred)
        return accuracy
    
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred
