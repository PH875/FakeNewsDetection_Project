import numpy as np
from .base import TrainableModel

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression(TrainableModel):
    """
    Binary logistic regression model with optional regularization, adapted for sparse matrix compatibility

    Parameters:
        - w: Initial weights (array-like)
        - b: Initial bias (scalar)
        - optimizer: Optimizer object (e.g., GDOptimizer)
        - penalty: One of {"none", "ridge", "lasso"}
        - lam: Regularization strength (default: 0.0)
        - offset: offset of all X in the followin function, the methods give the same result for some X as the corresponding methods
            in the original LogisticRegression with X-offset. offset is used to make the computation more efficient, useful for z-score normalization of sparse matrices
    """
    
    def __init__(self, w, b, optimizer, penalty="none", lam=0.0, offset=None):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)
        self.penalty = penalty
        self.lam = float(lam)
        if offset is None:
            self.offset=np.zeros(len(w))
        else:
            self.offset=offset
        

    def loss_grad(self, X, y):
        residual = self.decision_function(X) - y
        grad_w = (X.T @ residual-np.sum(residual)*self.offset) / X.shape[0]
        grad_b = np.mean(residual)

        # Add regularization if specified
        if self.penalty == "ridge":
            grad_w += self.lam * self.w
        elif self.penalty == "lasso":
            grad_w += self.lam * np.sign(self.w)

        return {"w": grad_w, "b": grad_b}
    
    def decision_function(self, X):
        return sigmoid(X @ self.w -np.dot(self.offset,self.w) + self.b)
    
    def _get_params(self):
        """
        Return model parameters as a dict for the optimizer.
        """
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return np.where(self.decision_function(X) >= 0.5, 1, 0)