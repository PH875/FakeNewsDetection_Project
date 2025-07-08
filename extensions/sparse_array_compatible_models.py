import numpy as np
import scipy.sparse as sp

class TrainableModel_S:
    """
    Base class for models trained using iterative optimization, adapted for sparse matrix compatibility
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def loss_grad(self, X, y): 
        """Subclasses must override this to return parameter gradients."""
        raise NotImplementedError
    
    def decision_function(self, X):
        """Subclasses must override this to return model's decision function."""
        raise NotImplementedError
    
    def _get_params(self):
        """Subclasses must override this to return dictionary of model's parameters."""
        raise NotImplementedError

    def compute_metrics(self, X, Y, metrics_dict):
        """
        Compute metrics 

        Parameters:
        ----------
        X: array
          Features matrix 
        Y: array
          Labels vector
        metrics_dict: dict
          Dictionary with metric names as keys and
          functions to comoute metrics as values

          The expected syntax of a metric function is metric(y_pred, y_true) 
        
        Returns:
        -------
        metrics: dict
          Dictionary of current metric values
        """
        metrics = {}
        for metric_name in metrics_dict:
            metrics[metric_name] = metrics_dict[metric_name](self.decision_function(X),Y)
        return metrics

    def fit(self, X, y, num_epochs=10, batch_size=100, compute_metrics=False, metrics_dict=None):
        if compute_metrics:
            metrics_history = {name: [] for name in metrics_dict}
            metrics = self.compute_metrics(X, y, metrics_dict)
            for name in metrics_dict:
                metrics_history[name].append(metrics[name])
        else:
            metrics_history = None

        for _ in range(num_epochs):
            indices = np.random.permutation(X.shape[0])
            batches = np.array_split(indices, np.ceil(X.shape[0] / batch_size))

            for idx in batches:
                grads = self.loss_grad(X[idx], y[idx])
                self.optimizer.update(self._get_params(), grads)

                if compute_metrics:
                    metrics = self.compute_metrics(X, y, metrics_dict)
                    for name in metrics_dict:
                        metrics_history[name].append(metrics[name])

        return metrics_history
    
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression_S(TrainableModel_S):
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
    
    

class LinearSVM_S(TrainableModel_S):

    def __init__(self, w, b, optimizer, C=10., offset=None):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)
        self.C = C
        if offset is None:
            self.offset=np.zeros(len(w))
        else:
            self.offset=offset
    
    def loss_grad(self, X, y):
       # Compute raw model output
        output = self.decision_function(X)

        # Identify margin violations: where 1 - y*h(x) > 0
        mask = (1 - y * output) > 0
        y_masked = y[mask]
        X_masked = X[mask]

        # Compute 
        if len(y_masked) > 0:
            if sp.issparse(X_masked):
                grad_w=2 * self.w - self.C * (np.array(X_masked.multiply(y_masked[:, None]).mean(axis=0)).reshape((-1,))-np.mean(y_masked)*self.offset)
            else:
                grad_w = 2 * self.w - self.C * np.mean(y_masked[:, None] * X_masked, axis=0)
            grad_b = - self.C * np.mean(y_masked)
        else:
            grad_b = 0.0
            grad_w = 2 * self.w

        return {"w": grad_w, "b": grad_b}
    
    def decision_function(self, X):
        return X @ self.w -np.dot(self.offset,self.w) + self.b
    
    def _get_params(self):
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
