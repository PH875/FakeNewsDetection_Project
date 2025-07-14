import numpy as np
import scipy.sparse as sp
import cvxopt
from .base import TrainableModel

class Kernel:
    def _check_shapes(self, X1, X2):
        if X1.shape[-1] != X2.shape[-1]:
            raise ValueError("Inputs must have the same number of features (last dimension).")

        d = X1.shape[-1]
        return X1.reshape(-1, d), X2.reshape(-1, d)
    

class LinearKernel(Kernel):
    def __call__(self, X1, X2):
        X1_flat, X2_flat = self._check_shapes(X1, X2)
        return (X1_flat @ X2_flat.T).reshape(X1.shape[:-1] + X2.shape[:-1])


class PolynomialKernel(Kernel):
    def __init__(self, degree=2, intercept=1):
        self.degree = degree
        self.intercept = intercept

    def __call__(self, X1, X2):
        X1_flat, X2_flat = self._check_shapes(X1, X2)
        prod = (X1_flat @ X2_flat.T).reshape(X1.shape[:-1] + X2.shape[:-1])
        return np.power(prod + self.intercept, self.degree)


class RBFKernel(Kernel):
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, X1, X2):
        X1_flat, X2_flat = self._check_shapes(X1, X2)
        diff = np.linalg.norm(
            X1_flat[:, np.newaxis, :] - X2_flat[np.newaxis, :, :],
            axis=-1
        ).reshape(X1.shape[:-1] + X2.shape[:-1])
        return np.exp(-diff**2 / (2 * self.sigma ** 2))


class LinearSVM(TrainableModel):
    """
    Linear SVM model with optional regularization, adapted for sparse matrix compatibility

    Parameters:
        - w: Initial weights (array-like)
        - b: Initial bias (scalar)
        - C: C-parameter
        - optimizer: Optimizer object (e.g., GDOptimizer)
        - penalty: One of {"none", "ridge", "lasso"}
        - offset: 1-dimensional array, default None (corresponding to 0); offset of all X in the following functions, the methods gives the same result for some X as the
        corresponding methods in the original LogisticRegression class would give with X - offset (broadcasted). The offset can be used to make the computation more 
        efficient, for example when one wants to z-score normalize sparse matrices without effectively destroying sparsity
    """

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
    


class BinaryKernelSVM:
    def __init__(self, C=1.0, kernel='linear', **kwargs):
        self.C = C

        if kernel == 'rbf':
            sigma = kwargs.get('sigma', 1.0)
            self.kernel = RBFKernel(sigma)
        elif kernel == 'polynomial':
            degree = kwargs.get('degree', 2)
            intercept = kwargs.get('intercept', 1)
            self.kernel = PolynomialKernel(degree, intercept)
        elif kernel == 'linear':
            self.kernel = LinearKernel()
        elif kernel == 'custom':
            self.kernel = kwargs['kernel_function']
        else:
            raise ValueError(f"Unknown kernel type: '{kernel}'")
        
        self.sv = None
        self.sv_y = None
        self.alphas = None
        self.b = None

    def fit(self, X, Y):
        n = X.shape[0]
        _Y = Y.astype(float)
        K = self.kernel(X, X)

        # Construct QP matrices
        P = cvxopt.matrix(np.outer(_Y, _Y) * K)
        q = cvxopt.matrix(-np.ones(n))
        A = cvxopt.matrix(_Y.reshape(1, -1))
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.vstack((-np.eye(n), np.eye(n))))
        h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))

        # Disable output
        cvxopt.solvers.options['show_progress'] = False
        
        # Solve QP
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(solution['x'])

        # Support vectors: alphas > threshold
        sv_mask = alphas > 1e-5
        self.sv = X[sv_mask]
        self.sv_y = Y[sv_mask]
        self.alphas = alphas[sv_mask]

        # Compute bias term b using support vector on the margin
        for i in range(len(self.alphas)):
            if self.alphas[i] < self.C - 1e-5:
                K_i = self.kernel(self.sv[[i]], self.sv)
                self.b = self.sv_y[i] - np.sum(self.alphas * self.sv_y * K_i)
                break
        else:
            self.b = 0.0  # fallback if all support vectors are violating the margin

    def decision_function(self, X):
        K = self.kernel(X, self.sv)
        return np.sum(K * self.sv_y * self.alphas, axis=-1) + self.b

    def __call__(self, X):
        return np.where(self.decision_function(X) > 0, 1, -1)
    

