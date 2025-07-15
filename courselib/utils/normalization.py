import numpy as np
import scipy.sparse as sp

def standardize(x):
    """
    Standardization normalization.
    Scales the data to have mean 0 and standard deviation 1.
    Parameters:
    - x: numpy array of shape (n_samples, n_features)
    Returns:
    - numpy array of the same shape as x, with mean 0 and std 1
    """
    return (x - np.mean(x, axis=0))/np.std(x,axis=0)


def min_max(x):
    """
    Min-Max normalization.
    Scales the data to a range of [0, 1] by transforming each feature individually.
    Parameters:
    - x: numpy array of shape (n_samples, n_features)
    Returns:
    - numpy array of the same shape as x, with all values scaled to the range [0, 1]
    """
    return (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))



def lp_normalize(X, ord=None):
    """
    normalizes rows of an array, e.g. for l2 or l1 normalization, compatible with sparse matrices
    
    Paramters:
    X: np array or scipy sparse ndarray or matrix
    ord: non-zero None|1|inf, default=None (L2 normalization); order of the norm (None=l2-norm, 1=l1-norm, inf=l\infty norm)
    
    Returns:
    X_normalized: np array or scipy sparse csr matrix (depending on type of X); rows of X normalized to ord-norm 1
    """
    if sp.issparse(X):
        X=X.tocsr() # make sure X is csr matrix
        norms=sp.linalg.norm(X, ord=ord, axis=1)
        norms[norms==0]=1 # avoid dividing by zero (without changing normalization)
        X_normalized=(X.T.multiply(1/norms).tocsr()).T
    else:
        norms=np.linalg.norm(X, ord, axis=1)
        norms[norms==0]=1 # avoid dividing by zero (without changing normalization)
        X_normalized=(X.T*(1/norms)).T
    
    return X_normalized

def standardize_sparse_matrix(X):
    """
    scales sparse matrix, s.t each column has norm standard deviation one (without shifting, so sparsity is contained)
    
    Parameters:
        - X: sparse matrix
    Returns: 
        - X_scaled: scaled sparse matrix
        - offset: array; vector of the mean/std built column wise
        
        
    z-score standardized X is then given by X_scaled - offset (broadcasted). This can be used outside to implement z-score standardization more efficiently.
    """

    mean=np.array(X.mean(axis=0)).reshape((-1,))
    var=np.array((X.multiply(X)).mean(axis=0)).reshape((-1,))-mean*mean
    std=np.sqrt(var)
    X_scaled=(X.multiply(1/std)).tocsr()
    offset=mean/std
    return X_scaled, offset