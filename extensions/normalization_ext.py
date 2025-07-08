import numpy as np
import scipy.sparse as sp


def lp_normalize(X, ord=2):
    """
    normalizes rows of an array, e.g. for l2 or l1 normalization, compatible with sparse matrices
    
    Paramters:
    X: np array or scipy sparse ndarray or matrix
    ord: non-zero int| inf| -inf, default=2 (L2 normalization); order of the norm
    
    Returns:
    X_normalized: np array or scipy sparse csr matrix (depending on type of X); rows of X normalized to ord-norm 1
    """
    if sp.issparse(X):
        X=X.tocsr()
        norms=sp.linalg.norm(X, ord=ord, axis=1)
        norms[norms==0]=1 # prevent dividing by zero (without changing normalization)
        X_normalized=(X.T.multiply(1/norms).tocsr()).T
    else:
        norms=np.linalg.norm(X, ord, axis=1)
        norms[norms==0]=1 # Avoid dividing by zero (without changing normalization)
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
        
        
    z-score standardized X is then given by X_scaled - offset. This can be used outside to implement z-score standardization more efficiently.
    """

    mean=np.array(X.mean(axis=0)).reshape((-1,))
    var=np.array((X.multiply(X)).mean(axis=0)).reshape((-1,))-mean*mean
    std=np.sqrt(var)
    X_scaled=(X.multiply(1/std)).tocsr()
    offset=mean/std
    return X_scaled, offset
    
