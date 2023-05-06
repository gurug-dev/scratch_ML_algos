import numpy as np
from numpy.linalg import eig

def compute_svd_mine(matrix, full_matrices=True):
    """
    Error Handling has not been done, please don't try to break it-- you will succeed
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    if m>=n:
        A = matrix.copy()
    else:
        A = (matrix.T).copy()

    ma = A.shape[0]
    na = A.shape[1]
    
    gram_v = (A.T) @ A
    edv_vals , edv_vectors = eig(gram_v)
    
    idx = np.argsort(edv_vals)[::-1]
    edv_vals = np.abs(edv_vals[idx])
    edv_vectors = edv_vectors[:, idx]
    
    sigma_matrix = np.vstack((np.diag(edv_vals) ** 0.5, np.zeros((ma-na, (na)))))
    sigma_matrix_other = (np.vstack((np.diag((1/edv_vals)) ** 0.5, np.zeros((ma-na, na))))).T

    edu_vectors = ((A @ edv_vectors) @ (sigma_matrix_other)) 
    
    if full_matrices:
        if m>=n:
            return edu_vectors, sigma_matrix, edv_vectors.T
        else:
            return edv_vectors, sigma_matrix.T, edu_vectors.T

    sigma_matrix_reduced = sigma_matrix[:na, :]
    edu_vectors_reduced = edu_vectors[:, :na]
    if full_matrices==False:
        if m>=n:
            return edu_vectors_reduced, edv_vals**0.5, edv_vectors.T
        else:
            return edv_vectors, edv_vals**0.5, edu_vectors_reduced.T