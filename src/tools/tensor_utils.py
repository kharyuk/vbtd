import numpy as np
import copy

'''
def prodTenMat(T, A, mode=0, modeA=1);
def krp_cw(a, b, reverse=False);
def batch_krp_cw(A, reverse=False);
def reshape(x, shape);
def unfold(x, mode);
def inversedMatrixKP(X, shapeA, shapeB, docopy=True);
'''


def prodTenMat(T, A, mode=0, modeA=1):
    '''
    Performs tensor by matrix multiplication
    
    Parameters:
        T, np.ndarray
            Tensor
        A, np.ndarray(ndim=2)
            Matrix
        mode, integer [default=0]
            Number of mode for multiplication (contraction) for tensor operand
        modeA, integer [default=1]
            Number of mode for multiplication (contraction) for matrix operand
    Returns:
        np.ndarray
            Result of multiplication with natural order of axes
            
    Example for default parameters:
        T(\alpha_1, \ldots, \alpha_{mode-1}, \alpha_{mode}, \alpha_{mode+1}, \ldots, \alpha_d)
            \times_{mode, modeA}
        A(\beta, \alpha_{mode})
            =
        X(\alpha_1, \ldots, \alpha_{mode-1}, \beta, \alpha_{mode+1}, \ldots, \alpha_d)
    '''
    tmp = np.swapaxes(T, 0, mode)
    result = np.tensordot(A, tmp, axes=([modeA], [0]))
    return np.swapaxes(result, 0, mode)

def krp_cw(a, b, reverse=False):
    '''
    Column-wise Khartri-Rao product
    
    Parameters:
        a, np.ndarray(ndim=2)
            1st operand (matrix)
        b, np.ndarray(ndim=2)
            2nd operand (matrix)
    Result:
        np.ndarray(ndim=2)
            Matrix which columns are Kronecker products of a[:, i] and b[:, i]
    '''
    ncol = a.shape[1]
    assert b.shape[1] == ncol
    result = np.empty([a.shape[0]*b.shape[0], ncol])
    for k in range(ncol):
        if reverse:
            result[:, k] = np.kron(b[:, k], a[:, k])
        else:
            result[:, k] = np.kron(a[:, k], b[:, k])
    return result

def batch_krp_cw(A, reverse=False):
    '''
    Batch column-wise Khartri-Rao product
    
    Parameters:
        A, list of np.ndarray(ndim=2)
            Operands that are matrices with equal number of columns
    Result:
        Matrix which columns are Kronecker product of A[k][:, :i]
    '''
    lenA = len(A)
    assert len(A) > 1, 'batch must contain not less than 2 representatives'
    result = krp_cw(A[0], A[1], reverse)
    for k in range(2, lenA):
        result = krp_cw(result, A[k], reverse)
    return result

def reshape(x, shape):
    '''
    Reshapes input into given shape using F-order
    '''
    return np.reshape(x, shape, order='F')

def unfold(x, mode):
    '''
    Returns unfolding matrix of x:
    
    R = unfold(X^{n_1 \times \ldots \times n_d}, mode);
    R = R^{n_{mode} \times \prod_{k \neq mode} n_k}
    '''
    d = x.ndim
    n = x.shape
    sigma = [mode] + list(range(mode)) + list(range(mode+1, d))
    return reshape(np.transpose(x, sigma), [n[mode], -1])

def inversedMatrixKP(X, shapeA, shapeB, docopy=True):
    '''
    Solves the following problem: \min_{A, B} \|X - A \otimes B \|_F
    
    Example:
        nA = [3, 2]
        nB = [2, 3]
        A = np.random.uniform(size=(nA))
        B = np.random.uniform(size=(nB))
        X = np.kron(A, B)
        C, D = inversedMatrixKP(X, nA, nB)
        print(np.linalg.norm(np.kron(C, D) - X)/np.linalg.norm(X))
    '''
    n = X.shape
    assert np.all(np.array(shapeA)*np.array(shapeB) == np.array(n))
    nA, nB = list(shapeA), list(shapeB)
    nY = [nB[0], nA[0], nB[1], nA[1]]
    nYt = [nA[0]*nA[1], nB[0]*nB[1]]
    if docopy:
        Y = X.copy()
    else:
        Y = X
    Y = reshape(Y, nY)
    permutation = [1, 3, 0, 2]
    Y = np.transpose(Y, permutation)
    Y = reshape(Y, nYt)
    A, S, B = np.linalg.svd(Y, full_matrices=False)
    A = np.sqrt(S[0])*A[:, 0]
    B = np.sqrt(S[0])*B[0, :]
    A = reshape(A, nA)
    B = reshape(B, nB)
    return A, B

def fast_svd(a, chi=1.2, eps=1e-5):
    assert a.ndim == 2
    m, n = a.shape
    if m/n >= chi:
        _, s, v = np.linalg.svd(a.T@a)
        s = np.sqrt(s)
        s = s[s > eps]
        R = len(s)
        v = v[:R, :].T
        u = a@(v/s)
    elif n/m >= chi:
        u, s, _ = np.linalg.svd(a@a.T)
        s = np.sqrt(s)
        s = s[s > eps]
        R = len(s)
        u = u[:, :R]
        v = a.T@(u/s)
    else:
        u, s, v = np.linalg.svd(a, full_matrices=False)
        s = s[s > eps]
        R = len(s)
        u = u[:, :R]
        v = v[:R, :].T
    return u, s, v