import numpy as np
import copy

from .utils import reshape
from .utils import unfold
from .utils import prodTenMat
from .utils import krp_cw
from .utils import batch_krp_cw

'''
def initializeU(n, r, orth=False, return_utu=True);
def recoverCPD(U, Lambda=None);
def innerDensedCP(T, U, Lambda=None);
def innerCP(U, Lambda=None, UtU=None);
def computeALS_CPD(Y, r=1, tol=1e-6, maxitnum=100, decalg='eig', orth_init=False);
'''



def initializeU(n, r, orth=False, return_utu=True):
    '''
    Initialiation of factor matrices for CPD. Uses normal distribution.
    
    Parameters:
        n, array-like
            mode sizes
        r, integer
            CP rank
        orth, bool [default=False]
            either perform orthogonalization (QR) or not
        return_utu, bool [default=True]
            either produce gramians or not
    Returns:
        list of factor matrices of (n[k], r) shape
        [AND - if return_utu - list of their gramians] 
    '''
    d = len(n)
    u = []
    if return_utu:
        utu = []
    for i in range(d):
        tmp = np.random.normal(size=[n[i], r])
        if orth:
            tmp, _ = np.linalg.qr(tmp)
        u.append(tmp)
        if return_utu:
            if orth:
                utu.append(np.eye(r))
            else:
                utu.append(np.dot(tmp.T, tmp))
    if return_utu:
        return u, utu
    return u

def recoverCPD(U, Lambda=None):
    '''
    Recover original tensor from CPD
    
    Parameters:
        U, list of np.ndarrays(ndim=2)
            Factor-matrices of decomposition (with equal number of columns)
        Lambda, np.ndarray(ndim=1) [default=None]
            Values of diagonal tensor
    Returns:
        np.ndarray(ndim=d)
            Recovered tensor
    '''
    n = list(map(lambda x: x.shape[0], U))
    result = batch_krp_cw(U[1:][::-1])
    if Lambda is None:
        result = U[0]@result.T
    else:
        result = (U[0]*Lambda)@result.T
    return reshape(result, n)

    
def innerDensedCP(T, U, Lambda=None):
    '''
    Computes inner product between densed tensor T and CP tensor |[Lambda; U]| (contraction)
    
    Parameters:
        T, np.ndarray
            Densed tensor
        U, list of np.ndarray(ndim=2)
            Factor-matrices of CP tensor
        Lambda, np.ndarray(ndim=1) [default=None]
            Diagonal of core tensor of CPD [default behaviour: [1, ..., 1]]
    Returns:
        float
            Result of contraction: <T, |[Lambda; U_1, \ldots, U_d]|>
    '''
    d = T.ndim
    k = 1
    r = U[0].shape[1]
    IC_stepper = np.sum(r**np.arange(d-1))
    IC_stepper = int(IC_stepper)
    result = prodTenMat(T, U[k], k, 0)
    for k in range(2, d):
        result = prodTenMat(result, U[k], k, 0)
    result = unfold(result, 0)[:, ::IC_stepper]
    if Lambda is None:
        Lambda = np.ones(r)
    result = np.einsum('ij,ij', U[0]*Lambda, result)
    return float(result)

def innerCP(U, Lambda=None, UtU=None):
    '''
    Computes inner product for CP tensor |[Lambda; U]| with itself (self-contraction)
    
    Parameters:
        U, list of np.ndarray(ndim=2)
            Factor-matrices of CP tensor
        Lambda, np.ndarray(ndim=1) [default=None]
            Diagonal of core tensor of CPD [default behaviour: [1, ..., 1]]
        UtU, list of np.ndarray(ndim=2)
            Gramians of factor-matrices
    Returns:
        float
            Result of contraction: <|[Lambda; U_1, \ldots, U_d]|, |[Lambda; U_1, \ldots, U_d]|>
    '''
    r = U[0].shape[1]
    d = len(U)
    if Lambda is None:
        Lambda = np.ones(r)
    if UtU is None:
        UtU = []
        for k in range(d):
            UtU.append(U[k].T@U[k])
    result = (UtU[0]*Lambda)@(np.prod(UtU[1:], axis=0)*Lambda)
    result = np.einsum('ii', result)
    return float(result)
    
def computeALS_CPD(Y, r=1, tol=1e-6, maxitnum=100, init=None, decalg='eig', orth_init=False):
    '''
    ALS. Solves the following problem:
    \min_{U, \|U[:, i]\| = 1} \|Y - X(U, eta)\|_F^2
    
    Parameters:
        Y, np.ndarray
            Tensor to be approximated
        r, integer [default=1]
            CP rank
        tol, float [default=1e-6]
            Square of desired approximation error
        maxitnum, integer [default=100]
            Maximal number of iterations
        
    Returns:
        U, list of np.ndarray(ndim=2)
            Factor-matrices with normalized columns
        eta, np.ndarray(ndim=1)
            Diagonal of core tensor
        funval, list of floats
            Functional values for each iteration (including initial)
    '''
    assert decalg.lower() in ['eig', 'svd']
    n = Y.shape
    d = len(n)
    if init is not None:
        U = init['U']
        if not ('UtU' in init):
            UtU = [x.T@x for x in U]
        else:
            UtU = init['UtU']
        if 'D' in init:
            eta = init['D']
        else:
            eta = np.ones([1, r])
    else:
        U, UtU = initializeU(n, r, orth=orth_init)
        eta = np.ones([1, r])
    normY = np.linalg.norm(Y)
    normY2 = np.power(normY, 2)
    funval = [normY2 - 2*innerDensedCP(Y, U, eta) + innerCP(U, eta, UtU)]
    for itnum in range(maxitnum):
        for k in range(d):
            G = batch_krp_cw((U[:k]+U[k+1:])[::-1]) # -1!
            if decalg.lower() == 'eig':
                G = unfold(Y, k)@G
                UtUk = np.prod(UtU[:k]+UtU[k+1:], axis=0)
                E, V = np.linalg.eig(UtUk)
                E = reshape(E, [1, -1])
                U[k] = G@(V/E)@(V.T)
            else:
                # assuming that \prod_k n_k >> R
                V, S, _ = np.linalg.svd(G.T@G)
                G = unfold(Y, k)@G
                U[k] = ((G@V)/S)@(V.T)
            eta = np.linalg.norm(U[k], axis=0, keepdims=True)
            U[k] /= eta
            UtU[k] = (U[k].T)@U[k]
        current_funval = normY2 - 2*innerDensedCP(Y, U, eta) + innerCP(U, eta, UtU)
        funval.append(current_funval)
        dfv = np.abs(funval[-2] - funval[-1])
        if dfv < tol:
            break
    return U, eta, funval