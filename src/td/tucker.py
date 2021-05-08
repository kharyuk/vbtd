import numpy as np
import copy

from .utils import reshape
from .utils import unfold
from .utils import prodTenMat
from .utils import krp_cw
from .utils import batch_krp_cw
from .utils import fast_svd

'''
def initializeAG(n, r, orth=False, return_ata=True);
def recoverTucker(, Lambda=None);
def innerDensedTucker(T, U, Lambda=None);
def innerTucker(U, Lambda=None, UtU=None);
def computeALS_Tucker(Y, r, tol=1e-6, maxitnum=100, decalg='eig', orth_init=False);
'''



def initializeGA(n, r, orth=False, return_g=True, return_a=True, return_ata=True):
    '''
    Initialiation of factor matrices for Tucker decomposition. Uses normal distribution.
    
    Parameters:
        n, array-like
            mode sizes
        r, integer
            Tucker rank
        orth, bool [default=False]
            either perform orthogonalization (QR) or not
        return_utu, bool [default=True]
            either produce gramians or not
    Returns:
        list of factor matrices of (n[k], r) shape
        [AND - if return_ata - list of their gramians] 
    '''
    d = len(n)
    a = []
    if not(return_a and return_g):
        return None, None, None
    if return_ata:
        assert return_a
        ata = []
    else:
        ata = None
    if return_g:
        g = np.random.normal(size=r)
        g /= np.linalg.norm(g)
    else:
        g = None
    if return_a:
        for i in range(d):
            tmp = np.random.normal(size=[n[i], r[i]])
            if orth:
                tmp, _ = np.linalg.qr(tmp)
            a.append(tmp)
            if return_ata:
                if orth:
                    ata.append(np.eye(r))
                else:
                    ata.append(np.dot(tmp.T, tmp))
    else:
        a = None
    return g, a, ata

def recoverTucker(tucker_dict):
    '''
    Recover original tensor from CPD
    
    Parameters:
        tucker_dict
            'A': list of np.ndarrays(ndim=2)
                Factor-matrices of decomposition (with equal number of columns)
            'G': Tucker core
    Returns:
        np.ndarray(ndim=d)
            Recovered tensor
    '''
    #n = list(map(lambda x: x.shape[0], tucker_dict['A']))
    #r = tucker_dict['G']
    d = tucker_dict['G'].ndim
    result = tucker_dict['G'].copy()
    for k in range(d):
        result = prodTenMat(result, tucker_dict['A'], k)
    return result

    
def innerDensedTucker(T, tucker_dict):
    '''
    Computes inner product between densed tensor T and Tucker tensor |[G; A]| (contraction)
    
    Parameters:
        T, np.ndarray
            Densed tensor
        tucker_dict
            'A', list of np.ndarray(ndim=2)
                Factor-matrices of Tucker tensor
            'G', np.ndarray(ndim=d)
                Tucker core
    Returns:
        float
            Result of contraction: <T, |[G; A_1, \ldots, A_d]|>
    '''
    d = T.ndim
    result = T.copy()
    for k in range(d):
        result = prodTenMat(result, tucker_dict['A'][k], k, 0)
    axes = list(range(d))
    result = np.tensordot(result, tucker_dict['G'], axes=(axes, axes))
    return float(result)

def innerTucker(tucker_dict, AtA=None):
    '''
    Computes inner product for Tucker tensor |[G; A]| with itself (self-contraction)
    
    Parameters:
        tucker_dict
            A, list of np.ndarray(ndim=2)
                Factor-matrices of Tucker tensor
            G, np.ndarray(ndim=d)
                Tucker core
        AtA, list of np.ndarray(ndim=2)
            Gramians of factor-matrices
    Returns:
        float
            Result of contraction: <|[G; A_1, \ldots, A_d]|, |[G; A_1, \ldots, A_d]|>
    '''
    d = len(tucker_dict['A'])
    result = tucker_dict['G'].copy()
    for k in range(d):
        if AtA is None:
            result = prodTenMat(result, tucker_dict['A'][k].T@tucker_dict['A'][k], k)
        else:
            result = prodTenMat(result, AtA[k], k)
    axes = list(range(d))
    result = np.tensordot(result, tucker_dict['G'], axes=(axes, axes))
    return float(result)
    
def computeALS_Tucker(T, tucker_dict, tol=1e-6, maxitnum=100, orth_init=False):
    '''
    
    Parameters:
        Y, np.ndarray
            Tensor to be approximated
        tucker_dict
            'r', list of integers
                Tucker ranks
            'A', list of np.ndarrays(ndim=2) [optional]
                initialization of factor matrices
            'G', np.ndarray(ndim=d) [optional]
                initializaion of Tucker core
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
    n = T.shape
    d = len(n)
    G, A, _ = initializeGA(
        n,
        tucker_dict['r'],
        orth=orth_init,
        return_g=not ('G' in tucker_dict),
        return_a=not ('A' in tucker_dict),
        return_ata=False
    )
    if G is not None:
        tucker_dict['G'] = G
    if A is not None:
        tucker_dict['A'] = A
    Ua, Sa, Va = [], [], []
    for k in range(d):
        u, s, v = fast_svd(tucker_dict['A'][k], chi=1.2, eps=1e-8)
        Ua.append(u)
        Sa.append(s)
        Va.append(v)
    normT = np.linalg.norm(T)
    normT2 = np.power(normT, 2)
    funval = [normT2 - 2*innerDensedTucker(T, tucker_dict) + innerTucker(tucker_dict)]
    axes = list(range(d))
    for itnum in range(maxitnum):
        for k in range(d):
            newAk = T.copy()
            for l in range(d):
                if l == k:
                    continue
                newAk = prodTenMat(newAk, (Ua[l]/Sa[l])@Va[l].T, l, 0)
            tmp_axes = axes[:k]+axes[k+1:]
            newAk = np.tensordot(newAk, tucker_dict['G'], axes=(tmp_axes, tmp_axes))
            newAk /= np.linalg.norm(newAk, axis=0, keepdims=True)
            tucker_dict['A'][k] = newAk
            Ua[k], Sa[k], Va[k] = fast_svd(newAk, chi=1.2, eps=1e-8)
        newGk = T.copy()
        for k in range(d):
            newGk = prodTenMat(newGk, (Ua[k]/Sa[k])@Va[k].T, k, 0)
        tucker_dict['G'] = newGk
            
        current_funval = normT2 - 2*innerDensedTucker(T, tucker_dict) + innerTucker(tucker_dict)
        funval.append(current_funval)
        dfv = np.abs(funval[-2] - funval[-1])
        if dfv < tol:
            break
    return tucker_dict, funval