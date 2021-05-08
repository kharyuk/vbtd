import copy
import numpy as np

from .btd_lro_utils import Ckl
from .btd_lro_utils import batch_krp_bcw
from .btd_lro_utils import hadamardCprTCql

from .utils import reshape
from .utils import unfold
from .utils import prodTenMat


'''
def recoverLRO(lro_dict);
def innerDensedLRO(T, lro_dict);
def innerLRO(lro_dict);
def initializeLRO(n, L, P, orth=False);
def computeALS_LRO(Y, lro_params, tol=1e-6, maxitnum=100, decalg='eig');
'''

def recoverLRO(lro_dict):
    L, P, C = lro_dict['L'], lro_dict['P'], lro_dict['C']
    Rl, sL = len(L), int(round(sum(L)))
    D = lro_dict['D']
    n = list(map(lambda x: x.shape[0], C))
    d = len(n)
    result = np.zeros(n)
    for l in range(Rl):
        krpC = batch_krp_bcw(lro_dict, l, skipped=0, reverse=True)
        '''
        krpC = batch_krp_bcw_rm(C[P:], l, reverse=True)
        krpC = np.tile(krpC, [1, L[l]])
        if P > 1:
            krpC = krp_cw(
                krpC,
                batch_krp_bcw_fm(C[1:P], L, l, reverse=True),
                reverse=False
            )
        '''
        krpC = (Ckl(C, 0, l, L, P)*D[l])@krpC.T
        result += reshape(krpC, n)
    return result

def innerDensedLRO(T, lro_dict):
    '''
    Computes inner product between densed tensor T and (Lr, 1) tensor (contraction)
    
    Parameters:
        T, np.ndarray
            Densed tensor
        lro_dict, dict
            Contains parameters of (Lr, 1) decomposition
                P, int
                    Number of full-sized modes (first P modes)
                L, array-like of ints
                    CP ranks of full-sized modes, len(L) = #terms
                C, list of np.ndarray(ndim=2)
                    List of factor-matrices (block-partitioned according to terms)
                D, list of np.ndarray(mdim=2)
                    List of norms of factor-matrices, array size: [1, L_i]
    Returns:
        float
            Result of contraction: <T, T(\theta_{(Lr, 1)})>
            
    ## TODO: replace contraction with block factor matrices by sum of subblock contractions
    '''
    d = T.ndim
    n = list(T.shape)
    L, P, C = lro_dict['L'], lro_dict['P'], lro_dict['C']
    Rl, sL = len(L), int(round(sum(L)))
    D = lro_dict['D']
    result = 0.
    for l in range(Rl):
        tmp = None
        for k in range(P, d):
            if tmp is None:
                tmp = prodTenMat(T, Ckl(C, k, l, L, P).T, k)
            else:
                tmp = prodTenMat(tmp, Ckl(C, k, l, L, P).T, k)
        if P > 1:
            tmp = reshape(tmp, n[:P]+[-1]) # without betas
            for k in range(1, P):
                tmp = prodTenMat(tmp, Ckl(C, k, l, L, P).T, k)
            IC_stepper = np.sum(L[l]**np.arange(P-1))
            IC_stepper = int(IC_stepper)
            tmp = unfold(tmp, 0)[:, ::IC_stepper]
        else:
            tmp = reshape(tmp, [n[0], 1])
            tmp = np.tile(tmp, [1, L[l]])
        result += np.einsum('ij,ij', Ckl(C, 0, l, L, P)*D[l], tmp)
    return result

def innerLRO(lro_dict):
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
    L, P, C = lro_dict['L'], lro_dict['P'], lro_dict['C']
    Rl, sL = len(L), int(round(sum(L)))
    D = lro_dict['D']
    n = list(map(lambda x: x.shape[0], C))
    d = len(n)
    result = 0.
    for l1 in range(Rl):
        for l2 in range(l1+1):
            tmp = np.ones([L[l1], L[l2]])
            for k in range(1, d):
                tmp *= Ckl(C, k, l1, L, P).T@Ckl(C, k, l2, L, P) # numpy can!
            tmp = np.einsum('ij,ij', (Ckl(C, 0, l1, L, P)*D[l1])@tmp, Ckl(C, 0, l2, L, P)*D[l2])
            if l1 == l2:
                result += tmp
            else:
                result += 2.*tmp
    return result

def initializeLRO(n, L, P, orth=False):
    d = len(n)
    C = []
    sumL = int(np.round(np.sum(L)))
    Rl = len(L)
    for k in range(d):
        if k < P:
            tmp = np.random.normal(size=[n[k], sumL])
            if orth:
                ind = 0
                for i in range(Rl):
                    tmp[:, ind:ind+L[i]], _ = np.linalg.qr(tmp[:, ind:ind+L[i]])
                    ind += L[i]
            else:
                tmp /= np.linalg.norm(tmp, axis=0, keepdims=True)
        else:
            tmp = np.random.normal(size=[n[k], Rl])
            if orth:
                tmp, _ = np.linalg.qr(tmp)
            else:
                tmp /= np.linalg.norm(tmp, axis=0, keepdims=True)
        C.append(tmp)
    D = []
    for l in range(Rl):
        #tmp = np.random.normal(size=[1, L[l]])
        tmp = np.ones([1, L[l]])
        D.append(tmp/np.linalg.norm(tmp))
    return C, D

def computeALS_LRO(Y, lro_params, tol=1e-6, maxitnum=100, decalg='eig', orth_init=False):
    '''
    Bounded ALS. Solves the following problem:
    \min_{U, \|U[:, i]\| = 1} \|Y - X(C, eta)\|_F^2
    
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
    L, P = lro_params['L'], lro_params['P']
    sL = int(round(sum(L)))
    Rl = len(L)
    
    if ('C' in lro_params) and ('D' in lro_params):
        C, D = copy.deepcopy(lro_params['C']), copy.deepcopy(lro_params['D'])
    else:
        C, D = initializeLRO(n, L, P, orth=orth_init)
    
    lro_dict = {
        'L': L,
        'P': P,
        'C': C,
        'D': D
    }
    
    normY = np.linalg.norm(Y)
    normY2 = np.power(normY, 2)
    funval = [normY2 - 2*innerDensedLRO(Y, lro_dict) + innerLRO(lro_dict)]
    for itnum in range(maxitnum):
        for l in range(Rl): # terms
            ind = int(round(sum(L[:l])))
            for k in range(d-1, -1, -1): # modes
            #for k in range(d): # modes
                G = batch_krp_bcw(lro_dict, l, skipped=k, reverse=True)
                if decalg.lower() == 'svd':
                    V, S, _ = np.linalg.svd(G.T@G)
                    S = S[S >= 1e-5]
                    V = V[:, :S.size]
                G = unfold(Y, k)@G
                for l2 in range(Rl):
                    if l == l2:
                        if decalg.lower() == 'eig':
                            CklTCkl = hadamardCprTCql(lro_dict, l, l, skipped=k)
                        continue
                    tmp = hadamardCprTCql(lro_dict, l2, l, skipped=k)
                    if k < P:
                        G -= (Ckl(lro_dict['C'], k, l2, L, P)*lro_dict['D'][l2])@tmp
                    else:
                        G -= (np.tile(lro_dict['C'][k][:, l2:l2+1], [1, L[l2]])*lro_dict['D'][l2])@tmp
                if decalg.lower() == 'eig':
                    E, V = np.linalg.eig(CklTCkl)
                    E = reshape(E, [1, -1])
                    tmp = G@(V/E)@(V.T)
                else:
                    tmp = G@(V/S)@V.T
                
                
                if k < P:
                    cur_norm = np.linalg.norm(tmp, axis=0, keepdims=True)
                    lro_dict['C'][k][:, ind:ind+L[l]] = tmp/cur_norm
                    lro_dict['D'][l][:, :] = cur_norm
                else:
                    tmp = np.sum(tmp, axis=1, keepdims=True)
                    cur_norm = np.linalg.norm(tmp)
                    lro_dict['C'][k][:, l:l+1] = tmp/cur_norm
                    lro_dict['D'][l][:, :] = cur_norm
        
        current_funval = innerLRO(lro_dict)-2*innerDensedLRO(Y, lro_dict)+normY2
        funval.append(current_funval)
        dfv = np.abs(funval[-2] - funval[-1])
        if dfv < tol:
            break
    return lro_dict, funval