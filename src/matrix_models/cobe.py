import numpy as np
import copy
from scipy.optimize import brute # grid search
from sklearn.base import BaseEstimator
import numpy_tools

def fk(Ql, zl, a):
    rv = 0
    N = len(Ql)
    for n in range(N):
        rv += np.linalg.norm(Ql[n]@zl[n] - a)
    return rv
    
def cobe(
    Yl,
    eps=1e-8,
    gamma=1e-8,
    maxitnum=20,
    verbose=False,
    maxInnerItNum=500,
    inform=False,
    fast=True,
    dtype=np.float32,
    random_state=None
):
    # QR factorization
    if random_state is not None:
        np.random.seed(random_state)
    N = len(Yl)
    assert np.all([x.dtype == dtype for x in Yl])
    Ql = []
    zl = []
    for n in range(N):
        Qn, Rn = np.linalg.qr(Yl[n])
        Ql.append(Qn)
        zl.append(np.random.normal(size = [Qn.shape[1], 1]).astype(dtype))
    Al = []
    k = 1
    inner_list = []
    fval_list = []
    for itnum in range(maxitnum):
        a = np.random.normal(size = [Ql[0].shape[0], 1]).astype(dtype)
        inner_fvl = []
        for innerItNum in range(maxInnerItNum):
            # a update
            anew = np.zeros([Ql[0].shape[0], 1], dtype=dtype)
            for n in range(N):
                anew += Ql[n]@zl[n]
            anew /= np.linalg.norm(anew)
            # z update
            for n in range(N):
                zl[n] = numpy_tools.reshape_np(Ql[n].T@anew, [-1, 1], order='F', use_batch=False)
            inner_fvl.append(np.linalg.norm(a - anew))
            a = anew.copy()
            if inner_fvl[-1] < gamma:
                break
        inner_list.append(inner_fvl)
        fval_list.append(fk(Ql, zl, a))
        Al.append(a)
        k += 1
        for n in range(N):
            Pmat = np.eye(Ql[n].shape[1], dtype=dtype) - zl[n]@zl[n].T
            Ql[n] = Ql[n]@Pmat
        if verbose:
            print(
                f"Itnum: {itnum+1}/{maxitnum}\t fval: {fval_list[-1]:.3e}\t"
                f"inner_itnum: {innerItNum+1}/{maxInnerItNum}\t inner_diff: {inner_fvl[-1]:.3e}"
            )
        if fval_list[-1] >= eps:
            break
    support = {
        'fval': fval_list,
        'inner_it': inner_list
    }
    if inform:
        return np.hstack(Al), support
    return np.hstack(Al)

# COBEC
def cobec(
    Yl,
    C,
    eps=1e-8,
    maxitnum=30,
    verbose=False,
    inform=False,
    fast=False,
    dtype=np.float32,
    random_state=None
):
    # QR factorization
    if random_state is not None:
        np.random.seed(random_state)
    N = len(Yl)
    assert np.all([x.dtype == dtype for x in Yl])
    Ql = []
    Zl = []
    fval_list = []
    for n in range(N):
        if not fast:
            Qn, Rn = np.linalg.qr(Yl[n])
        else:
            Qn, _, _ = numpy_tools.fast_svd_np(Yl[n], eps=eps)
        Ql.append(Qn)
        Zl.append(np.random.normal(size=[Ql[n].shape[1], C]).astype(dtype))
    A = np.random.uniform(-1, 1, [Ql[0].shape[0], C]).astype(dtype)
    for itnum in range(maxitnum):
        P = np.zeros([Ql[0].shape[0], Zl[0].shape[1]], dtype=dtype)
        for n in range(N):
            P += Ql[n]@Zl[n]
        if not fast:
            U, S, Vt = np.linalg.svd(P)
        else:
            U, S, Vt = numpy_tools.fast_svd_np(P, eps=eps)
            Vt = Vt.T
        # truncated SVD
        Anew = U[:, :C]@Vt[:C, :]
        for n in range(N):
            Zl[n] = Ql[n].T@Anew
        fval_list.append(np.linalg.norm(A - Anew))
        A = Anew.copy()
        if verbose:
            print(f"Itnum: {itnum+1}/{maxitnum}\t fval: {fval_list[-1]:.3e}")
        if fval_list[-1] < eps:
            break
    if inform:
        support = {'fval': fval_list}
        return A, support
    return A

# SORTE
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.305.1290&rep=rep1&type=pdf
def sorte_fun(lamb, p):
    m = lamb.size
    assert (0 <= p < m-2)
    dlamb = -np.diff(lamb)
    var2 = np.var(dlamb[p:])
    if var2 == 0:
        return np.inf
    return np.var(dlamb[p+1:]) / var2

def sorte(lamb):
    m = lamb.size
    p = brute(my_func, list(range(m-2)), full_output=False)
    return p
