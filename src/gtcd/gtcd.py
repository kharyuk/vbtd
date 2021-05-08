#import numba
import scipy.sparse.linalg as spla
#from line_profiler import LineProfiler
import numpy as np
#import autograd
#import autograd.numpy as np

import time
import copy

from functools import reduce

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
#from utils import computeE
#from utils import prodTenMat
#from utils import reshape, krp_cw, unfold

from scipy.sparse.linalg import eigsh

_SQRT_CLOBAL = 1


def mineigen(matvec, tol=1e-5):
    result = eigsh(matvec, k=1, which='SA', tol=tol, return_eigenvectors=False)#, maxiter=75)
    return result[0]

def vec(x):
    return x.flatten(order='F')

def swing(x, k):
    return  np.transpose(x, list(range(1, k+1)) + [0] + list(range(k+1, x.ndim)))

def prodTenMat(T, M, mode_t, mode_m=1):
    assert M.ndim == 2, "Second operand must be a matrix"
    #result = _np.swapaxes(T, 0, mode_t)
    #result = _np.tensordot(M, result, axes=[(mode_m), (0)])
    #result = _np.swapaxes(result, 0, mode_t)
    subT = list(range(T.ndim))
    subR = list(range(T.ndim))
    subR[mode_t] = T.ndim
    subM = [T.ndim, T.ndim]
    subM[mode_m] = subT[mode_t]
    result = np.einsum(T, subT, M, subM, subR)
    return result

def reshape(a, shape):
    return np.reshape(a, shape, order = 'F')

def krp_cw(A, B):
    [I, N] = A.shape
    [J, M] = B.shape
    assert N == M, "Column-wise Khatri-Rao product requiers equal number of columns"
    assert A.dtype == B.dtype
    C = np.zeros([I*J, N], dtype=A.dtype)
    for i in range(N):
        C[:, i] += np.kron(A[:, i], B[:, i])
    return C

def unfold(x, mode, return_sigma=False, reverse=False):
    n = x.shape
    d = x.ndim
    if not reverse:
        sigma = [mode] + list(range(mode)) + list(range(mode+1, d))
    else:
        sigma = [mode] + list(range(d-1, mode, -1)) + list(range(mode-1, -1, -1))
    tmp = np.transpose(x, sigma)
    tmp = reshape(tmp, [n[mode], -1])
    if return_sigma:
        return tmp, sigma
    return tmp

def fastSVD(x, R=None, tol=None, scale=1.5):
    [m, n] = x.shape
    if m >= scale*n:
        tmp = np.dot(x.T, x)
        _, s, vt = np.linalg.svd(tmp)
        if tol is not None:
            csumdiff = np.diff(np.cumsum(s))
            Rtol = csumdiff[csumdiff >= tol].sum()
        if R is not None:
            Rs = R
            if tol is not None:
                Rs = min(Rtol, Rs)
            s = s[:Rs]
            vt = vt[:Rs, :]
        s = s**0.5
        u = np.dot(x, (vt.T / s))
    elif n >= scale*n:
        tmp = np.dot(x, x.T)
        u, s, _ = np.linalg.svd(tmp)
        if tol is not None:
            csumdiff = np.diff(_np.cumsum(s))
            Rtol = csumdiff[csumdiff >= tol].sum()
        if R is not None:
            Rs = R
            if tol is not None:
                Rs = min(Rtol, Rs)
            s = s[:Rs]
            u = u[:, :Rs]
        s = s**0.5
        vt = np.dot(u.T, x).T / s
    else:
        u, s, vt = np.linalg.svd(x)
        if tol is not None:
            csumdiff = np.diff(np.cumsum(s**2.))
            Rtol = csumdiff[csumdiff >= tol].sum()
        if R is not None:
            Rs = R
            if tol is not None:
                Rs = min(Rtol, Rs)
            u = u[:, :Rs]
            s = s[:Rs]
            vt = vt[:Rs, :]
    return u, s, vt

def computeE(L, iden=_SQRT_CLOBAL, dtype=np.float64):
    R = len(L)
    M = int(sum(L))
    E = np.zeros([R, M], dtype=dtype)
    cind = 0
    for t in range(R):
        coef = 1.
        if iden:
            coef = 1./(L[t]**0.5)
        E[t, cind : cind + int(L[t])] = coef*np.ones(int(L[t]), dtype=dtype)
        cind += int(L[t])
    return E

def recover(n, canonical_dict=None, lro_dict=None, tucker_dict=None, dtype=np.float64):
    assert (canonical_dict is not None) or (lro_dict is not None) or (tucker_dict is not None), "recover(): Empty input!"
    d = len(n)
    partCP = None
    if canonical_dict is not None:
        C = canonical_dict['C']
        partCP = C[1].copy()
    if lro_dict is not None:
        B = lro_dict['B']
        P = lro_dict['P']
        L = lro_dict['L']
        M = int(sum(L))
        if 'E' in lro_dict.keys():
            E = lro_dict['E']
        else:
            E = computeE(L)
            lro_dict['E'] = E
        fmc = None
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
        if (fmc is not None) and (fmc[1] is not None) and (P > 1):
            indL = int(sum(L[:fmc[1][1]]))
            partB = np.zeros([n[1], M], dtype=dtype)
            partB[:, :indL] += np.dot(B[1][0], B[1][1].T)
            if indL < M:
                partB[:, indL:] += B[1][2]
        else:
            partB = B[1].copy()
        if P <= 1:
            partB = np.dot(B[1], E)
            
        if canonical_dict is not None:
            tmp = np.zeros([n[1], partCP.shape[1] + partB.shape[1]], dtype=dtype)
            tmp[:, :partCP.shape[1]] += partCP
            tmp[:, partCP.shape[1]:] += partB
            partCP = tmp.copy()
            del tmp, partB
        else:
            partCP = partB.copy()
            del partB
    Rt = 1
    if tucker_dict is not None:
        A = tucker_dict['A']
        G = tucker_dict['G']
        r = tucker_dict['r']
        Rt = r.shape[0]
    
    result = np.zeros(n, dtype=dtype)
    for m in range(Rt):
        if tucker_dict is not None:
            partGm = G[m].copy()
        for k in range(d):
            if tucker_dict is not None:
                partGm = prodTenMat(partGm, A[k][:, int(sum(r[:m, k])) : int(sum(r[:m+1, k]))], k)
            if m == 0:
                if k > 1:
                    if canonical_dict is not None:
                        tmp = C[k].copy()
                    if lro_dict is not None:
                        if k < P:
                            if (fmc is not None) and (fmc[k] is not None):
                                partB = np.zeros([n[k], M], dtype=dtype)
                                indL = int(sum(L[:fmc[k][1]]))
                                partB[:, :indL] += np.dot(B[k][0], B[k][1].T)
                                if indL < M:
                                    partB[:, indL:] += B[k][2]
                            else:
                                partB = B[k].copy()
                        else:
                            partB = np.dot(B[k], E)
                        if canonical_dict is not None:
                            tmp1 = tmp.copy()
                            tmp = np.zeros([n[k], tmp1.shape[1] + partB.shape[1]], dtype=dtype)
                            tmp[:, :tmp1.shape[1]] += tmp1
                            tmp[:, tmp1.shape[1]:] += partB
                            del tmp1, partB
                        else:
                            tmp = partB.copy()
                            del partB
                    if (canonical_dict is not None) or (lro_dict is not None):
                        partCP = krp_cw(tmp, partCP)
        if tucker_dict is not None:
            result += partGm
    if canonical_dict is not None:
        tmp = C[0].copy()
    if lro_dict is not None:
        if P > 0:
            if (fmc is not None) and (fmc[0] is not None):
                indL = int(sum(L[:fmc[0][1]]))
                partB = np.zeros([n[0], M], dtype=dtype)
                partB[:, :indL] += np.dot(B[0][0], B[0][1].T)
                if indL < M:
                    partB[:, indL:] += B[0][2]
            else:
                partB = B[0].copy()
        else:
            partB = np.dot(B[0], E)
        if canonical_dict is not None:
            tmp1 = tmp.copy()
            tmp = np.zeros([n[0], tmp1.shape[1] + partB.shape[1]], dtype=dtype)
            tmp[:, :tmp1.shape[1]] += tmp1
            tmp[:, tmp1.shape[1]:] += partB
            del tmp1, partB
        else:
            tmp = partB.copy()
            del partB
    if (canonical_dict is not None) or (lro_dict is not None):
        result += reshape(np.dot(tmp, partCP.T), n)
    return result

def generate_random_values(x, rtype='normal', dtype=np.float64):
    if rtype == 'normal':
        return np.random.normal(loc=0., scale=1., size=x).astype(dtype)
    if rtype == 'beta':
        return np.random.beta(a=2, b=1, size=x).astype(dtype)
    if rtype == 'binomial':
        return np.random.binomial(n=2, p=0.5, size=x).astype(dtype)
    if rtype == 'chisquare':
        return np.random.chisquare(df=10, size=x).astype(dtype)
    if rtype == 'dirichlet':
        return np.random.dirichlet(alpha=2., size=x).astype(dtype)
    if rtype == 'exponential':
        return np.random.exponential(scale=2., size=x).astype(dtype)
    if rtype == 'f':
        return np.random.f(dfnum=2., dfden=1., size=x).astype(dtype)
    if rtype == 'gamma':
        return np.random.gamma(shape=3., scale=1., size=x).astype(dtype)
    if rtype == 'geometric':
        return np.random.geometric(p=0.8, size=x).astype(dtype)
    if rtype == 'gumbel':
        return np.random.gumbel(loc=0., scale=1., size=x).astype(dtype)
    if rtype == 'hypergeometric':
        return np.random.hypergeometric(ngood=5, nbad=5, nsample=5, size=x).astype(dtype)
    if rtype == 'laplace':
        return np.random.laplace(loc=0., scale=1., size=x).astype(dtype)
    if rtype == 'logistic':
        return np.random.logistic(loc=0., scale=1., size=x).astype(dtype)
    if rtype == 'lognormal':
        return np.random.lognormal(mean=0., sigma=1., size=x).astype(dtype)
    if rtype == 'logseries':
        return np.random.logseries(p=0.5, size=x).astype(dtype)
    if rtype == 'multinomial':
        return np.random.multinomial(n=5., pvals=5., size=x).astype(dtype)
    if rtype == 'negative_binomial':
        return np.random.negative_binomial(n=5., p=0.5, size=x).astype(dtype)
    if rtype == 'pareto':
        return np.random.pareto(a=5., size=x).astype(dtype)
    if rtype == 'poisson':
        return np.random.poisson(lam=0.5, size=x).astype(dtype)
    if rtype == 'power':
        return np.random.power(a=5., size=x).astype(dtype)
    if rtype == 'rayleigh':
        return np.random.rayleigh(scale=1., size=x).astype(dtype)
    if rtype == 'standart_cauchy':
        return np.random.standard_cauchy(size=x).astype(dtype)
    if rtype == 'standart_t':
        return np.random.standard_t(df=5., size=x).astype(dtype)
    if rtype == 'beta':
        return np.random.triangular(left=-1., mode=0., right=-1, size=x).astype(dtype)
    if rtype == 'uniform':
        return np.random.uniform(low=0, high=0.5, size=x).astype(dtype)
    if rtype == 'vonmises':
        return np.random.vonmises(mu=0., kappa=1., size=x).astype(dtype)
    if rtype == 'wald':
        return np.random.wald(mean=0., scale=1., size=x).astype(dtype)
    if rtype == 'weibul':
        return np.random.weibull(a=5., size=x).astype(dtype)
    if rtype == 'zipf':
        return np.random.zipf(a=5., size=x).astype(dtype)
    raise NotImplemented


def initializeCBT(
    n,
    canonical_param=None,
    lro_param=None,
    tucker_param=None,
    rtype='uniform',
    normalize=True,
    dtype=np.float64
):
    '''
    Tucker blocks
        r = rank parameters
    Canonical block
        Rc = rank
    (Lr, 1) blocks
        L = full-mode sizes
        P = number of full modes
    '''
    assert (
        (canonical_param is not None) or
        (lro_param is not None) or
        (tucker_param is not None)
    ), "initializeCBT(): Empty input!"
    d = len(n)
    if canonical_param is not None:
        Rc = canonical_param['Rc']
    if lro_param is not None:
        L = lro_param['L']
        P = lro_param['P']
        Rl = len(L)
        M = int(sum(L))
        if 'fullModesConfig' in lro_param.keys():
            fmc = copy.deepcopy(lro_param['fullModesConfig'])
        else:
            fmc = None
    Rt = 0
    if tucker_param is not None:
        r = tucker_param['r']
        Rt = r.shape[0]
    
    Cres = []
    Bres = []
    Ares = []
    Gres = []
    for k in range(max(Rt, d)):
        if k < d:
            if tucker_param is not None:
                shapeAk = [n[k], int(sum(r[:, k]))]
                Ares.append(generate_random_values(shapeAk, rtype, dtype))
                Ares[k] = Ares[k].astype(dtype)
                if normalize:
                    Ares[-1] /= np.linalg.norm(Ares[-1])#, axis=0, keepdims=True)
                #print(Ares[-1].dtype)
            if lro_param is not None:
                if k < P:
                    if (fmc is not None) and (fmc[k] is not None):
                        indL = int(sum(L[:fmc[k][1]]))
                        tmpBk = [
                            generate_random_values([n[k], fmc[k][0]], rtype).astype(dtype),
                            generate_random_values([indL, fmc[k][0]], rtype).astype(dtype)
                        ]
                        if fmc[k][1] < Rl:
                            tmpBk.append( generate_random_values([n[k], M-indL], rtype).astype(dtype) )
                        else:
                            tmpBk.append(None)
                        if normalize:
                            tmpBk[0], _ = np.linalg.qr(tmpBk[0])
                            tmpBk[1] /= np.linalg.norm(tmpBk[1])
                            if fmc[k] < Rl:
                                tmpBk[2] /= np.linalg.norm(tmpBk[2])
                            
                    else:
                        tmpBk = generate_random_values([n[k], M], rtype, dtype).astype(dtype)
                        if normalize:
                            tmpBk /= np.linalg.norm(tmpBk)
                        #print(tmpBk[-1].dtype)
                else:
                    tmpBk = generate_random_values([n[k], Rl], rtype, dtype).astype(dtype)
                    if normalize:
                        tmpBk /= np.linalg.norm(tmpBk)
                    #print(tmpBk[-1].dtype)
                Bres.append(tmpBk)
            if canonical_param is not None:
                shapeCk = [n[k], Rc]
                Cres.append(generate_random_values(shapeCk, rtype, dtype).astype(dtype))
                #Cres[k] = Cres[k].astype(dtype)
                if normalize:
                    Cres[-1] /= np.linalg.norm(Cres[-1])#, axis=0, keepdims=True)
                #print(Cres[-1].dtype)
        if k < Rt:
            Gres.append(generate_random_values(r[k, :], rtype, dtype).astype(dtype))
            #Gres[k] = Gres[k].astype(dtype)
            if normalize:
                Gres[-1] /= np.linalg.norm(Gres[-1])
            #print(Gres[-1].dtype)
    return Cres, Bres, Ares, Gres

def fcore2fvec(
    n,
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None
):
    assert (
        (canonical_dict is not None) or
        (lro_dict is not None) or
        (tucker_dict is not None)
    ), "fcore2fvec(): Empty input!"
    d = len(n)
    indC = 0
    sizeC = 0
    indB = 0
    sizeB = 0
    indA = 0
    sizeA = 0
    indG = 0
    sizeG = 0
    dtype = None
    if canonical_dict is not None:
        Rc = canonical_dict['Rc']
        C = canonical_dict['C']
        sizeC = Rc*int(sum(n))
        dtype = C[0].dtype
    if lro_dict is not None:
        B = lro_dict['B']
        if dtype is None:
            dtype = B[0].dtype
        else:
            assert dtype == B[0].dtype
        L = lro_dict['L']
        P = lro_dict['P']
        if 'E' in lro_dict.keys():
            E = lro_dict['E']
        else:
            E = computeE(L)
            lro_dict['E'] = E
        Rl = len(L)
        M = int(sum(L))
        indB = sizeC
        fmc = None
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
        if P > 0:
            if fmc is not None:
                fmcInd = np.where(np.array(fmc) != None)[0]
                #nf1 = len(fmcInd)
                #indL = sum(L[fmc[k][1]:])
                #sizeB += M*(P-nf1)
                for k in range(P):
                    if not (k in fmcInd):
                        sizeB += M*n[k]
                    else:
                        indL = int(sum(L[:fmc[k][1]]))
                        sizeB += (n[k]+indL)*fmc[k][0]
                        if indL != M:
                            sizeB += (M-indL)*n[k]
            else:
                sizeB += M*int(sum(n[:P]))
        if P < d:
            sizeB += Rl*int(sum(n[P:]))

    Rt = 0
    if tucker_dict is not None:
        A = tucker_dict['A']
        G = tucker_dict['G']
        if dtype is None:
            dtype = A[0].dtype
        else:
            assert dtype == A[0].dtype
        r = tucker_dict['r']
        Rt = r.shape[0]
        indA = sizeC + sizeB
        sizeA = int(np.inner(n, np.sum(r, axis=0)))
        indG = indA + sizeA
        sizeG = int(np.sum(np.prod(r, axis=1)))
    size = int(sizeC + sizeB + sizeA + sizeG)
    result = np.zeros(size, dtype=dtype)
    
    for k in range(max(Rt, d)):
        if k < d:
            if sizeC > 0:
                offsetC = int(n[k]*Rc)
                result[indC : indC+offsetC] = vec(C[k])
                indC += offsetC
            if sizeB > 0:
                if (k < P) and (fmc is not None) and (fmc[k] is not None):
                    indL = int(sum(L[:fmc[k][1]]))
                    offsetB = int(n[k]*fmc[k][0])
                    result[indB : indB+offsetB] = vec(B[k][0])
                    indB += offsetB
                    offsetB = int(indL*fmc[k][0])
                    result[indB : indB+offsetB] = vec(B[k][1])
                    indB += offsetB
                    if indL < M:
                        offsetB = int((M-indL)*n[k])
                        result[indB : indB+offsetB] = vec(B[k][2])
                        indB += offsetB
                else:
                    if k < P:
                        offsetB = int(n[k]*M)
                    else:
                        offsetB = int(n[k]*Rl)
                    result[indB : indB+offsetB] = vec(B[k])
                    indB += offsetB
            if sizeA > 0:
                offsetA = int(n[k]*np.sum(r[:, k]))
                result[indA : indA+offsetA] = vec(A[k])
                indA += offsetA
        if k < Rt:
            # if we here than Rt > 0
            offsetG = int(np.prod(r[k, :]))
            result[indG : indG+offsetG] = vec(G[k])
            indG += offsetG 
    return result

def fvec2fcore(
    x,
    n,
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None,
    full_result=True
):
    assert (
        (canonical_dict is not None) or
        (lro_dict is not None) or
        (tucker_dict is not None)
    ), "fvec2fcore(): Empty input!"
    d = len(n)
    indC = 0
    sizeC = 0
    indB = 0
    sizeB = 0
    indA = 0
    sizeA = 0
    indG = 0
    sizeG = 0
    resC = None
    if canonical_dict is not None:
        resC = []
        Rc = canonical_dict['Rc']
        sizeC = Rc*int(sum(n))
    resB = None
    if lro_dict is not None:
        resB = []
        L = lro_dict['L']
        P = lro_dict['P']
        if 'E' in lro_dict.keys():
            E = lro_dict['E']
        else:
            E = computeE(L)
            lro_dict['E'] = E
        Rl = len(L)
        M = int(sum(L))
        indB = sizeC
        fmc = None
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
        if P > 0:
            if fmc is not None:
                fmcInd = np.where(np.array(fmc) != None)[0]
                #nf1 = len(fmcInd)
                #indL = sum(L[fmc[k][1]:])
                #sizeB += M*(P-nf1)
                for k in range(P):
                    if not (k in fmcInd):
                        sizeB += M*n[k]
                    else:
                        indL = int(sum(L[:fmc[k][1]]))
                        sizeB += (n[k]+indL)*fmc[k][0]
                        if indL != M:
                            sizeB += (M-indL)*n[k]
            else:
                sizeB += M*int(sum(n[:P]))
        if P < d:
            sizeB += Rl*int(sum(n[P:]))
    Rt = 0
    resA = None
    resG = None
    if tucker_dict is not None:
        resA = []
        resG = []
        r = tucker_dict['r']
        Rt = r.shape[0]
        indA = sizeC + sizeB
        sizeA = int(np.inner(n, np.sum(r, axis=0)))
        indG = indA + sizeA
        sizeG = int(np.sum(np.prod(r, axis=1)))
    for k in range(max(Rt, d)):
        if k < d:
            if sizeC > 0:
                offsetC = int(n[k]*Rc)
                tmp = x[indC : indC+offsetC].copy()
                resC.append(reshape(tmp, [n[k], Rc]))
                indC += offsetC
            if sizeB > 0:
                if (k < P) and (fmc is not None) and (fmc[k] is not None):
                    indL = int(sum(L[:fmc[k][1]]))
                    Bk = []
                    offsetB = int(n[k]*fmc[k][0])
                    shapeBk = [n[k], fmc[k][0]]
                    Bk.append( reshape(x[indB : indB+offsetB], shapeBk) )
                    indB += offsetB
                    offsetB = int(indL*fmc[k][0])
                    shapeBk = [indL, fmc[k][0]]
                    Bk.append( reshape(x[indB : indB+offsetB], shapeBk) )
                    indB += offsetB
                    if indL < M:
                        offsetB = int((M-indL)*n[k])
                        shapeBk = [n[k], M-indL]
                        Bk.append( reshape(x[indB : indB+offsetB], shapeBk) )
                        indB += offsetB
                    else:
                        Bk.append(None)
                    resB.append(Bk)
                else:
                    if k < P:
                        offsetB = int(n[k]*M)
                        shapeBk = [n[k], M]
                    else:
                        offsetB = int(n[k]*Rl)
                        shapeBk = [n[k], Rl]
                    tmp = x[indB : indB+offsetB].copy()
                    resB.append(reshape(tmp, shapeBk))
                    indB += offsetB
            if sizeA > 0:
                offsetA = int(n[k]*np.sum(r[:, k]))
                tmp = x[indA : indA+offsetA].copy()
                resA.append(reshape(tmp, [n[k], int(np.sum(r[:, k]))]))
                indA += offsetA
        if k < Rt:
            # if we here than Rt > 0
            offsetG = int(np.prod(r[k, :]))
            tmp = x[indG : indG+offsetG].copy()
            resG.append(reshape(tmp, r[k, :]))
            indG += offsetG
    if full_result:
        return resC, resB, resA, resG
    result = []
    if sizeC > 0:
        result.append(resC)
    if sizeB > 0:
        result.append(resB)
    if (sizeA > 0) and (sizeG > 0):
        result.append(resA)
        result.append(resG)
    return result

def jacTCD_matvec(
    x,
    Th,
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None,
    return_vector=False,
    sqrt=_SQRT_CLOBAL
):
    return gradientTCD(x, Th, canonical_dict, lro_dict, tucker_dict, return_vector, sqrt)

def gradientTCD(
    x,
    Th,
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None,
    return_vector=False,
    sqrt=_SQRT_CLOBAL
):
    #, Th, C, A, G, L, r):
    d = Th.ndim
    n = Th.shape
    dtype = x.dtype
    Ch, Bh, Ah, Gh = fvec2fcore(x, n, canonical_dict, lro_dict, tucker_dict, full_result=True)
    C, B, A, G, lhD = None, None, None, None, None
    Cres = None
    cFlag = False
    if canonical_dict is not None:
        Rc = canonical_dict['Rc']
        #C = canonical_dict['C']
        Cres = [None]*d
        cFlag = True
    Bres = None
    Rl = 1
    lFlag = False
    if lro_dict is not None:
        lFlag = True
        L = lro_dict['L']
        P = lro_dict['P']
        #B = lro_dict['B']
        if 'E' in lro_dict.keys():
            E = lro_dict['E']
        else:
            E = computeE(L)
            lro_dict['E'] = E
        fmc = None
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
        Rl = len(L)
        M = int(sum(L))
        Bres = [None]*d
        lD = {}
        lD['L'] = L
        lD['P'] = P
        if fmc is not None:
            lD['fullModesConfig'] = fmc
        lD['B'] = Bh
    Rt = 1
    Ares = None
    Gres = None
    tFlag = False
    if tucker_dict is not None:
        tFlag = True
        #A = tucker_dict['A']
        #G = tucker_dict['G']
        r = tucker_dict['r']
        Rt = len(Gh)
        Ares = [None]*d
        Gres = [None]*Rt
    
    for k in range(max(d, Rt)):
        if k < d:
            if cFlag:
                #Cres[k] = np.zeros(C[k].shape)
                Cres[k] = np.zeros(Ch[k].shape, dtype=dtype)
            if lFlag:
                if (k < P) and (fmc is not None) and (fmc[k] is not None):
                    Brk = []
                    indL = int(sum(L[:fmc[k][1]]))
                    Brk.append(np.zeros([n[k], fmc[k][0]], dtype=dtype))
                    Brk.append(np.zeros([indL, fmc[k][0]], dtype=dtype))
                    if indL < M:
                        Brk.append(np.zeros([n[k], M-indL], dtype=dtype))
                    else:
                        Brk.append(None)
                    Bres[k] = Brk
                else:
                    #Bres[k] = np.zeros(B[k].shape)
                    Bres[k] = np.zeros(Bh[k].shape, dtype=dtype)
            if tFlag:
                #Ares[k] = np.zeros(A[k].shape)
                Ares[k] = np.zeros(Ah[k].shape, dtype=dtype)
        if k < Rt:
            if tFlag:
                #Gres[k] = np.zeros(G[k].shape)
                Gres[k] = np.zeros(Gh[k].shape, dtype=dtype)
    
    if cFlag:
        IC_stepper = np.sum(Rc**np.arange(d-1))
        IC_stepper = int(IC_stepper)

            
    
    # CCk = k-th CP factor on CP
    # CBsk = k-th CP factor on (Lr, 1)
    # CTmk = k-th CP factor on m-th Tucker
    # CDk = k-th CP factor on Densed
    
    # BCsk = k-th (Lr, 1) factor of s-th term on CP
    # BBss2k = k-th (Lr, 1) factor of s-th term on s2-th (Lr, 1)
    # BTsmk = k-th (Lr, 1) factor of s-th term on m-th Tucker
    # BDk = k-th (Lr, 1) factor of s-th term on Densed
    
    # TCmk = k-th factor of m-th Tucker on CP
    # TCm = core of m-th Tucker on CP
    # TBsmk = k-th factor of m-th Tucker on (Lr, 1)
    # TBsm = core of m-th Tucker on (Lr, 1)
    # TDmk = k-th factor of m-th Tucker on Densed
    # TDm = core of m-th Tucker on Densed
    # TTmm2k = k-th factor of m-th Tucker on m2-th Tucker
    # TTmm2 = core of m-th Tucker on m2-th Tucker 
    
    # I use here explicit notation for conditions because it is more convenient to check
    for m1 in range(Rt):
        for m2 in range(m1+1):
            for s1 in range(Rl):
                for s2 in range(s1+1):
                    for k in range(d):
                        if cFlag and (m1 == 0) and (m2 == 0) and (s1 == 0) and (s2 == 0):
                            CCk = np.ones([Rc, Rc], dtype=dtype)
                            CDk = Th.copy()
                        if cFlag and lFlag and (m1 == 0) and (m2 == 0) and (s2 == 0):
                            CBsk = np.ones([L[s1], Rc], dtype=dtype)
                        if cFlag and tFlag and (m2 == 0) and (s1 == 0) and (s2 == 0):
                            CTmk = Gh[m1].copy()
                            if k == 0:
                                CTm = np.dot(Atk(Ah, m1, 1, r).T, Ch[1])
                        if lFlag and (m1 == 0) and (m2 == 0):
                            BBss2k = np.ones([L[s1], L[s2]], dtype=dtype)
                            if s2 == 0:
                                BDsk = Th.copy()
                        if lFlag and tFlag and (m2 == 0) and (s2 == 0):
                            BTsmk = Gh[m1].copy()
                            if (k == 0):
                                if P > 1:
                                    TBsm = np.dot(Atk(Ah, m1, 1, r).T, Bkl(lD, 1, s1, full=1, docopy=0))
                                else:
                                    if sqrt:
                                        TBsm = np.dot(
                                            Atk(Ah, m1, 1, r).T,
                                            np.tile(Bh[1][:, s1:s1+1]/np.sqrt(L[s1]), [1, L[s1]])
                                        ) # sqrt/
                                    else:
                                        TBsm = np.dot(Atk(Ah, m1, 1, r).T, np.tile(Bh[1][:, s1:s1+1], [1, L[s1]]))
                        if tFlag and (m2 == 0) and (s1 == 0) and (s2 == 0):
                            TDmk = Th.copy()
                        if tFlag and (s1 == 0) and (s2 == 0):
                            TTmm2k = Gh[m2].copy()
                            if (m1 != m2) and (k == 0):############################
                                TTmm2 = Gh[m1].copy()
                        for p in range(d):
                            if k == p:
                                if (k == 0) and (p > 1):
                                    #if cFlag and tFlag and (m2 == 0) and (s1 == 0) and (s2 == 0):
                                    #    CTm = krp_cw(np.dot(Atk(Ah, m1, p, r).T, Ch[p]), CTm)
                                    if lFlag and tFlag and (m2 == 0) and (s2 == 0):
                                        if p < P:
                                            tmp = np.dot(Atk(Ah, m1, p, r).T, Bkl(lD, p, s1, full=1, docopy=0))
                                        else:
                                            if sqrt:
                                                tmp = np.dot(
                                                    Atk(Ah, m1, p, r).T,
                                                    np.tile(Bh[p][:, s1:s1+1]/np.sqrt(L[s1]), [1, L[s1]])
                                                )
                                            else:
                                                tmp = np.dot(Atk(Ah, m1, p, r).T, np.tile(Bh[p][:, s1:s1+1], [1, L[s1]]))
                                        TBsm = krp_cw(tmp, TBsm)
                                        del tmp
                                continue
                            if cFlag and (m1 == 0) and (m2 == 0) and (s1 == 0) and (s2 == 0):
                                CCk *= np.dot(Ch[p].T, Ch[p])
                                CDk = prodTenMat(CDk, Ch[p].T, p)
                            if cFlag and lFlag and (m1 == 0) and (m2 == 0) and (s2 == 0):
                                if p < P:
                                    tmp = np.dot(Bkl(lD, p, s1, full=1, docopy=0).T, Ch[p])
                                else:
                                    if sqrt:
                                        tmp = np.dot(np.tile(Bh[p][:, s1:s1+1]/ (L[s1]**0.5), [1, L[s1]]).T , Ch[p]) # sqrt 
                                    else:
                                        tmp = np.dot(np.tile(Bh[p][:, s1:s1+1], [1, L[s1]]).T , Ch[p]) # sqrt 
                                CBsk *= tmp
                            if cFlag and tFlag and (m2 == 0) and (s1 == 0) and (s2 == 0):
                                CTmk = prodTenMat(CTmk, np.dot(Ch[p].T, Atk(Ah, m1, p, r)), p)
                                if (k == 0) and (p > 1):
                                    CTm = krp_cw(np.dot(Atk(Ah, m1, p, r).T, Ch[p]), CTm)
                            if lFlag and (m1 == 0) and (m2 == 0):
                                if p < P:
                                    BBss2k *= np.dot(
                                        Bkl(lD, p, s1, full=1, docopy=0).T,
                                        Bkl(lD, p, s2, full=1, docopy=0)
                                    )
                                    if s2 == 0:
                                        BDsk = prodTenMat(BDsk, Bkl(lD, p, s1, full=1, docopy=0).T, p)
                                else:
                                    if sqrt:
                                        BBss2k *= np.dot(
                                            Bh[p][:, s1:s1+1].T/(L[s1]**0.5),
                                            Bh[p][:, s2:s2+1]/(L[s2]**0.5) # sqrt
                                        )
                                    else:
                                        BBss2k *= np.dot(
                                            np.tile(Bh[p][:, s1:s1+1], [1, L[s1]]).T,
                                            np.tile(Bh[p][:, s2:s2+1], [1, L[s2]])
                                        )
                                    if s2 == 0:
                                        if sqrt:
                                            BDsk = prodTenMat(
                                                BDsk,
                                                Bh[p][:, s1:s1+1].T/(L[s1]**0.5),
                                                p
                                            ) ## sqrt
                                        else:
                                            BDsk = prodTenMat(
                                                BDsk,
                                                Bh[p][:, s1:s1+1].T,
                                                #np.tile(Bh[p][:, s1:s1+1], [1,L[s1]]).T,
                                                p
                                            )
                            if lFlag and tFlag and (m2 == 0) and (s2 == 0):
                                tmp = Atk(Ah, m1, p, r)
                                if p < P:
                                    tmp = np.dot(Bkl(lD, p, s1, full=1, docopy=0).T, tmp)
                                else:
                                    if sqrt:
                                        tmp = np.dot(Bh[p][:, s1:s1+1].T/(L[s1]**0.5), tmp) ## sqrt
                                    else:
                                        tmp = np.dot(Bh[p][:, s1:s1+1].T, tmp) ## sqrt/np.sqrt(L[s1])
                                BTsmk = prodTenMat(BTsmk, tmp, p)
                                del tmp
                                if (k == 0) and (p > 1):
                                    if p < P:
                                        tmp = np.dot(Atk(Ah, m1, p, r).T, Bkl(lD, p, s1, full=1, docopy=0))
                                    else:
                                        if sqrt:                                            
                                            tmp = np.dot(
                                                Atk(Ah, m1, p, r).T,
                                                np.tile(Bh[p][:, s1:s1+1], [1, L[s1]])/(L[s1]**0.5)
                                            )
                                        else:
                                            tmp = np.dot(
                                                Atk(Ah, m1, p, r).T,
                                                np.tile(Bh[p][:, s1:s1+1], [1, L[s1]])
                                            )
                                    TBsm = krp_cw(tmp, TBsm)
                                    
                            if tFlag and (m2 == 0) and (s1 == 0) and (s2 == 0):
                                TDmk = prodTenMat(TDmk, Atk(Ah, m1, p, r).T, p)
                            if tFlag and (s1 == 0) and (s2 == 0):
                                tmp = np.dot(Atk(Ah, m1, p, r).T,  Atk(Ah, m2, p, r))
                                TTmm2k = prodTenMat(TTmm2k, tmp, p)
                                if (m1 != m2) and (k == 0):
                                    TTmm2 = prodTenMat(TTmm2, tmp.T, p)
                        if cFlag and (m1 == 0) and (m2 == 0) and (s1 == 0) and (s2 == 0):
                            CDk = unfold(CDk, k)[:, ::IC_stepper]
                            Cres[k] += np.dot(Ch[k], CCk) - CDk
                            del CCk, CDk
                        if cFlag and lFlag and (m1 == 0) and (m2 == 0) and (s2 == 0):                            
                            #kpf = k < P
                            #IB_stepper = np.sum(L[s1]**np.arange(P-kpf))
                            #IB_stepper = int(IB_stepper)
                            #axes = range(k) + [d] + range(k+1, d)
                            #axes2 = range(k) + [d+1] + range(k+1, d)
                            #I2 = np.zeros([L[s1]]*d)
                            #np.fill_diagonal(I2, 1.)
                            #print CBsk.T - np.einsum(I, axes, I2, axes2)
                            BCsk = np.dot(Ch[k], CBsk.T) #/ np.sqrt(L[s1]**(d-P))
                            if k < P:
                                CBsk = np.dot(Bkl(lD, k, s1, full=1, docopy=0), CBsk)
                                #CBsk = np.dot(Bkl(lD, k, s1, full=1, docopy=0), CBsk.T)
                            else:
                                if sqrt:
                                    CBsk = np.dot(np.tile(Bh[k][:, s1:s1+1]/(L[s1]**0.5), [1, L[s1]]), CBsk)
                                else:
                                    CBsk = np.dot(np.tile(Bh[k][:, s1:s1+1], [1, L[s1]]), CBsk)
                                #CBsk = np.dot(np.tile(Bh[k][:, s1:s1+1]/ np.sqrt(L[s1]), [1, L[s1]]), CBsk.T) ## 
                            Cres[k] += CBsk 
                            del CBsk
                            #BCsk /= np.sqrt(np.prod(L[P:]))
                            if (fmc is not None) and (fmc[k] is not None): #### k>=P must be None!!!
                                if s1 < fmc[k][1]:
                                    Usk, Wsk = Bkl(lD, k, s1, full=0, docopy=0)
                                    Bres[k][0] += np.dot(BCsk, Wsk)
                                    ind = int(sum(L[:s1]))
                                    Bres[k][1][ind:ind+L[s1], :] += np.dot(Usk.T, BCsk).T
                                else:
                                    ind = int(sum(L[fmc[k][1]:s1]))
                                    Bres[k][2][:, ind:ind+L[s1]] += BCsk
                            else:
                                ind = int(sum(L[:s1]))
                                if k < P:
                                    Bres[k][:, ind:ind+L[s1]] += BCsk
                                else:
                                    tmp = np.sum(BCsk, axis=1, keepdims=1)
                                    if sqrt:
                                        tmp /= (L[s1]**0.5)
                                    Bres[k][:, s1:s1+1] += tmp
                                    '''
                                    if sqrt:
                                        Bres[k][:, s1:s1+1] += np.sum(BCsk, axis=1, keepdims=1)/np.sqrt(L[s1])## sqrt 
                                    else:
                                        Bres[k][:, s1:s1+1] += np.sum(BCsk, axis=1, keepdims=1)
                                    '''
                            del BCsk
                        if cFlag and tFlag and (m2 == 0) and (s1 == 0) and (s2 == 0):
                            ind1 = int(np.sum(r[:m1, k]))
                            ind2 = int(np.sum(r[:m1+1, k]))
                            CTmk = unfold(CTmk, k)[:, ::IC_stepper]
                            Ares[k][:, ind1:ind2] += np.dot(Ch[k], CTmk.T)
                            Cres[k] += np.dot(Atk(Ah, m1, k, r), CTmk)
                            if (k == 0):
                                CTm = np.dot(np.dot(Ch[0].T, Atk(Ah, m1, 0, r)).T, CTm.T)
                                CTm = reshape(CTm, Gh[m1].shape)
                                Gres[m1] += CTm
                                del CTm
                        if lFlag and (m1 == 0) and (m2 == 0):
                            BBs2sk = None
                            kpf = k < P
                            IB_stepper = np.sum(L[s1]**np.arange(P-kpf))
                            IB_stepper = int(IB_stepper)
                            if s2 == 0:
                                BDsk = unfold(BDsk, k)[:, ::IB_stepper]
                            if k < P:
                                if (s1!=s2):
                                    BBs2sk = np.dot(Bkl(lD, k, s1, full=1, docopy=0), BBss2k)
                                BBss2k = np.dot(Bkl(lD, k, s2, full=1, docopy=0), BBss2k.T)
                            else:
                                if (s1!=s2):
                                    BBs2sk = np.dot(np.tile(Bh[k][:, s1:s1+1], [1, L[s1]]), BBss2k)
                                    if sqrt:
                                        BBs2sk /= (L[s1]**0.5)
                                BBss2k = np.dot(np.tile(Bh[k][:, s2:s2+1], [1, L[s2]]), BBss2k.T)
                                if sqrt:
                                    BBss2k /= (L[s2]**0.5)
                                '''
                                if sqrt:
                                    if (s1!=s2):
                                        BBs2sk = np.dot(np.tile(Bh[k][:, s1:s1+1]/np.sqrt(L[s1]), [1, L[s1]]), BBss2k)
                                    BBss2k = np.dot(np.tile(Bh[k][:, s2:s2+1]/np.sqrt(L[s2]), [1, L[s2]]), BBss2k.T)
                                else:
                                    if (s1!=s2):
                                        BBs2sk = np.dot(np.tile(Bh[k][:, s1:s1+1], [1, L[s1]]), BBss2k)
                                    BBss2k = np.dot(np.tile(Bh[k][:, s2:s2+1], [1, L[s2]]), BBss2k.T)
                                '''
                            if (fmc is not None) and (fmc[k] is not None): #### k>=P must be None!!!
                                if s1 < fmc[k][1]:
                                    Usk, Wsk = Bkl(lD, k, s1, full=0, docopy=0)
                                    Bres[k][0] += np.dot(BBss2k, Wsk)
                                    ind = int(sum(L[:s1]))
                                    Bres[k][1][ind:ind+L[s1], :] += np.dot(Usk.T, BBss2k).T
                                    if s2 == 0:
                                        Bres[k][0] -= np.dot(BDsk, Wsk)
                                        Bres[k][1][ind:ind+L[s1], :] -= np.dot(Usk.T, BDsk).T
                                else:
                                    ind = int(sum(L[fmc[k][1]:s1]))
                                    Bres[k][2][:, ind:ind+L[s1]] += BBss2k
                                    if s2 == 0:
                                        Bres[k][2][:, ind:ind+L[s1]] -= BDsk
                                if (s2!=s1):
                                    if s2 < fmc[k][1]:
                                        Usk, Wsk = Bkl(lD, k, s2, full=0, docopy=0)
                                        Bres[k][0] += np.dot(BBs2sk, Wsk)
                                        ind = int(sum(L[:s2]))
                                        Bres[k][1][ind:ind+L[s2], :] += np.dot(Usk.T, BBs2sk).T
                                    else:
                                        ind = int(sum(L[fmc[k][1]:s2]))
                                        Bres[k][2][:, ind:ind+L[s2]] += BBs2sk
                            else:
                                ind1 = int(sum(L[:s1]))
                                ind2 = int(sum(L[:s2]))
                                if k < P:
                                    Bres[k][:, ind1:ind1+L[s1]] += BBss2k
                                    if s1!=s2:
                                        Bres[k][:, ind2:ind2+L[s2]] += BBs2sk
                                    if s2 == 0:
                                        Bres[k][:, ind1:ind1+L[s1]] -= BDsk
                                else:
                                    tmp = np.sum(BBss2k, axis=1, keepdims=1)
                                    if sqrt:
                                        tmp /= (L[s1]**0.5)
                                    Bres[k][:, s1:s1+1] += tmp
                                    if s1!=s2:
                                        tmp = np.sum(BBs2sk, axis=1, keepdims=1)
                                        if sqrt:
                                            tmp /= (L[s2]**0.5)
                                        Bres[k][:, s2:s2+1] += tmp
                                    if s2 == 0:
                                        tmp = np.sum(BDsk, axis=1, keepdims=1)
                                        if sqrt:
                                            tmp /= (L[s1]**0.5) 
                                        Bres[k][:, s1:s1+1] -= tmp
                                    '''
                                    if sqrt:
                                        Bres[k][:, s1:s1+1] += np.sum(BBss2k, axis=1, keepdims=1)/np.sqrt(L[s1])
                                        if s1!=s2:
                                            Bres[k][:, s2:s2+1] += np.sum(BBs2sk, axis=1, keepdims=1)/np.sqrt(L[s2])
                                        if s2 == 0:
                                            Bres[k][:, s1:s1+1] -= np.sum(BDsk, axis=1, keepdims=1)/np.sqrt(L[s1]) 
                                    else:
                                        Bres[k][:, s1:s1+1] += np.sum(BBss2k, axis=1, keepdims=1)
                                        if s1!=s2:
                                            Bres[k][:, s2:s2+1] += np.sum(BBs2sk, axis=1, keepdims=1)
                                        if s2 == 0:
                                            Bres[k][:, s1:s1+1] -= np.sum(BDsk, axis=1, keepdims=1)
                                    '''
                            del BBss2k, BBs2sk
                            if s2 == 0:
                                del BDsk
                        if lFlag and tFlag and (m2 == 0) and (s2 == 0):
                            if (k == 0):
                                if P > 0: # should be always True
                                    TBsm = np.dot(
                                        np.dot(Atk(Ah, m1, 0, r).T, Bkl(lD, 0, s1, full=1, docopy=0)),
                                        TBsm.T
                                    )
                                else:
                                    if sqrt:
                                        TBsm = np.dot(
                                            np.dot(Atk(Ah, m1, 0, r).T, np.tile(Bh[0][:, s1:s1+1]/np.sqrt(L[s1]), [1, L[s1]])),
                                            TBsm.T
                                        )
                                    else:
                                        TBsm = np.dot(
                                            np.dot(Atk(Ah, m1, 0, r).T, np.tile(Bh[0][:, s1:s1+1], [1, L[s1]])),
                                            TBsm.T
                                        )
                                TBsm = reshape(TBsm, Gh[m1].shape)
                                Gres[m1] += TBsm
                                del TBsm
                            ind1 = int(np.sum(r[:m1, k]))
                            ind2 = int(np.sum(r[:m1+1, k]))
                            if k < P:
                                IB_stepper = np.sum(L[s1]**np.arange(P-1))
                            else:
                                IB_stepper = np.sum(L[s1]**np.arange(P))
                            IB_stepper = int(IB_stepper)
                            BTsmk = unfold(BTsmk, k)[:, ::IB_stepper]
                            if k < P:
                                Ares[k][:, ind1:ind2] += np.dot(Bkl(lD, k, s1, full=1, docopy=0), BTsmk.T)
                            else:
                                if sqrt:
                                    Ares[k][:, ind1:ind2] += np.dot(
                                        np.tile(Bh[k][:, s1:s1+1]/np.sqrt(L[s1]), [1, L[s1]]),
                                        BTsmk.T
                                    )
                                else:
                                    Ares[k][:, ind1:ind2] += np.dot(np.tile(Bh[k][:, s1:s1+1], [1, L[s1]]), BTsmk.T)
                            BTsmk = np.dot(Atk(Ah, m1, k, r), BTsmk)
                            if (fmc is not None) and (fmc[k] is not None): #### k>=P must be None!!!
                                if s1 < fmc[k][1]:
                                    Usk, Wsk = Bkl(lD, k, s1, full=0, docopy=0)
                                    Bres[k][0] += np.dot(BTsmk, Wsk)
                                    ind = int(sum(L[:s1]))
                                    Bres[k][1][ind:ind+L[s1], :] += np.dot(Usk.T, BTsmk).T
                                else:
                                    ind = int(sum(L[fmc[k][1]:s1]))
                                    Bres[k][2][:, ind:ind+L[s1]] += BTsmk
                            else:
                                ind = int(sum(L[:s1]))
                                if k < P:
                                    Bres[k][:, ind:ind+L[s1]] += BTsmk
                                else:
                                    if sqrt:
                                        Bres[k][:, s1:s1+1] += np.sum(BTsmk, axis=1, keepdims=1)/np.sqrt(L[s1]) ## sqrt 
                                    else:
                                        Bres[k][:, s1:s1+1] += np.sum(BTsmk, axis=1, keepdims=1)
                            del BTsmk
                        if tFlag and (m2 == 0) and (s1 == 0) and (s2 == 0):
                            if (k == 0):
                                Gres[m1] -= prodTenMat(TDmk, Atk(Ah, m1, 0, r).T, 0)
                            ind1 = int(np.sum(r[:m1, k]))
                            ind2 = int(np.sum(r[:m1+1, k]))
                            axesD = list(range(k)) + [d] + list(range(k+1, d))
                            axesG = list(range(k)) + [d+1] + list(range(k+1, d))
                            Ares[k][:, ind1:ind2] -= np.einsum(TDmk, axesD, Gh[m1], axesG)
                            del TDmk
                        if tFlag and (s1 == 0) and (s2 == 0):
                            if k == 0:
                                tmp = np.dot(Atk(Ah, m1, 0, r).T,  Atk(Ah, m2, 0, r))
                                Gres[m1] += prodTenMat(TTmm2k, tmp, 0)
                                if m1 != m2:
                                    Gres[m2] += prodTenMat(TTmm2, tmp.T, 0)
                            axesG1 = list(range(k)) + [d] + list(range(k+1, d))
                            axesG2 = list(range(k)) + [d+1] + list(range(k+1, d))
                            TTmm2k = np.einsum(TTmm2k, axesG2, Gh[m1], axesG1)
                            ind1 = int(np.sum(r[:m1, k]))
                            ind2 = int(np.sum(r[:m1+1, k]))
                            Ares[k][:, ind1:ind2] += np.dot(Atk(Ah, m2, k, r), TTmm2k.T)
                            if m1 != m2:
                                ind1 = int(np.sum(r[:m2, k]))
                                ind2 = int(np.sum(r[:m2+1, k]))
                                Ares[k][:, ind1:ind2] += np.dot(Atk(Ah, m1, k, r), TTmm2k)
                            del TTmm2k
    if not return_vector:
        return Cres, Bres, Ares, Gres
    
    cdN, ldN, tdN = refactorizeDicts(canonical_dict, lro_dict, tucker_dict, Cres, Bres, Ares, Gres)
    rv = fcore2fvec(n, cdN, ldN, tdN)
    del cdN, ldN, tdN, Cres, Bres, Ares, Gres
    return rv
    
def refactorizeDicts(
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None,
    newC=None,
    newB=None,
    newA=None,
    newG=None
):
    cD = None
    if canonical_dict is not None:
        cD = {
            'Rc': canonical_dict['Rc'],
            'C': newC
        }
    lD = None
    if lro_dict is not None:
        lD = {
            'P': lro_dict['P'],
            'L': lro_dict['L'],
            'B': newB
        }
        if 'E' in lro_dict.keys():
            lD['E'] = lro_dict['E']
        
        if 'fullModesConfig' in lro_dict.keys():
            lD['fullModesConfig'] = copy.deepcopy(lro_dict['fullModesConfig'])
    tD = None
    if tucker_dict is not None:
        tD = {
            'r': tucker_dict['r'],
            'A': newA,
            'G': newG
        }
    return cD, lD, tD


def Atk(A, coreN, modeK, r):
    return A[modeK][:, int(np.sum(r[:coreN, modeK])) : int(np.sum(r[:coreN+1, modeK]))]

def AtAk_ij(AtAk, coreI, coreJ, modeK, r):
    return AtAk[
        int(np.sum(r[:coreI, modeK])) : int(np.sum(r[:coreI+1, modeK])),
        int(np.sum(r[:coreJ, modeK])) : int(np.sum(r[:coreJ+1, modeK]))
    ]

def CtAk_fj(CtAk, coreJ, modeK, r):
    return CtAk[:, int(np.sum(r[:coreJ, modeK])) : int(np.sum(r[:coreJ+1, modeK]))]

def mvHes(
    x,
    Th,
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None,
    curvature=True,
    return_vector=False,
    sqrt=_SQRT_CLOBAL
):
    #C, A, G, AtA, CtA, CtC, L, n, r, curvature=True):
    dtype = x.dtype
    n = Th.shape
    d = Th.ndim
    Cres = None
    Ch, Bh, Ah, Gh = fvec2fcore(x, n, canonical_dict, lro_dict, tucker_dict, full_result=True)
    if canonical_dict is not None:
        Rc = canonical_dict['Rc']
        C = canonical_dict['C']
        Cres = [None]*d
    Bres = None
    Rl = 1
    if lro_dict is not None:
        L = lro_dict['L']
        P = lro_dict['P']
        B = lro_dict['B']
        if 'E' in lro_dict.keys():
            E = lro_dict['E']
        else:
            E = computeE(L)
            lro_dict['E'] = E
        Rl = len(L)
        M = int(np.sum(L))
        Bres = [None]*d
        fmc = None
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
        lD = {}
        lD['L'] = L
        lD['P'] = P
        if fmc is not None:
            lD['fullModesConfig'] = fmc
        lhD = copy.deepcopy(lD)
        lD['B'] = B
        lhD['B'] = Bh
    Rt = 1
    Ares = None
    Gres = None
    if tucker_dict is not None:
        A = tucker_dict['A']
        G = tucker_dict['G']
        r = tucker_dict['r']
        Rt = len(G)
        Ares = [None]*d
        Gres = [None]*Rt
    cFlag = Cres is not None
    lFlag = Bres is not None
    tFlag = (Ares is not None) and (Gres is not None) #     
    
    for k in range(max(d, Rt)):
        if k < d:
            if cFlag:
                #Cres[k] = np.zeros(C[k].shape)
                Cres[k] = np.zeros(C[k].shape, dtype=dtype)
            if lFlag:
                if (k < P) and (fmc is not None) and (fmc[k] is not None):
                    Brk = []
                    indL = int(sum(L[:fmc[k][1]]))
                    Brk.append(np.zeros([n[k], fmc[k][0]], dtype=dtype))
                    Brk.append(np.zeros([indL, fmc[k][0]], dtype=dtype))
                    if indL < M:
                        Brk.append(np.zeros([n[k], M-indL], dtype=dtype))
                    else:
                        Brk.append(None)
                    Bres[k] = Brk
                else:
                    #Bres[k] = np.zeros(B[k].shape)
                    Bres[k] = np.zeros(B[k].shape, dtype=dtype)
            if tFlag:
                #Ares[k] = np.zeros(A[k].shape)
                Ares[k] = np.zeros(A[k].shape, dtype=dtype)
        if k < Rt:
            if tFlag:
                #Gres[k] = np.zeros(G[k].shape)
                Gres[k] = np.zeros(G[k].shape, dtype=dtype)
    
    if cFlag:
        IC_stepper = np.sum(Rc**np.arange(d-1))
        IC_stepper = int(IC_stepper)
    ############ Gramian part
        
    ##! dCpCq
    ##! dCpBtq <--> dBspCq # NOTE: Bsp = np.dot(Up, Wsp.T) if s < fmc[p]!
    ##! dCpAnq <--> dAmpCq
    ##! dCpGn  <--> dGmCq####### no GmCq
    
    ## dBspBtq
    ##! dBspAnq <--> dAmpBtq
    ##! dBspGn  <--> dGmBtq
    
    ##! dAmpAnq
    ##! dAmpGn <--> dGmAnq
    
    ##! dGmGn (Ghm, Ghn)
    
    ############ Quadratic parts:
    
    ## d2CpCq_C => dCpCq based
    # d2CpCq_B => dCpBtq based (?)
    # d2CpCq_T => dCpAnq based
    ## d2CpCq_D
    
    # d2BspBtq: if t != s => zero; thus t==s cases
    # d2BspBtq_C (?) 
    # d2BspBtq_B (?)
    # d2BspBtq_T (?)
    # d2BspBtq_D (?)
    
    # d2AmpAnq: if m != n => zero; thus m==n cases
    # d2AmpAnq_C => dCpAnq based
    # d2AmpAnq_B => (?)
    # d2AmpAnq_T => dAmpAnq based
    # d2AmpAnq_D
    
    #### double
    # d2AmpGn_C => dCpGn based
    # d2AmpGn_B => (?)
    # d2AmpGn_T => dAmpGn based
    # d2AmpGn_D
    
    #### others are zero
    
    for m1 in range(Rt):
        for m2 in range(m1+1):
            for s1 in range(Rl):
                for s2 in range(s1+1):
                    for p in range(d):
                        for q in range(p+1):
                            if cFlag and (m1 == 0) and (m2 == 0) and (s1 == 0) and (s2 == 0):
                                CpCq = np.ones([Rc, Rc], dtype=dtype)
                                if curvature and (s2 == s1):
                                    CpD = Th.copy()
                            if lFlag and (m1 == 0) and (m2 == 0):
                                BspBtq = np.ones([L[s1], L[s2]], dtype=dtype)
                                if curvature and (s1 == s2):
                                    BspD = Th.copy()
                                    # fmc?
                            if tFlag and (s1 == 0) and (s2 == 0):
                                AmpAnq = G[m1].copy()
                                if curvature and (m1 == m2):
                                    GmD = Th.copy()
                                if (m1 != m2) and (p == q):
                                    AmpAnq2 = G[m2].copy()
                                if ((p == 0) and (q == 0)) or (curvature and (p == q)):
                                    GmGn = Gh[m1].copy()
                                    if (m1 != m2):
                                        GnGm = Gh[m2].copy()
                                #if (m2 == 0):
                                #    AmpGn = G[m1].copy()
                            if cFlag and lFlag and (m1 == 0) and (m2 == 0) and (s2 == 0):
                                CpBtq = np.ones([Rc, L[s1]], dtype=dtype)
                            if cFlag and tFlag and (s1 == 0) and (s2 == 0) and (m2 == 0):
                                CpAnq = G[m1].copy()
                                if (p == q):
                                    CpGn = np.ones([1, Rc], dtype=dtype)
                                    GmCq = Gh[m1].copy()
                            if lFlag and tFlag and (m2 == 0) and (s2 == 0):
                                BspAnq = G[m1].copy()
                                if (p == q):
                                    BspGn = Gh[m1].copy()
                                    GmBtq = np.ones([1, L[s1]], dtype=dtype)                                    
                            for k in range(d):
                                if (k==p) or (k==q):
                                    '''
                                    if tFlag and (s1 == 0) and (s2 == 0) and ((p == q == 0) or (curvature and (p == q))):
                                        tmp = np.dot(Atk(A, m2, k, r).T, Atk(A, m1, k, r))
                                        GmGn = prodTenMat(GmGn, tmp, k)
                                        if (m1 != m2) and (p == 0):
                                            GnGm = prodTenMat(GnGm, tmp.T, k)
                                    '''
                                    if tFlag and cFlag and (s1 == 0) and (s2 == 0) and (m2 == 0) and (p == q):
                                        tmp = np.dot(Atk(A, m1, k, r).T, Ch[k])
                                        #CpGn = krp_cw(CpGn, tmp)
                                    continue
                                if cFlag and (m1 == 0) and (m2 == 0) and (s1 == 0) and (s2 == 0):
                                    CpCq *= np.dot(C[k].T, C[k])
                                    if curvature:
                                        CpD = prodTenMat(CpD, C[k].T, k)
                                if lFlag and (m1 == 0) and (m2 == 0):
                                    if k < P:
                                        tmp = Bkl(lD, k, s1, full=1, docopy=0)
                                        BspBtq *= np.dot(tmp.T, Bkl(lD, k, s2, full=1, docopy=0))
                                        if curvature and (s1 == s2):
                                            BspD = prodTenMat(BspD, tmp.T, k)
                                    else:
                                        tmp = B[k][:, s1:s1+1].copy()
                                        if sqrt:
                                            tmp /= (L[s1]**0.5)
                                        if curvature and (s1 == s2):
                                            BspD = prodTenMat(BspD, tmp.T, k)
                                        tmp = np.dot(tmp.T, B[k][:, s2:s2+1])
                                        if sqrt:
                                            tmp /= (L[s2]**0.5)
                                        BspBtq *= tmp
                                if tFlag and (s1 == 0) and (s2 == 0):
                                    tmp = Atk(A, m2, k, r)
                                    if curvature and (m1 == m2):
                                        GmD = prodTenMat(GmD, tmp.T, k)
                                    tmp = np.dot(tmp.T, Atk(A, m1, k, r))
                                    AmpAnq = prodTenMat(AmpAnq, tmp, k)
                                    #if curvature and (m1 == m2) and (p == q):
                                    #    GmD = prodTenMat(GmD, k)
                                    if (m1 != m2) and (p == q):
                                        AmpAnq2 = prodTenMat(AmpAnq2, tmp.T, k)
                                    if ((p == 0) and (q == 0)) or (curvature and (p==q)):
                                        GmGn = prodTenMat(GmGn, tmp, k)
                                        if (m1 != m2) and (p == q):
                                            GnGm = prodTenMat(GnGm, tmp.T, k)
                                if cFlag and lFlag and (m1 == 0) and (m2 == 0) and (s2 == 0):
                                    if k < P:
                                        CpBtq *= np.dot(C[k].T, Bkl(lD, k, s1, full=1, docopy=0))
                                    else:
                                        tmp = np.dot(C[k].T, np.tile(B[k][:, s1:s1+1], [1, L[s1]]))
                                        if sqrt:
                                            tmp /= (L[s1]**0.5)
                                        CpBtq *= tmp
                                if cFlag and tFlag and (s1 == 0) and (s2 == 0) and (m2 == 0):
                                    tmp = np.dot(C[k].T, Atk(A, m1, k, r))
                                    CpAnq = prodTenMat(CpAnq, tmp, k)
                                    if p == q:
                                        CpGn = krp_cw(tmp.T, CpGn)
                                        GmCq = prodTenMat(GmCq, tmp, k)
                                if lFlag and tFlag and (m2 == 0) and (s2 == 0):
                                    if k < P:
                                        tmp = Bkl(lD, k, s1, full=1, docopy=0)
                                    else:
                                        tmp = B[k][:, s1:s1+1].copy()
                                        if sqrt:
                                            tmp /= (L[s1]**0.5)
                                    tmp = np.dot(tmp.T, Atk(A, m1, k, r))
                                    BspAnq = prodTenMat(BspAnq, tmp, k)
                                    if p == q:
                                        BspGn = prodTenMat(BspGn, tmp, k)
                                        if k >= P:
                                            tmp = np.tile(tmp.T, [1, L[s1]]).T
                                        GmBtq = krp_cw(tmp.T, GmBtq)
                            if cFlag and (m1 == 0) and (m2 == 0) and (s1 == 0) and (s2 == 0):
                                if (p == q):
                                    Cres[p] += np.dot(Ch[p], CpCq)
                                else:
                                    tmp1 = np.dot(C[q].T, Ch[q])
                                    Cres[p] += np.dot(CpCq*tmp1, C[p].T).T
                                    tmp2 = np.dot(Ch[p].T, C[p])
                                    Cres[q] += np.dot(C[q], CpCq*tmp2)
                                    if curvature:
                                        Cres[p] += np.dot(CpCq*tmp1.T, C[p].T).T
                                        Cres[q] += np.dot(CpCq*tmp2, C[q].T).T
                                        tmp1 = prodTenMat(CpD, Ch[q].T, q)
                                        Cres[p] -= unfold(tmp1, p)[:, ::IC_stepper]
                                        tmp2 = prodTenMat(CpD, Ch[p].T, p)
                                        Cres[q] -= unfold(tmp2, q)[:, ::IC_stepper]
                            if lFlag and (m1 == 0) and (m2 == 0):
                                if (p == q):
                                    if p < P:
                                        ### should be corrected 
                                        if (fmc is not None) and (fmc[p] is not None):
                                            if s2 < fmc[p][1]:
                                                Utl, Wtl = Bkl(lD, p, s2, full=0, docopy=0)
                                                Uhtl, Whtl = Bkl(lhD, p, s2, full=0, docopy=0)
                                                tmpU = np.dot(np.dot(Uhtl, Wtl.T), BspBtq.T)
                                                tmpW = np.dot(np.dot(Utl, Whtl.T), BspBtq.T)
                                                Usk, Wsk = Utl, Wtl         #########
                                                Uhsk, Whsk = Uhtl, Whtl     #########                                       
                                            else:
                                                tmp = np.dot(Bkl(lhD, p, s2, full=1, docopy=0), BspBtq.T)
                                            if s1 != s2:
                                                if (s1 < fmc[p][1]):
                                                    Usk, Wsk = Bkl(lD, p, s1, full=0, docopy=0)
                                                    Uhsk, Whsk = Bkl(lhD, p, s1, full=0, docopy=0)
                                                    tmpU1 = np.dot(np.dot(Uhsk, Wsk.T), BspBtq)
                                                    tmpW1 = np.dot(np.dot(Usk, Whsk.T), BspBtq)
                                                else:
                                                    tmp2 = np.dot(Bkl(lhD, p, s1, full=1, docopy=0), BspBtq)
                                            if (s2 < fmc[p][1]):
                                                if (s1 < fmc[p][1]):
                                                    ind = int(sum(L[:s1]))
                                                    Bres[p][0] += np.dot(Wsk.T, tmpU.T + tmpW.T).T
                                                    Bres[p][1][ind:ind+L[s1], :] += np.dot(Usk.T, tmpW+tmpU).T
                                                else:
                                                    ind = int(sum(L[fmc[p][1]:s1]))
                                                    Bres[p][2][:, ind:ind+L[s1]] += tmpU + tmpW
                                            else:
                                                if (s1 < fmc[p][1]):
                                                    ind = int(sum(L[:s1]))
                                                    Bres[p][0] += np.dot(Wsk, tmp.T).T
                                                    Bres[p][1][ind:ind+L[s1], :] += np.dot(Usk.T, tmp).T
                                                else:
                                                    ind = int(sum(L[fmc[p][1]:s1]))
                                                    Bres[p][2][:, ind:ind+L[s1]] += tmp
                                            if curvature:
                                                if (s1 < fmc[p][1]):
                                                    tmpCurve = np.dot(Bkl(lD, p, s2, full=1, docopy=0), BspBtq.T)
                                                    ind = int(sum(L[:s1]))
                                                    if s1 == s2:
                                                        IB_stepper = int(sum(L[s1]**np.arange(P-1)))
                                                        tmpCurve = tmpCurve - unfold(BspD, p)[:, ::IB_stepper]
                                                    Uhsk, Whsk = Bkl(lhD, p, s1, full=0, docopy=0)
                                                    Bres[p][0] += np.dot(Whsk.T, tmpCurve.T).T
                                                    Bres[p][1][ind:ind+L[s1], :] += np.dot(Uhsk.T, tmpCurve).T
                                                if (s1 != s2) and (s2 < fmc[p][1]):
                                                    tmpCurve = np.dot(Bkl(lD, p, s1, full=1, docopy=0), BspBtq)
                                                    ind = int(sum(L[:s2]))
                                                    Uhtk, Whtk = Bkl(lhD, p, s2, full=0, docopy=0)
                                                    Bres[p][0] += np.dot(Whtk.T, tmpCurve.T).T
                                                    Bres[p][1][ind:ind+L[s2], :] += np.dot(Uhtk.T, tmpCurve).T
                                                
                                                
                                            if (s1 != s2):
                                                if (s1 < fmc[p][1]):
                                                    if (s2 < fmc[p][1]):
                                                        ind = int(sum(L[:s2]))
                                                        Bres[p][0] += np.dot(Wtl.T, tmpU1.T+tmpW1.T).T
                                                        Bres[p][1][ind:ind+L[s2], :] += np.dot(Utl.T, tmpW1+tmpU1).T
                                                    else:
                                                        ind = int(sum(L[fmc[p][1]:s2]))
                                                        Bres[p][2][:, ind:ind+L[s2]] += tmpU1 + tmpW1
                                                else:
                                                    if (s2 < fmc[p][1]):
                                                        ind = int(sum(L[:s2]))
                                                        Bres[p][0] += np.dot(Wtl.T, tmp2.T).T
                                                        Bres[p][1][ind:ind+L[s2], :] += np.dot(Utl.T, tmp2).T
                                                    else:
                                                        ind = int(sum(L[fmc[p][1]:s2]))
                                                        Bres[p][2][:, ind:ind+L[s2]] += tmp2

                                        else: #no fmc constraint
                                            tmp = np.dot(Bkl(lhD, p, s2, full=1, docopy=0), BspBtq.T)
                                            if (s1!=s2):
                                                tmp2 = np.dot(Bkl(lhD, p, s1, full=1, docopy=0), BspBtq)
                                            ind = int(sum(L[:s1]))
                                            Bres[p][:, ind:ind+L[s1]] += tmp
                                            if s1 != s2:
                                                ind = int(sum(L[:s2]))
                                                Bres[p][:, ind:ind+L[s2]] += tmp2
                                    else:
                                        tmp = np.dot(np.tile(Bh[p][:, s2:s2+1], [1, L[s2]]), BspBtq.T)
                                        if sqrt:
                                            tmp /= (L[s2]**0.5)
                                        tmp = np.sum(tmp, axis=1, keepdims=1)
                                        if sqrt:
                                            tmp /= (L[s1]**0.5)
                                        Bres[p][:, s1:s1+1] += tmp
                                        if s1 != s2:
                                            tmp = np.dot(np.tile(Bh[p][:, s1:s1+1], [1, L[s1]]), BspBtq)
                                            if sqrt:
                                                tmp /= (L[s1]**0.5)
                                            tmp = np.sum(tmp, axis=1, keepdims=1)
                                            if sqrt:
                                                tmp /= (L[s2]**0.5)
                                            Bres[p][:, s2:s2+1] += tmp
                                else:
                                    for pi, qi in [[p, q], [q, p]]:
                                        if qi < P:
                                            if (fmc is not None) and (fmc[qi] is not None):
                                                Bsl = Bkl(lD, qi, s1, full=1, docopy=0)
                                                if (s2 < fmc[qi][1]):
                                                    Utl, Wtl = Bkl(lD, qi, s2, full=0, docopy=0)
                                                    Uhtl, Whtl = Bkl(lhD, qi, s2, full=0, docopy=0)
                                                    UhtlWtl = np.dot(Uhtl, Wtl.T)
                                                    UtlWhtl = np.dot(Utl, Whtl.T)
                                                    tmpU1 = np.dot(Bsl.T, UhtlWtl)
                                                    tmpW1 = np.dot(Bsl.T, UtlWhtl)
                                                    Usl, Wsl = Utl, Wtl         #########
                                                    Uhsl, Whsl = Uhtl, Whtl     #########
                                                else:
                                                    Bhtl = Bkl(lhD, qi, s2, full=1, docopy=0)
                                                    tmp1 = np.dot(Bsl.T, Bhtl)
                                                if (s1 != s2):
                                                    Btl = Bkl(lD, qi, s2, full=1, docopy=0)
                                                    if (s1 < fmc[qi][1]):
                                                        Usl, Wsl = Bkl(lD, qi, s1, full=0, docopy=0)
                                                        Uhsl, Whsl = Bkl(lhD, qi, s1, full=0, docopy=0)
                                                        UhslWsl = np.dot(Uhsl, Wsl.T)
                                                        UslWhsl = np.dot(Usl, Whsl.T)
                                                        tmpU2 = np.dot(Btl.T, UhslWsl)
                                                        tmpW2 = np.dot(Btl.T, UslWhsl)
                                                    else:
                                                        Bhsl = Bkl(lhD, qi, s1, full=1, docopy=0)
                                                        tmp2 = np.dot(Btl.T, Bhsl)              
                                            else: # no fmc
                                                Bhtl = Bkl(lhD, qi, s2, full=1, docopy=0)
                                                tmp1 = np.dot(
                                                    Bkl(lD, qi, s1, full=1, docopy=0).T,
                                                    Bhtl
                                                )
                                                if (s1 != s2):
                                                    Bhsl = Bkl(lhD, qi, s1, full=1, docopy=0)
                                                    tmp2 = np.dot(
                                                        Bkl(lD, qi, s2, full=1, docopy=0).T,
                                                        Bhsl
                                                    )  
                                        else:
                                            Bhtl =  Bh[qi][:, s2:s2+1].copy()
                                            if sqrt:
                                                Bhtl /= (L[s2]**0.5)
                                            tmp1 = np.dot(B[qi][:, s1:s1+1].T, Bhtl)
                                            if sqrt:
                                                tmp1 /= (L[s1]**0.5)
                                            if s1 != s2:
                                                Bhsl = Bh[qi][:, s1:s1+1].copy()
                                                if sqrt:
                                                    Bhsl /= (L[s1]**0.5)
                                                tmp2 = np.dot(B[qi][:, s2:s2+1].T, Bhsl)
                                                if sqrt:
                                                    tmp2 /= (L[s2]**0.5)
                                        if curvature:
                                            #if qi == q:
                                            if (fmc is not None) and (fmc[qi] is not None):
                                                if (s2 < fmc[qi][1]):
                                                    tmpDU1 = BspBtq*tmpU1
                                                    tmpDW1 = BspBtq*tmpW1
                                                    tmpD1 = tmpDU1+tmpDW1
                                                else:
                                                    tmpD1 = BspBtq*tmp1
                                                if (s1 != s2):
                                                    if (s1 < fmc[qi][1]):
                                                        tmpDU2 = BspBtq*tmpU2.T
                                                        tmpDW2 = BspBtq*tmpW2.T
                                                        tmpD2 = (tmpDU2+tmpDW2)
                                                    else:
                                                        tmpD2 = BspBtq*tmp2.T
                                                else:
                                                    step_cor = pi < P
                                                    IB_stepper = int(sum(L[s1]**np.arange(P-step_cor)))
                                                    if (s1 < fmc[qi][1]):
                                                        BspDU_tmp = prodTenMat(BspD, UhtlWtl.T, qi)
                                                        BspDU_tmp = unfold(BspDU_tmp, pi)[:, ::IB_stepper]
                                                        BspDW_tmp = prodTenMat(BspD, UtlWhtl.T, qi)
                                                        BspDW_tmp = unfold(BspDW_tmp, pi)[:, ::IB_stepper]
                                                        BspD_tmp = BspDU_tmp + BspDW_tmp
                                                    else:
                                                        BspD_tmp = prodTenMat(BspD, Bhtl.T, qi)
                                                        BspD_tmp = unfold(BspD_tmp, pi)[:, ::IB_stepper]
                                                    
                                            else:
                                                tmpD1 = BspBtq*tmp1
                                                if s1 != s2:
                                                    tmpD2 = BspBtq*tmp2.T
                                                else:
                                                    step_cor = pi < P
                                                    IB_stepper = int(sum(L[s1]**np.arange(P-step_cor)))
                                                    BspD_tmp = prodTenMat(BspD, Bhtl.T, qi)
                                                    BspD_tmp = unfold(BspD_tmp, pi)[:, ::IB_stepper]
                                        if (fmc is not None):
                                            if (fmc[qi] is not None):
                                                if (s2 < fmc[qi][1]):
                                                    tmp1 = (tmpU1 + tmpW1)*BspBtq
                                                    #if pi < P:
                                                    tmpU1 = tmpU1*BspBtq
                                                    tmpW1 = tmpW1*BspBtq
                                                    #
                                                    
                                                else:
                                                    tmp1 = tmp1*BspBtq
                                            else:
                                                tmp1 = tmp1*BspBtq
                                            if (s1 != s2):
                                                if (fmc[qi] is not None):
                                                    if (s1 < fmc[qi][1]):
                                                        tmp2 = (tmpU2 + tmpW2)*BspBtq.T
                                                        #if pi < P:
                                                        tmpU2 = tmpU2*BspBtq.T
                                                        tmpW2 = tmpW2*BspBtq.T
                                                        #else:
                                                    else:
                                                        tmp2 = tmp2*BspBtq.T
                                                else:
                                                    tmp2 = tmp2*BspBtq.T
                                        else:
                                            tmp1 = tmp1*BspBtq
                                            if s1 != s2:
                                                tmp2 = tmp2*BspBtq.T
                                        if pi < P:
                                            if (fmc is not None) and (fmc[pi] is not None):
                                                Bkt = Bkl(lD, pi, s2, full=1, docopy=0)
                                                if (s1 < fmc[pi][1]):
                                                    Usk, Wsk = Bkl(lD, pi, s1, full=0, docopy=0)
                                                    ind = int(sum(L[:s1]))
                                                    if (fmc[qi] is not None) and (s2 < fmc[qi][1]):
                                                        tmp = np.dot(Bkt, tmpU1.T+tmpW1.T)
                                                        Bres[pi][0] += np.dot(Wsk.T, tmp.T).T
                                                        ###  + tmpW1 may be not valid
                                                        Bres[pi][1][ind:ind+L[s1], :] += np.dot(Usk.T, tmp).T
                                                    else:
                                                        tmp = np.dot(Bkt, tmp1.T)
                                                        Bres[pi][0] += np.dot(Wsk.T, tmp.T).T
                                                        Bres[pi][1][ind:ind+L[s1], :] += np.dot(Usk.T, tmp).T
                                                else:
                                                    ind = int(sum(L[fmc[pi][1]:s1]))
                                                    if (fmc[qi] is not None) and (s2 < fmc[qi][1]):
                                                        ###  + tmpW1 may be not valid
                                                        tmp = np.dot(Bkt, tmpU1.T+tmpW1.T)
                                                        Bres[pi][2][:, ind:ind+L[s1]] += tmp
                                                    else:
                                                        tmp = np.dot(Bkt, tmp1.T)
                                                        Bres[pi][2][:, ind:ind+L[s1]] += tmp
                                                if (s1 != s2):
                                                    Bks = Bkl(lD, pi, s1, full=1, docopy=0)
                                                    if (s2 < fmc[pi][1]):
                                                        Utk, Wtk = Bkl(lD, pi, s2, full=0, docopy=0)
                                                        ind = int(sum(L[:s2]))
                                                        if (fmc[qi] is not None) and (s1 < fmc[qi][1]):
                                                            tmp = np.dot(Bks, tmpU2.T+tmpW2.T)
                                                            Bres[pi][0] += np.dot(Wtk.T, tmp.T).T
                                                            ###  + tmpW1 may be not valid
                                                            Bres[pi][1][ind:ind+L[s2], :] += np.dot(Utk.T, tmp).T
                                                        else:
                                                            tmp = np.dot(Bks, tmp2.T)
                                                            Bres[pi][0] += np.dot(Wtk.T, tmp.T).T
                                                            Bres[pi][1][ind:ind+L[s2], :] += np.dot(Utk.T, tmp).T
                                                    else:
                                                        ind = int(sum(L[fmc[pi][1]:s2]))
                                                        if (fmc[qi] is not None) and (s1 < fmc[qi][1]):
                                                            ###  + tmpW1 may be not valid
                                                            tmp = np.dot(Bks, tmpU2.T+tmpW2.T)
                                                            Bres[pi][2][:, ind:ind+L[s2]] += tmp
                                                        else:
                                                            tmp = np.dot(Bks, tmp2.T)
                                                            Bres[pi][2][:, ind:ind+L[s2]] += tmp 
                                            else:
                                                ind = int(sum(L[:s1]))
                                                #tmp1 = tmp1*BspBtq
                                                Bres[pi][:, ind:ind+L[s1]] += np.dot(
                                                    Bkl(lD, pi, s2, full=1, docopy=0),
                                                    tmp1.T
                                                )
                                                if s1 != s2:
                                                    ind = int(sum(L[:s2]))
                                                    #tmp2 = tmp2*BspBtq.T
                                                    Bres[pi][:, ind:ind+L[s2]] += np.dot(
                                                        Bkl(lD, pi, s1, full=1, docopy=0),
                                                        tmp2.T
                                                    )
                                            if curvature:
                                                if (fmc is not None) and (fmc[pi] is not None):
                                                    Bsk = Bkl(lD, pi, s1, full=1, docopy=0)
                                                    tmpD1 = np.dot(Bsk, tmpD1)
                                                    if (s1 == s2):
                                                        tmpD1 = tmpD1 - BspD_tmp
                                                    if (s2 < fmc[pi][1]):
                                                        ind = int(sum(L[:s2]))
                                                        Utk, Wtk = Bkl(lD, pi, s2, full=0, docopy=0)
                                                        Bres[pi][0] += np.dot(Wtk.T, tmpD1.T).T
                                                        Bres[pi][1][ind:ind+L[s2], :] += np.dot(Utk.T, tmpD1).T
                                                    else:
                                                        ind = int(sum(L[fmc[pi][1]:s2]))
                                                        Bres[pi][2][:, ind:ind+L[s2]] += tmpD1
                                                    if (s1 != s2):
                                                        Btk = Bkl(lD, pi, s2, full=1, docopy=0)
                                                        tmpD2 = np.dot(Btk, tmpD2.T)
                                                        if (s1 < fmc[pi][1]):
                                                            ind = int(sum(L[:s1]))
                                                            Usk, Wsk = Bkl(lD, pi, s1, full=0, docopy=0)
                                                            Bres[pi][0] += np.dot(Wsk.T, tmpD2.T).T
                                                            Bres[pi][1][ind:ind+L[s1], :] += np.dot(Usk.T, tmpD2).T
                                                        else:
                                                            ind = int(sum(L[fmc[pi][1]:s1]))
                                                            Bres[pi][2][:, ind:ind+L[s1]] += tmpD2
                                                else:
                                                    ind = int(sum(L[:s2]))
                                                    Bres[pi][:, ind:ind+L[s2]] += np.dot(
                                                        Bkl(lD, pi, s1, full=1, docopy=0),
                                                        tmpD1
                                                    )
                                                    if s1 != s2:
                                                        ind = int(sum(L[:s1]))
                                                        Bres[pi][:, ind:ind+L[s1]] += np.dot(
                                                            Bkl(lD, pi, s2, full=1, docopy=0),
                                                            tmpD2.T
                                                        )
                                                    else:
                                                        Bres[pi][:, ind:ind+L[s1]] -= BspD_tmp
                                        else:
                                            #tmp1 = tmp1*BspBtq
                                            tmp1 = np.dot(np.tile(B[pi][:, s2:s2+1], [1, L[s2]]), tmp1.T)
                                            tmp1 = np.sum(tmp1, axis=1, keepdims=1)
                                            if sqrt:
                                                tmp1 /= (L[s2]*L[s1])**0.5
                                            Bres[pi][:, s1:s1+1] += tmp1
                                            if s1 != s2:
                                                #tmp2 = tmp2*BspBtq.T
                                                tmp2 = np.dot(np.tile(B[pi][:,s1:s1+1], [1, L[s1]]), tmp2.T)
                                                tmp2 = np.sum(tmp2, axis=1, keepdims=1)
                                                if sqrt:
                                                    tmp2 /= (L[s1]*L[s2])**0.5
                                                Bres[pi][:, s2:s2+1] += tmp2
                                            if curvature:
                                                tmpD1 = np.dot(np.tile(B[pi][:, s1:s1+1], [1, L[s1]]), tmpD1)
                                                tmpD1 = np.sum(tmpD1, axis=1, keepdims=1)
                                                if sqrt:
                                                    tmpD1 /= (L[s1]*L[s2])**0.5
                                                Bres[pi][:, s2:s2+1] += tmpD1
                                                if s1 != s2:
                                                    tmpD2 = np.dot(np.tile(B[pi][:, s2:s2+1], [1, L[s2]]), tmpD2.T)
                                                    tmpD2 = np.sum(tmpD2, axis=1, keepdims=1)
                                                    if sqrt:
                                                        tmpD2 /= (L[s2]*L[s1])**0.5
                                                    Bres[pi][:, s1:s1+1] += tmpD2
                                                else:
                                                    tmpD3 = np.sum(BspD_tmp, axis=1, keepdims=1)
                                                    if sqrt:
                                                        tmpD3 /= (L[s1])**0.5
                                                    Bres[pi][:, s1:s1+1] -= tmpD3
                                del BspBtq
                            if tFlag and (s1 == 0) and (s2 == 0):
                                #tmp = Atk(A, m1, k, r)
                                #AmpAnq = prodTenMat(AmpAnq, np.dot(tmp.T, Atk(A, m2, k, r)), k)
                                # GmGn
                                if (p == 0) and (q == 0):
                                    tmp = np.dot(Atk(A, m2, p, r).T, Atk(A, m1, p, r))
                                    Gres[m2] += prodTenMat(GmGn, tmp, p)
                                    if (m1 != m2):
                                        Gres[m1] += prodTenMat(GnGm, tmp.T, p)
                                        #del GnGm
                                    #del GmGn
                                if (p == q):
                                    axesG2 = list(range(p)) + [d+1] + list(range(p+1, d))
                                    axesG1 = list(range(p)) + [d] + list(range(p+1, d))
                                    tmp = np.einsum(AmpAnq, axesG1, G[m2], axesG2)
                                    ind1 = int(np.sum(r[:m1, p]))
                                    ind2 = int(np.sum(r[:m1+1, p]))
                                    Ares[p][:, ind1:ind2] += np.dot(Atk(Ah, m2, p, r), tmp.T)
                                    # AmpGn
                                    tmp2 = np.dot(Atk(A, m2, p, r).T, Atk(Ah, m1, p, r))
                                    Gres[m2] += prodTenMat(AmpAnq, tmp2, p)
                                    #axesA = [p, d+1]
                                    tmp3 = np.einsum(AmpAnq, axesG1, Gh[m2], axesG2)
                                    Ares[p][:, ind1:ind2] += np.dot(Atk(A, m2, p, r), tmp3.T) 
                                    if (m1 != m2):
                                        ind1 = int(np.sum(r[:m2, p]))
                                        ind2 = int(np.sum(r[:m2+1, p]))
                                        Ares[p][:, ind1:ind2] += np.dot(Atk(Ah, m1, p, r), tmp) ### ?
                                        tmp = np.dot(Atk(A, m1, p, r).T, Atk(Ah, m2, p, r))
                                        Gres[m1] += prodTenMat(AmpAnq2, tmp, p)
                                        tmp3 = np.einsum(AmpAnq2, axesG1, Gh[m1], axesG2)
                                        Ares[p][:, ind1:ind2] += np.dot(Atk(A, m1, p, r), tmp3.T)
                                    if curvature:
                                        ind1 = int(np.sum(r[:m1, p]))
                                        ind2 = int(np.sum(r[:m1+1, p]))
                                        axes = list(range(p)) + [d] + list(range(p+1, d))
                                        axes2 = list(range(p)) + [d+1] + list(range(p+1, d))
                                        Ahmp = Atk(Ah, m1, p, r)
                                        Ahnp = Atk(Ah, m2, p, r)
                                        Ares[p][:, ind1:ind2] += np.dot(
                                            np.einsum(GmGn, axes, G[m2], axes2),
                                            Atk(A, m2, p, r).T
                                        ).T
                                        Gres[m2] += prodTenMat(AmpAnq, np.dot(Ahnp.T, Atk(A, m1, p, r)), p)
                                        if (m1!= m2):
                                            ind1 = int(np.sum(r[:m2, p]))
                                            ind2 = int(np.sum(r[:m2+1, p]))
                                            Ares[p][:, ind1:ind2] += np.dot(
                                                np.einsum(GnGm, axes, G[m1], axes2),
                                                Atk(A, m1, p, r).T
                                            ).T
                                            Gres[m1] += prodTenMat(AmpAnq2, np.dot(Ahmp.T, Atk(A, m2, p, r)), p)
                                        if (m1 == m2):
                                            ind1 = int(np.sum(r[:m2, p]))
                                            ind2 = int(np.sum(r[:m2+1, p]))
                                            Ares[p][:, ind1:ind2] -= np.einsum(GmD, axes, Gh[m1], axes2)
                                            Gres[m1] -= prodTenMat(GmD, Ahmp.T, p)
                                        
                                        
                                                                   
                                else:
                                    indices = [[p, q], [q, p]]
                                    for pi, qi in indices:
                                        tmp = np.dot(Atk(Ah, m2, qi, r).T, Atk(A, m1, qi, r))
                                        tmp = prodTenMat(AmpAnq, tmp, qi)
                                        axesG2 = list(range(pi)) + [d+1] + list(range(pi+1, d))
                                        axesG1 = list(range(pi)) + [d] + list(range(pi+1, d))
                                        tmp = np.einsum(tmp, axesG1, G[m2], axesG2)
                                        ind1 = int(np.sum(r[:m1, pi]))
                                        ind2 = int(np.sum(r[:m1+1, pi]))
                                        Ares[pi][:, ind1:ind2] += np.dot(Atk(A, m2, pi, r), tmp.T)
                                        if (m1 != m2):
                                            ind1 = int(np.sum(r[:m2, pi]))
                                            ind2 = int(np.sum(r[:m2+1, pi]))
                                            tmp = np.dot(Atk(A, m2, qi, r).T, Atk(Ah, m1, qi, r))
                                            tmp = prodTenMat(AmpAnq, tmp, qi)
                                            tmp = np.einsum(tmp, axesG1, G[m2], axesG2)
                                            Ares[pi][:, ind1:ind2] += np.dot(Atk(A, m1, pi, r), tmp)
                                        if (curvature):
                                            ind1 = int(np.sum(r[:m2, pi]))
                                            ind2 = int(np.sum(r[:m2+1, pi]))
                                            axes = list(range(pi)) + [d] + list(range(pi+1, d))
                                            axes2 = list(range(pi)) + [d+1] + list(range(pi+1, d))
                                            Ahmq = Atk(Ah, m1, qi, r)
                                            Ahnq = Atk(Ah, m2, qi, r)
                                            tmp = prodTenMat(AmpAnq, np.dot(Ahnq.T, Atk(A, m1, qi, r)), qi)
                                            Ares[pi][:, ind1:ind2] += np.dot(
                                                Atk(A, m1, pi, r),
                                                np.einsum(tmp, axes, G[m2], axes2),
                                            )
                                            if (m1 != m2):
                                                ind1 = int(np.sum(r[:m1, pi]))
                                                ind2 = int(np.sum(r[:m1+1, pi]))
                                                tmp = prodTenMat(AmpAnq, np.dot(Atk(A, m2, qi, r).T, Ahmq), qi)
                                                Ares[pi][:, ind1:ind2] += np.dot(
                                                    np.einsum(tmp, axes, G[m2], axes2),
                                                    Atk(A, m2, pi, r).T
                                                ).T
                                            if (m1 == m2):
                                                GmD_tmp = prodTenMat(GmD, Ahmq.T, qi)
                                                Ares[pi][:, ind1:ind2] -= np.einsum(GmD_tmp, axes, G[m1], axes2)
                                            
                                del AmpAnq
                            if cFlag and lFlag and (m1 == 0) and (m2 == 0) and (s2 == 0):
                                if (p == q):
                                    tmp = np.dot(Ch[p], CpBtq)
                                    if p < P:
                                        if (fmc is not None) and (fmc[p] is not None):
                                            if s1 < fmc[p][1]:
                                                Usp, Wsp = Bkl(lD, p, s1, full=0, docopy=0)
                                                Uhsp, Whsp = Bkl(lhD, p, s1, full=0, docopy=0)
                                                ind = int(sum(L[:s1]))
                                                Bres[p][0] += np.dot(tmp, Wsp)
                                                Bres[p][1][ind:ind+L[s1], :] += np.dot(Usp.T, tmp).T
                                                Cres[p] += np.dot(np.dot(Usp, Whsp.T), CpBtq.T)
                                                Cres[p] += np.dot(np.dot(Uhsp, Wsp.T), CpBtq.T)
                                                if curvature:
                                                    tmp2 = np.dot(C[p], CpBtq)
                                                    Bres[p][0] += np.dot(tmp2, Whsp)
                                                    Bres[p][1][ind:ind+L[s1], :] += np.dot(Uhsp.T, tmp2).T
                                            else:
                                                ind = int(sum(L[fmc[p][1]:s1]))
                                                Bres[p][2][:, ind:ind+L[s1]] += tmp                                           
                                                Cres[p] += np.dot(Bkl(lhD, p, s1, full=1, docopy=0), CpBtq.T)
                                        else:
                                            ind = int(sum(L[:s1]))
                                            Bres[p][:, ind:ind+L[s1]] += tmp
                                            Cres[p] += np.dot(Bh[p][:, ind:ind+L[s1]], CpBtq.T)
                                    else:
                                        tmp = np.sum(tmp, axis=1, keepdims=1)
                                        tmp2 = np.dot(np.tile(Bh[p][:, s1:s1+1], [1, L[s1]]), CpBtq.T)
                                        if sqrt:
                                            tmp /= (L[s1]**0.5)
                                            tmp2 /= (L[s1]**0.5)
                                        Bres[p][:, s1:s1+1] += tmp
                                        Cres[p] += tmp2
                                else:
                                    indices = [[p, q], [q, p]]
                                    for pi, qi in indices:
                                        if qi < P:
                                            # no curvature update
                                            if (fmc is not None) and (fmc[qi] is not None):
                                                if (s1 < fmc[qi][1]):
                                                    Utq, Wtq = Bkl(lD, qi, s1, full=0, docopy=0)
                                                    Uhtq, Whtq = Bkl(lhD, qi, s1, full=0, docopy=0)
                                                    tmp = np.dot(C[qi].T, np.dot(Utq, Whtq.T) + np.dot(Uhtq, Wtq.T))
                                                else:
                                                    ind = int(sum(L[:s1]))
                                                    tmp = np.dot(C[qi].T, Bkl(lhD, qi, s1, full=0, docopy=0))
                                            else:
                                                ind = int(sum(L[:s1]))
                                                tmp = np.dot(C[qi].T, Bh[qi][:, ind:ind+L[s1]])
                                            if curvature:
                                                tmpC = np.dot(Ch[qi].T, Bkl(lD, qi, s1, full=1, docopy=0))
                                                tmp += tmpC
                                            tmp = tmp*CpBtq    
                                                    #tmpB = CpBtq*np.dot(C[qi].T, Bh[qi][:, ind:ind+L[s1]])
                                        else:
                                            tmp = np.dot(C[qi].T, np.tile(Bh[qi][:, s1:s1+1], [1, L[s1]]))
                                            if sqrt:
                                                tmp /= (L[s1]**0.5)
                                            if curvature:
                                                tmpC = np.dot(Ch[qi].T, np.tile(B[qi][:, s1:s1+1], [1, L[s1]]))
                                                if sqrt:
                                                    tmpC /= (L[s1]**0.5)
                                                tmp += tmpC
                                            tmp = tmp*CpBtq
                                        if pi < P:
                                            Cres[pi] += np.dot(Bkl(lD, pi, s1, full=1, docopy=0), tmp.T)
                                            tmp = np.dot(Ch[pi].T, Bkl(lD, pi, s1, full=1, docopy=0))
                                            if curvature:
                                                #Cres[pi] += np.dot(Bkl(lD, pi, s1, full=1, docopy=0), tmpC.T)
                                                if (fmc is not None) and (fmc[pi] is not None) and (s1 < fmc[pi][1]):
                                                    Utp, Wtp = Bkl(lD, pi, s1, full=0, docopy=0)
                                                    Uhtp, Whtp = Bkl(lhD, pi, s1, full=0, docopy=0)
                                                    tmpC = np.dot(C[pi].T, np.dot(Utp, Whtp.T) + np.dot(Uhtp, Wtp.T))
                                                else:
                                                    tmpC = np.dot(C[pi].T, Bkl(lhD, pi, s1, full=1, docopy=0))
                                                tmp += tmpC
                                            tmp = tmp*CpBtq
                                        else:
                                            tmp = np.dot(np.tile(B[pi][:, s1:s1+1], [1, L[s1]]), tmp.T)
                                            if sqrt:
                                                tmp /= (L[s1]**0.5)
                                            Cres[pi] += tmp
                                            tmp = np.dot(Ch[pi].T, np.tile(B[pi][:, s1:s1+1], [1, L[s1]]))
                                            if sqrt:
                                                tmp /= (L[s1]**0.5)
                                            if curvature:
                                                #tmpC = np.dot(np.tile(B[pi][:, s1:s1+1], [1, L[s1]]), tmpC.T)
                                                #if sqrt:
                                                #    tmpC /= np.sqrt(L[s1])
                                                #Cres[pi] += tmpC
                                                tmpC = np.dot(C[pi].T, np.tile(Bh[pi][:, s1:s1+1], [1, L[s1]]))
                                                if sqrt:
                                                    tmpC /= (L[s1]**0.5)
                                                tmp += tmpC
                                            tmp = tmp*CpBtq
                                        tmp = np.dot(C[qi], tmp)
                                        #if curvature:
                                        #    tmpC = np.dot(C[qi], tmpC)
                                        if qi < P:
                                            if (fmc is not None) and (fmc[qi] is not None):
                                                if (s1 < fmc[qi][1]):
                                                    ind = int(sum(L[:s1]))
                                                    Bres[qi][0] += np.dot(tmp, Wtq)
                                                    Bres[qi][1][ind:ind+L[s1], :] += np.dot(Utq.T, tmp).T
                                                else:
                                                    ind = int(sum(L[fmc[qi][1]:s1]))
                                                    Bres[qi][2][:, ind:ind+L[s1]] += tmp
                                            else:
                                                ind = int(sum(L[:s1]))
                                                Bres[qi][:, ind:ind+L[s1]] += tmp
                                                #if curvature:
                                                #    Bres[qi][:, ind:ind+L[s1]] += tmpC
                                        else:
                                            tmp = np.sum(tmp, axis=1, keepdims=1)
                                            if sqrt:
                                                tmp /= (L[s1]**0.5)
                                            Bres[qi][:, s1:s1+1] += tmp
                                            #if curvature:
                                            #    tmpC = np.sum(tmpC, axis=1, keepdims=1)
                                            #    if sqrt:
                                            #        tmpC /= np.sqrt(L[s1])
                                            #    Bres[qi][:, s1:s1+1] += tmpC
                                del CpBtq
                            if cFlag and tFlag and (s1 == 0) and (s2 == 0) and (m2 == 0):      
                                if p == q:
                                    tmp = unfold(CpAnq, p)[:, ::IC_stepper]
                                    ind1 = int(np.sum(r[:m1, p]))
                                    ind2 = int(np.sum(r[:m1+1, p]))
                                    Ares[p][:, ind1:ind2] += np.dot(Ch[p], tmp.T)
                                    Cres[p] += np.dot(Atk(Ah, m1, p, r), tmp)
                                    shp = list(G[m1].shape)
                                    shp = shp[p:p+1] + shp[:p] + shp[p+1:]
                                    tmp = np.dot(Atk(A, m1, p, r).T, Ch[p])
                                    if curvature:
                                        tmp += np.dot(Atk(Ah, m1, p, r).T, C[p])
                                    CpGn = np.dot(tmp, CpGn.T)
                                    CpGn = reshape(CpGn, shp)
                                    sigma = list(range(1, p+1)) + [0] + list(range(p+1, d))
                                    CpGn = np.transpose(CpGn, sigma)
                                    Gres[m1] += CpGn
                                    if curvature:
                                        tmp = unfold(GmCq, p)[:, ::IC_stepper]
                                        ind1 = int(np.sum(r[:m1, p]))
                                        ind2 = int(np.sum(r[:m1+1, p]))
                                        Ares[p][:, ind1:ind2] += np.dot(C[p], tmp.T)
                                    GmCq = prodTenMat(GmCq, Atk(A, m1, p, r), p)
                                    GmCq = unfold(GmCq, p)[:, ::IC_stepper]
                                    Cres[p] += GmCq
                                    del CpGn, GmCq
                                else:
                                    pqs = [[p,q], [q, p]]
                                    for pi, qi in pqs:
                                        tmp = np.dot(Ch[pi].T, Atk(A, m1, pi, r))
                                        if curvature:
                                            tmp += np.dot(C[pi].T, Atk(Ah, m1, pi, r))
                                        tmp = prodTenMat(CpAnq, tmp, pi)
                                        tmp = unfold(tmp, qi)[:, ::IC_stepper]
                                        ind1 = int(np.sum(r[:m1, qi]))
                                        ind2 = int(np.sum(r[:m1+1, qi]))
                                        Ares[qi][:, ind1:ind2] += np.dot(C[qi], tmp.T)
                                        tmp = np.dot(C[qi].T, Atk(Ah, m1, qi, r))
                                        if curvature:
                                            tmp += np.dot(Ch[qi].T, Atk(A, m1, qi, r))
                                        tmp = prodTenMat(CpAnq, tmp, qi)
                                        tmp = unfold(tmp, pi)[:, ::IC_stepper]
                                        Cres[pi] += np.dot(Atk(A, m1, pi, r), tmp)
                                
                               
                            if lFlag and tFlag and (m2 == 0) and (s2 == 0):
                                if p == q:
                                    if p < P:
                                        if (fmc is not None) and (fmc[p] is not None) and (s1 < fmc[p][1]):
                                            Usp, Wsp = Bkl(lD, p, s1, full=0, docopy=0)
                                            Uhsp, Whsp = Bkl(lhD, p, s1, full=0, docopy=0)
                                            tmpUW = np.dot(Usp, Whsp.T) + np.dot(Uhsp, Wsp.T)
                                            tmp = tmpUW.copy()
                                        else:
                                            tmp = Bkl(lhD, p, s1, full=1, docopy=0)
                                        if curvature:
                                            tmp2 = Bkl(lD, p, s1, full=1, docopy=0)
                                    else:
                                        tmp = Bh[p][:, s1:s1+1].copy()
                                        if sqrt:
                                            tmp /= (L[s1]**0.5)
                                        if curvature:
                                            tmp2 = np.tile(B[p][:, s1:s1+1], [1, L[s1]])
                                            if sqrt:
                                                tmp2 /= (L[s1]**0.5)
                                    tmp = np.dot(tmp.T, Atk(A, m1, p, r))
                                    if p >= P:
                                        tmp = np.tile(tmp.T, [1, L[s1]]).T
                                    if curvature:
                                        tmp += np.dot(tmp2.T, Atk(Ah, m1, p, r))
                                    GmBtq = np.dot(tmp.T, GmBtq.T)
                                    shp = list(G[m1].shape)
                                    shp = shp[p:p+1] + shp[:p] + shp[p+1:]
                                    GmBtq = reshape(GmBtq, shp)
                                    sigma = list(range(1, p+1)) + [0] + list(range(p+1, d))
                                    GmBtq = np.transpose(GmBtq, sigma)
                                    Gres[m1] += GmBtq
                                    if p < P:
                                        if (fmc is not None) and (fmc[p] is not None) and(s1 < fmc[p][1]):
                                            #if :
                                            #Usk, Wsk = Bkl(lD, p, s1, full=0, docopy=0)
                                            #Uhsk, Whsk = Bkl(lhD, p, s1, full=0, docopy=0)
                                            #tmp = np.dot(Usk, Whsk.T) + np.dot(Uhsk, Wsk.T)
                                            tmp = tmpUW.copy()
                                        else:
                                            tmp = Bkl(lhD, p, s1, full=1, docopy=0)
                                        IB_stepper = np.sum(L[s1]**np.arange(P-1))
                                    else:
                                        tmp = np.tile(Bh[p][:, s1:s1+1], [1, L[s1]])
                                        if sqrt:
                                            tmp /= (L[s1]**0.5)
                                        IB_stepper = np.sum(L[s1]**np.arange(P))
                                    IB_stepper = int(IB_stepper)
                                    BspGn = unfold(BspGn, p)[:, ::IB_stepper]
                                    if curvature:
                                        BspGn2 = np.dot(tmp2, BspGn.T)
                                    BspGn = np.dot(BspGn.T, Atk(A, m1, p, r).T)
                                    BspAnq_tmp = unfold(BspAnq, p)[:, ::IB_stepper]
                                    tmp = np.dot(BspAnq_tmp, tmp.T).T
                                    ind1 = int(np.sum(r[:m1, p]))
                                    ind2 = int(np.sum(r[:m1+1, p]))
                                    if curvature:
                                        tmp += BspGn2
                                    Ares[p][:, ind1:ind2] += tmp
                                    tmp = np.dot(Atk(Ah, m1, p, r), BspAnq_tmp)
                                    tmp += BspGn.T #############
                                    if p < P:
                                        if (fmc is not None) and (fmc[p] is not None):
                                            if s1 < fmc[p][1]:
                                                Bres[p][0] += np.dot(Wsp.T, tmp.T).T
                                                ind = int(sum(L[:s1]))
                                                Bres[p][1][ind:ind+L[s1], :] += np.dot(Usp.T, tmp).T
                                                if curvature:
                                                    tmpUW2 = np.dot(Atk(A, m1, p, r), BspAnq_tmp)
                                                    Bres[p][0] += np.dot(Whsp.T, tmpUW2.T).T
                                                    Bres[p][1][ind:ind+L[s1], :] += np.dot(Uhsp.T, tmpUW2).T
                                            else:
                                                ind = int(sum(L[fmc[p][1]:s1]))
                                                Bres[p][2][:, ind:ind+L[s1]] += tmp
                                        else:
                                            ind = int(sum(L[:s1]))
                                            Bres[p][:, ind:ind+L[s1]] += tmp
                                    else:
                                        tmp = np.sum(tmp, axis=1, keepdims=1)
                                        if sqrt:
                                             tmp /= (L[s1]**0.5)
                                        Bres[p][:, s1:s1+1] += tmp
                                else: # p!=q
                                    pqs = [[p, q], [q, p]]
                                    for pi, qi in pqs:
                                        if qi < P:
                                            tmp = Bkl(lD, qi, s1, full=1, docopy=0)
                                            if curvature:
                                                if (fmc is not None) and (fmc[qi] is not None) and (s1 < fmc[qi][1]):
                                                    Usl, Wsl = Bkl(lD, qi, s1, full=0, docopy=0)
                                                    Uhsl, Whsl = Bkl(lhD, qi, s1, full=0, docopy=0)
                                                    tmp2 = np.dot(Usl, Whsl.T) + np.dot(Uhsl, Wsl.T)
                                                else:
                                                    tmp2 = Bkl(lhD, qi, s1, full=1, docopy=0)
                                        else:
                                            tmp = B[qi][:, s1:s1+1].copy()
                                            if sqrt:
                                                tmp /= (L[s1]**0.5)
                                            if curvature:
                                                tmp2 = Bh[qi][:, s1:s1+1].copy()
                                                if sqrt:
                                                    tmp2 /= (L[s1]**0.5)
                                        differ = pi < P # 0 if False, 1 if True
                                        IB_stepper = np.sum(L[s1]**np.arange(P-differ))
                                        IB_stepper = int(IB_stepper)
                                        tmp = np.dot(tmp.T, Atk(Ah, m1, qi, r))
                                        if curvature:
                                            tmp += np.dot(tmp2.T, Atk(A, m1, qi, r))
                                        BspAnq_tmp = prodTenMat(BspAnq, tmp, qi)
                                        BspAnq_tmp = unfold(BspAnq_tmp, pi)[:, ::IB_stepper]
                                        BspAnq_tmp = np.dot(Atk(A, m1, pi, r), BspAnq_tmp)
                                        if pi < P: 
                                            if (fmc is not None) and (fmc[pi] is not None):
                                                if s1 < fmc[pi][1]:
                                                    Usk, Wsk = Bkl(lD, pi, s1, full=0, docopy=0)
                                                    Uhsk, Whsk = Bkl(lhD, pi, s1, full=0, docopy=0)
                                                    #tmp_B = [np.dot(Usk, Whsk.T), np.dot(Uhsk, Wsk.T)]
                                                    tmp_B = np.dot(Usk, Whsk.T) + np.dot(Uhsk, Wsk.T)
                                                    Bres[pi][0] += np.dot(BspAnq_tmp, Wsk)
                                                    ind = int(sum(L[:s1]))
                                                    Bres[pi][1][ind:ind+L[s1], :] += np.dot(Usk.T, BspAnq_tmp).T
                                                else:
                                                    #tmp_B = [Bkl(lhD, pi, s1, full=0, docopy=0)]
                                                    tmp_B = Bkl(lhD, pi, s1, full=0, docopy=0)
                                                    ind = int(sum(L[fmc[pi][1]:s1]))
                                                    Bres[pi][2][:, ind:ind+L[s1]] += BspAnq_tmp
                                            else:
                                                #tmp_B = [Bh[pi]] # copy?
                                                ind = int(sum(L[:s1]))
                                                Bres[pi][:, ind:ind+L[s1]] += BspAnq_tmp
                                                tmp_B = Bh[pi][:, ind:ind+L[s1]].copy()
                                            if curvature:
                                                tmp_B2 = Bkl(lD, pi, s1, full=1, docopy=0) #B[pi][:, ind:ind+L[s1]].copy()
                                        else:
                                            #tmp_B = [Bh[pi][:, s1:s1+1]]
                                            tmp_B = Bh[pi][:, s1:s1+1].copy()
                                            if sqrt:
                                                tmp_B /= (L[s1]**0.5)
                                            if curvature:
                                                tmp_B2 = B[pi][:, s1:s1+1].copy()
                                                if sqrt:
                                                    tmp_B2 /= (L[s1]**0.5)
                                            tmp = np.sum(BspAnq_tmp, axis=1, keepdims=1)
                                            if sqrt:
                                                tmp /= (L[s1]**0.5)
                                            Bres[pi][:, s1:s1+1] += tmp
                                        differ = qi < P # 0 if False, 1 if True
                                        IB_stepper = np.sum(L[s1]**np.arange(P-differ))
                                        IB_stepper = int(IB_stepper)
                                        #for tmpI in tmp:
                                            #tmpI = np.dot(tmpI.T, Atk(A, m1, pi, r))
                                        tmp = np.dot(tmp_B.T, Atk(A, m1, pi, r))
                                        if curvature:
                                            tmp += np.dot(tmp_B2.T, Atk(Ah, m1, pi, r))
                                        BspAnq_tmp = prodTenMat(BspAnq, tmp, pi)
                                        BspAnq_tmp = unfold(BspAnq_tmp, qi)[:, ::IB_stepper]
                                        ind1 = int(np.sum(r[:m1, qi]))
                                        ind2 = int(np.sum(r[:m1+1, qi]))
                                        if qi < P:
                                            Ares[qi][:, ind1:ind2] += np.dot(Bkl(lD, qi, s1, full=1, docopy=0), BspAnq_tmp.T)
                                        else:
                                            tmp = np.dot(np.tile(B[qi][:, s1:s1+1], [1,L[s1]]), BspAnq_tmp.T)
                                            if sqrt:
                                                tmp /= (L[s1]**0.5)
                                            Ares[qi][:, ind1:ind2] += tmp
                                        # end for
  
    if not return_vector:
        return Cres, Bres, Ares, Gres
    cdN, ldN, tdN = refactorizeDicts(canonical_dict, lro_dict, tucker_dict, Cres, Bres, Ares, Gres)
    rv = fcore2fvec(n, cdN, ldN, tdN)
    del cdN, ldN, tdN, Cres, Bres, Ares, Gres
    return rv

def commutationMatrix(m, n, dtype=np.float64):
    '''
    python equivalent of
    https://www.mathworks.com/matlabcentral/fileexchange/26781-vectorized-transpose-matrix
    '''

    d = m*n
    P = np.zeros(d*d, dtype=dtype)

    i = np.arange(d)
    rI = m*i - (d-1)*(i/n)
    #print rI
    I = np.ravel_multi_index([i, rI], [d, d], order='F')
    P[I] = 1
    P = reshape(P, [d, d])
    #P = P.T
    return P

def JtJ_diag(
    n,
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None,
    which='full',
    sqrt=_SQRT_CLOBAL,
    dtype=np.float64
):
    '''
    For several variants of Levenberg-Marquardt algorithm
    we need only the maximal element from Gramian's diagonal.
    Thus we compute it here without explicit representation of diagonal.
    
    WARNING: for shared low-rank (Lr,1) part there is a permutaton problem in explicit representation of diag(JtJ)
    (in Wsk part)
    '''
    d = len(n)
    cFlag = canonical_dict is not None
    lFlag = lro_dict is not None
    tFlag = tucker_dict is not None
    assert cFlag or lFlag or tFlag
    whiches = ['max', 'full', 'unique']
    assert which in whiches
    if cFlag:
        Rc = canonical_dict['Rc']
        C = canonical_dict['C']
    Rl = 1
    if lFlag:
        L = lro_dict['L']
        Rl = len(L)
        M = int(sum(L))
        P = lro_dict['P']
        B = lro_dict['B']
        
        if 'E' in lro_dict.keys():
            E = lro_dict['E']
        else:
            E = computeE(L, sqrt)
            lro_dict['E'] = E
        fmc = None
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
    Rt = 1
    if tFlag:
        A = tucker_dict['A']
        G = tucker_dict['G']
        r = tucker_dict['r']
        Rt = r.shape[0]
    if cFlag:
        dCk2 = []
    if lFlag:
        dBmk2 = []
        if fmc is not None:
            dUk2 = [[]]*d
            dWmk2 = []
    if tFlag:
        dAmk2 = []
        dGm2 = []
    rv = -np.inf
    for m in range(max(Rt, Rl)):
        if tFlag:
            dAk2 = []
        if lFlag:
            dBk2 = []
        for k in range(d):
            if tFlag and (m < Rt):
                core = G[m].copy()
                for l in range(d):
                    if l != k:
                        tmp = Atk(A, m, l, r)
                        core = prodTenMat(core, np.dot(tmp.T, tmp), l)
                prodAxes = list(range(k)) + [d] + list(range(k+1, d))
                prodAxes2 = list(range(k)) + [d+1] + list(range(k+1, d))
                core = np.einsum(core, prodAxes, G[m], prodAxes2)
                if (which == 'full') or (which == 'unique'):
                    dAk2.append(np.diag(core))
                else:
                    rv = max(rv, np.diag(core).max())
            if lFlag and (m < Rl):
                if k < P:
                    tmp = Bkl(lro_dict, k, m, full=1, docopy=0)
                    tmp = np.dot(tmp.T, tmp)
                    #tmp = np.linalg.norm(tmp, axis=0)**2.
                else:
                    #tmp = np.linalg.norm(B[k], axis=0)**2.
                    tmp = B[k][:, m:m+1].copy()
                    if sqrt:
                        tmp /= (L[m]**0.5)
                    tmp = np.dot(tmp.T, tmp)
                    #tmp = np.repeat(tmp, L)
                    #tmp = np.dot(tmp, E)
                dBk2.append(tmp.copy())  
                
            #    if (fmc is not None) and (fmc[k] is not None):
            #        if m < fmc[k][1]:
            #            pass#for l in range(
            #    else:
                
            if (m == 0):
                if tFlag:
                    # Tucker core
                    tmp = np.linalg.norm(A[k], axis=0, keepdims=True)**2.
                    dGm2.append(tmp.copy())
                if cFlag:
                    # CP factors
                    tmp = np.linalg.norm(C[k], axis=0)**2.
                    dCk2.append(tmp.copy())
                if lFlag and (fmc is not None) and (fmc[k] is not None):
                    WtVt = np.zeros([fmc[k][0], int(np.prod(n[:k]+n[k+1:]))], dtype=dtype)
                    for s in range(fmc[k][1]):
                        tmp = Bkl(lro_dict, k, s, full=False, docopy=False)
                        Vt = None
                        for l in range(d):
                            if l == k:
                                continue
                            if Vt is None:
                                if l < P:
                                    Vt = Bkl(lro_dict, l, s, full=True, docopy=False)
                                else:
                                    Vt = np.tile(B[l][:, s:s+1], [1, L[s]]) ## sqrt
                                    if sqrt:
                                        Vt /= (L[s]**0.5)
                            else:
                                if l < P:
                                    Vt = krp_cw(Bkl(lro_dict, l, s, full=True, docopy=False), Vt)
                                else:
                                    tmpVt = np.tile(B[l][:, s:s+1], [1, L[s]])
                                    if sqrt:
                                        tmpVt /= (L[s]**0.5)
                                    Vt = krp_cw(tmpVt, Vt) #sqrt
                        WtVt += np.dot(tmp[1].T, Vt.T)
                    WtVt = np.dot(WtVt, WtVt.T)                    
                    if which == 'max':
                        rv = max(rv, np.diag(WtVt).max())
                    else:
                        dUk2[k] = np.diag(WtVt)
                '''
                if lFlag:
                    if k < P:
                        tmp = Bkl(lro_dict, k)#, m, full=1, docopy=1)
                        tmp = np.linalg.norm(tmp, axis=0)**2.
                    else:
                        tmp = np.linalg.norm(B[k], axis=0)**2.
                        if sqrt:
                            tmp /= np.array(L)
                        tmp = np.repeat(tmp, L)
                        #tmp = np.dot(tmp, E)
                    dBk2.append(tmp.copy())
                '''
        if tFlag:
            dAmk2.append(dAk2)
        if lFlag:
            dBmk2.append(dBk2)
    if cFlag:
        cdiag = []
    if lFlag:
        ldiag = []
        tmpB = [[]]*d
        #dBk2 =
    if tFlag:
        gdiag = [np.ones([1, 1], dtype=dtype) for m in range(Rt)]
        adiag = []
        tmpA = [[]]*d
    for k in range(d):
        if tFlag or lFlag:
            for m in range(max(Rt, Rl)):
                if tFlag and (m < Rt):
                    ind1 = int(np.sum(r[:m, k]))
                    ind2 = int(np.sum(r[:m+1, k]))
                    gdiag[m] = np.kron(dGm2[k][:, ind1:ind2], gdiag[m])
                    if k == (d-1):
                        gdiag[m] = vec(gdiag[m])
                    if which == 'full':
                        tmp = np.repeat(dAmk2[m][k], n[k]).tolist()
                        tmpA[k] = tmpA[k] + tmp
                    elif which == 'unique':
                        tmpA[k] = tmpA[k] + tmp                    
                if lFlag and (m < Rl):
                    Wk = np.prod(dBmk2[m][:k] + dBmk2[m][k+1:], axis=0)
                    if k < P:
                        #else:
                        Wk = np.diag(Wk)
                    else:
                        Wk = np.sum(Wk, keepdims=True) / float(L[m])
                    if (fmc is not None) and (fmc[k] is not None) and (m < fmc[k][1]):
                        Uk, _ = Bkl(lro_dict, k, m, full=0, docopy=0)
                        tmp = np.linalg.norm(Uk, axis=0)**2.
                        #tmp = np.dot(Uk.T, Uk)
                        Wk = np.kron(Wk, tmp)
                        #CM = commutationMatrix(L[m], fmc[k][1])
                        #print Wk.shape, L[m], fmc
                        Wk = reshape(Wk, [L[m], fmc[k][0]])
                        Wk = vec(Wk.T)
                        #Wk = np.dot(CM, np.diag(Wk))
                        #print 'Wsk', m, k, Wk.shape
                    if which == 'full':
                        if not ((fmc is not None) and (fmc[k] is not None) and (m < fmc[k][1])):
                            Wk = np.repeat(Wk, n[k])
                            #print 'Bsk', m, k, Wk.shape
                        if (fmc is not None) and (fmc[k] is not None) and (m == 0):
                            dUk2[k] = np.repeat(dUk2[k], n[k])
                            #print 'Uk', k, len(dUk2[k])
                            tmpB[k] = tmpB[k] + dUk2[k].tolist()
                        tmpB[k] = tmpB[k] + Wk.tolist()
                    elif which == 'max':                        
                        rv = max(rv, Wk.max())
                    else:
                        if (fmc is not None) and (fmc[k] is not None) and (m == 0):
                            tmpB[k] = tmpB[k] + dUk2[k].tolist()
                        tmpB[k] = tmpB[k] + Wk.tolist() 
        if cFlag:
            Wk = np.prod(dCk2[:k] + dCk2[k+1:], axis=0)
            if which == 'full':
                tmp = np.repeat(Wk, n[k]).tolist()
                cdiag = cdiag + tmp
            elif which == 'unique':
                cdiag = cdiag + Wk.tolist()
            elif which == 'max':
                rv = max(rv, Wk.max())
        
        #if lFlag:
        #    print len(tmpB[k]), k
            '''
        Wk = np.prod(dBk2[:k] + dBk2[k+1:], axis=0)
            if k < P:
                Wk = np.diag(Wk)
            else:
                Wk = np.sum(Wk) / float(L[m])
            
            if (fmc is not None) and (fmc[k] is not None):
                Uk, _ = Bkl(lro_dict, k, 0, full=0, docopy=0)
                tmpWlist = []
                for m in range(fmc[k][1]):
                    tmpW = np.linalg.norm(Uk, axis=0)**2.
                    tmpWlist = tmpWlist + tmpW
                tmpW = np.array(tmpWlist)
                sizeW = tmpW.size
                tmpW = np.kron(Wk[:sizeW], tmpW)
                if which == 'full':
                    tmpW2= []
                    if Wk.size > sizeW:
                        tmpW2 = np.repeat(Wk[sizeW:], n[k]).tolist()
                    tmpB[k] = tmpB[k] + dU2[k] + tmpW.tolist() + tmpW2
                elif which == 'unique':
                    tmpB[k] = tmpB[k] + dU2[k] + tmpW.tolist() + Wk[sizeW:].tolist()
                elif which == 'max':
                    dUk_max = np.max(dU2[k])
                    rv = max(rv, dUk_max, Wk[sizeW:].max(), tmpW.max())
            else:
                if (which == 'full'):
                    tmp = np.repeat(Wk, n[k]).tolist()
                    ldiag = ldiag + tmp
                elif (which == 'unique'):
                    ldiag = ldiag + Wk.tolist()
                elif which == 'max':
                    rv = max(rv, Wk.max())
            '''
    if lFlag:
        if (which == 'full') or (which == 'unique') :
            ldiag = reduce(lambda x, y: x+y, tmpB)
    if tFlag:
        if (which == 'full') or (which == 'unique') :
            adiag = reduce(lambda x, y: x+y, tmpA)
        tmp = []
        for m in range(Rt):
            if (which == 'full') or (which == 'unique'):
                tmp = tmp + gdiag[m].tolist()
            else:
                rv = max(rv, gdiag[m].max())
        del gdiag
        gdiag = tmp
    if (which == 'full') or (which == 'unique') :
        rv = []
        if cFlag:
            rv = rv + cdiag
        if lFlag:
            rv = rv + ldiag
        if tFlag:
            rv = rv + adiag + gdiag
        rv = vec(np.array(rv, dtype=dtype))
    return rv


def BJprecTCD(x, canonical_dict=None, lro_dict=None, tucker_dict=None, CtC=None, AtA=None):
    #x, C, A, G, L, n, r, CtC, AtA):
    cFlag = 0
    dtype = x.dtype
    if canonical_dict is not None:
        assert CtC is not None, "If there is a canonical part, CtC must be specified"
        Rc = canonical_dict['Rc']
        C = canonical_dict['C']
        d = len(C)
        cFlag = 1
    lFlag = 0
    if lro_dict is not None:
        #assert BtB is not None, "If there is an (Lr, 1) part, BtB must be specified"
        L = lro_dict['L']
        P = lro_dict['P']
        B = lro_dict['B']
        if 'E' in lro_dict.keys():
            E = lro_dict['E']
        else:
            E = computeE(L)
            lro_dict['E'] = E
        Rl = len(L)
        M = int(np.sum(L))
        d = len(B)
        lFlag = 1
    Rt = 1
    tFlag = 0
    if tucker_dict is not None:
        assert AtA is not None, "If there is a Tucker part, AtA must be specified"
        A = tucker_dict['A']
        G = tucker_dict['G']
        r = tucker_dict['r']
        Rt = len(G)
        d = len(A)
        tFlag = 1
    C1, B1, A1, G1 = fvec2fcore(x, n, canonical_dict, lro_dict, tucker_dict, full_result=True)
    C2 = None
    B2 = None
    A2 = None
    G2 = None
    if cFlag:
        C2 = [None]*len(C1)
    if lFlag:
        B2 = [None]*len(B1)
    if tFlag:
        A2 = copy.deepcopy(A1)
        G2 = [None]*len(G1)
    for m in range(Rt): # m
        if tFlag:
            GGm = G1[m].copy()
        for p in range(d): # m, p
            if tFlag:
                AAmk = G[m].copy()
            if m == 0: # p
                if cFlag:
                    CCk = np.ones([Rc, Rc], dtype=dtype)
                if lFlag:
                    BBk = np.ones([M, M], dtype=dtype)
            for k in range(d): # m, p, k
                if tFlag:
                    if k!=p: # m, k
                        AAmk = prodTenMat(AAmk, AtAk_ij(AtA[k], m, m, k, r), k)
                if m == 0: # p, k
                    if k!=p: # k
                        if cFlag:
                            if lFlag:
                                CCk *= CtC[k][:Rc, :Rc]
                            else:
                                CCk *= CtC[k]
                        if lFlag:
                            if cFlag:
                                tmp = CtC[k][Rc:, Rc:]
                            else:
                                tmp = CtC[k]
                            BBk *= tmp
            
            if m == 0: # p, k
                if cFlag:
                    CCk = np.linalg.pinv(CCk)
                    C2[p] = np.dot(C1[p], CCk)
                if lFlag:
                    BBk = np.linalg.pinv(BBk)
                    if p < P:
                        B2[p] = np.dot(B1[p], BBk)
                    else:
                        B2[p] = np.dot(B1[p], np.dot(E, np.dot(BBk, E.T)))
            if tFlag:
                prodAxes = list(range(p)) + list(range(p+1, d))
                AAmk = np.tensordot(AAmk, G[m], axes=(prodAxes, prodAxes))
                AAmk = np.linalg.pinv(AAmk)
                A2[p][:, int(np.sum(r[:m, p])) : int(np.sum(r[:m+1, p]))] = np.dot(
                    A1[p][:, int(np.sum(r[:m, p])) : int(np.sum(r[:m+1, p]))], AAmk
                )
                GGm = prodTenMat(GGm, np.linalg.pinv(AtAk_ij(AtA[p], m, m, p, r)), p)
        if tFlag:
            G2[m] = GGm.copy()
    cdN, ldN, tdN = refactorizeDicts(canonical_dict, lro_dict, tucker_dict, C2, B2, A2, G2)
    rv = fcore2fvec(n, cdN, ldN, tdN)
    del cdN, ldN, tdN, C2, B2, A2, G2
    return rv





def backtrackArmijo(functional, x, d, g, projector=None, beta=0.9, sigma=0.5, beta_lim=1e-8, normT=1., sign=None):
    assert (beta > 0) and (beta < 1.), f"backtrackArmijo: uncorrect value of beta ({beta:.3g})"
    assert (sigma > 0) and (sigma < 1.), f"backtrackArmijo: uncorrect value of sigma ({sigma:.3g})"
    assert (beta_lim > 0) and (beta_lim < beta), f"backtrackArmijo: uncorrect value of beta_lim ({beta_lim:.3g})"
    
    assert (sign is None) or (sign == 'pos') or (sign == 'neg')
    
    fx = functional(x)
    tkp = beta
    tkn = beta
    l = 1
    if (sign is None) or (sign == 'pos'):
        posF = False
    else:
        posF = True
        fpxk_pos = np.inf
    if (sign is None) or (sign == 'neg'):
        negF = False
    else:
        negF = True
        fpxk_neg = np.inf
    while not (posF and negF):
        if not posF:
            tkp = tkp ** l
            if tkp < beta_lim:
                posF = True
            xtk_pos = x + tkp * d
            if projector is not None:
                xtk_pos = projector(xtk_pos)
            fpxk_pos = functional(xtk_pos)
            if (fpxk_pos <= fx + sigma * (g * (xtk_pos - x)).sum()/normT**2.):
                posF = True
        if not negF:
            tkn = tkn ** l
            if tkn < beta_lim:
                negF = True
            xtk_neg = x - tkn * d
            if projector is not None:
                xtk_neg = projector(xtk_neg)
            fpxk_neg = functional(xtk_neg)
            if (fpxk_neg <= fx + sigma * (g * (xtk_neg - x)).sum()/normT**2.):
                negF = True
        l += 1
    #print fpxk_pos, fpxk_neg, l
    fv = np.inf
    fv = min(fpxk_pos, fpxk_neg, fv)
    xtk = None
    fpxk = None
    tk = None
    if (fv == fpxk_pos) or (sign == 'pos'):
        tk = tkp
        xtk = xtk_pos
        fpxk = fpxk_pos
    elif (fv == fpxk_neg) or (sign == 'neg'):
        tk = tkn
        xtk = xtk_neg
        fpxk = fpxk_neg
    assert xtk is not None
    return xtk, fpxk, tk

def conjugate_residual_method(A, b, x0=None, iM=None, maxitnum=100, tol=1e-8):
    N = b.size
    dtype = b.dtype
    norm_b = np.linalg.norm(b)
    b /= norm_b
    if x0 is None:
        x = np.random.uniform(-1, 1, size=N).astype(dtype)
    else:
        x = x0.copy()
    r = b - A(x)
    if iM is not None:
        r = iM(r)
    p = r.copy()
    info = 1
    Ar = A(r)
    for k in range(maxitnum):
        if np.linalg.norm(r) < tol:
            info = 0
            break
        Ap = A(p)
        alpha = np.inner(A(r), r)
        if iM is not None:
            iMAp = iM(Ap)
            alpha /= np.inner(iMAp, Ap)
        else:
            alpha /= np.inner(Ap, Ap)
        xnew = x + alpha*p
        if iM is not None:
            rnew = r - alpha*iMAp
        else:
            rnew = r - alpha*Ap
        Arnew = A(rnew)
        beta = np.inner(Arnew, rnew) / np.inner(Ar, r)
        pnew = rnew + beta*p
        Ar = Arnew.copy()
        x = xnew.copy()
        p = pnew.copy()
        r = rnew.copy()
        del Arnew, xnew, rnew, pnew
    if np.linalg.norm(r) >= tol:
        info = -1
    b *= norm_b
    x*= norm_b
    return x, info
        
def conjugate_gradient_method(A, b, x0=None, iM=None, maxitnum=100, tol=1e-8):
    dtype = b.dtype
    N = b.size
    norm_b = np.linalg.norm(b)
    b /= norm_b
    if x0 is None:
        x = np.random.uniform(-1, 1, size=N).astype(dtype)
    else:
        x = x0.copy()
    r = b - A(x)
    if iM is not None:
        z = iM(r)
        p = z.copy()
    else:
        p = r.copy()
    info = 1
    for k in range(maxitnum):
        Ap = A(p)
        if iM is not None:
            alpha = np.inner(z, r)
        else:
            alpha = np.inner(r, r)
        alpha /= np.inner(Ap, p)
        xnew = x + alpha*p
        rnew = r - alpha*Ap
        if np.linalg.norm(rnew) < tol:
            info = 0
            break
        if iM is not None:
            znew = iM(rnew)
            beta = np.inner(znew, rnew) / np.inner(z, r)
            pnew = znew + beta*p
            z = znew.copy()
            del znew
        else:
            beta = np.inner(rnew, rnew) / np.inner(r, r)
            pnew = rnew + beta*p
        x = xnew.copy()
        p = pnew.copy()
        r = rnew.copy()
        del xnew, rnew, pnew
    b *= norm_b
    x*= norm_b
    return x, info        

def computeTauStar(u, v, deltaK):
    cA = np.sum(u*u)
    cB = np.sum(u*v)
    cC = np.sum(v*v) - deltaK**2.
    tauStar = (-cB + (cB**2. - cA*cC)**0.5) / cA
    return tauStar

def dogleg_step(gk, Hk, dk, prec=None, maxInnerIt=100, tol=1e-6):
    yku = -np.sum(gk*gk) / np.sum(Hk(gk)*gk) * gk
    dist1 = np.linalg.norm(yku)
    if dist1 >= dk:
        yNew = dk / dist1 * yku
        return yNew
    # Hk must be positive definite
    ykqn, infor = conjugate_gradient_method(
        Hk, -gk, x0=gk.copy(), iM=prec, maxitnum=maxInnerIt, tol=tol
    )
    dist2 = np.linalg.norm(ykqn)
    if dist2 <= dk:
        yNew = ykqn
        return yNew
    else:
        # find tauStar s.t. \|yku + tauStar*(ykqn - yku)\| = dk
        dy = ykqn-yku
        tauStar = computeTauStar(dy, yku, dk)
        yNew = yku + tauStar*dy
        return yNew

def scg_step(gk, Hk, deltaK, eps=1e-8, maxInnerIt=100):
    dtype = gk.dtype
    n = gk.size
    dj = -gk
    zj = np.zeros(n, dtype=dtype)
    for j in range(n):
        Hkdj = Hk(dj)
        hdd = np.sum(Hkdj * dj)
        if hdd <= 0:
            # (*) find tauStar s.t. \|zk + tauStar*dj -xk\| = deltaK 
            tauStar = computeTauStar(dj, zj, deltaK)
            return zj + tauStar*dj
        else:
            # (**) find tauj = arg min_(tau >= 0) mk(zj + tau * dj)
            tauj = - np.sum((Hk(zj) + gk)*dj) / np.sum(Hkdj*dj)
            zNew = zj + tauj*dj
            if np.linalg.norm(zNew) >= deltaK:
                # (*) find tauStar s.t. \|zk + tauStar*dj -xk\| = deltaK
                tauStar = computeTauStar(dj, zj, deltaK)
                return zj + tauStar*dj
            gNew = gk + Hk(zNew)
            if (np.linalg.norm(gNew) <= eps) or (j == maxInnerIt):
                return zNew
            else:
                dj = -gNew + np.sum(gNew*gNew) / np.sum(gk*gk) * dj
                zj = zNew.copy()
    print("You broke the program! Or something have gone wrong")
    return

def kModeInverse(Z, k):
    dtype = Z.dtype
    d = Z.ndim
    r = np.array(Z.shape)
    iZ, sigma = unfold(Z, k, return_sigma=True)
    iZ = np.linalg.pinv(iZ).T
    iSigma = list(range(1, sigma[0]+1)) + [0] + list(range(sigma[0]+1, d))
    iZ = reshape(iZ, r[sigma]) 
    iZ = np.transpose(iZ, iSigma)
    return iZ

def alsUpdAim(i, m, T, iA, canonical_dict=None, lro_dict=None, tucker_dict=None, orthog='qr', sqrt=_SQRT_CLOBAL):
    d = T.ndim
    n = T.shape
    cFlag = canonical_dict is not None
    lFlag = lro_dict is not None
    tFlag = tucker_dict is not None
    assert tFlag
    if cFlag:
        Rc = canonical_dict['Rc']
        C = canonical_dict['C']
    Rl = 1
    if lFlag:
        L = lro_dict['L']
        Rl = len(L)
        M = int(sum(L))
        P = lro_dict['P']
        B = lro_dict['B']
        E = lro_dict['E']
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
        else:
            fmc = None
    Rt = 1
    if tFlag:
        A = tucker_dict['A']
        G = tucker_dict['G']
        r = tucker_dict['r']
        Rt = r.shape[0]
    iG = kModeInverse(G[m], i)
    partTD = T.copy()
    if cFlag:
        partTC = iG.copy()
    for s in range(max(Rl, Rt)):
        if (s != m) and (s < Rt):
            partTT = iG.copy()
        if lFlag and (s < Rl):
            partTB = iG.copy()
        for k in range(d):
            if k == i:
                continue
            if lFlag and (s < Rl):
                if k < P:
                    partTB = prodTenMat(
                        partTB,
                        np.dot(Bkl(lro_dict, k, s, full=True, docopy=False).T, Atk(iA, m, k, r)),
                        k
                    )
                else:
                    tmp = B[k][:, s:s+1].copy()
                    if sqrt:
                        tmp /= (L[s]**0.5)
                    partTB = prodTenMat(partTB, np.dot(tmp.T, Atk(iA, m, k, r)), k)
            if (s != m) and (s < Rt):
                partTT = prodTenMat(partTT, np.dot(Atk(A, s, k, r).T, Atk(iA, m, k, r)), k)
            if s == 0:
                # deal with densed tensor T
                partTD = prodTenMat(partTD, Atk(iA, m, k, r).T, k)
                # deal with CPD tensor
                if cFlag:
                    partTC = prodTenMat(partTC, np.dot(C[k].T, Atk(iA, m, k, r)), k)                
        if s == 0:
            # deal with densed tensor T    
            #prodAxes = range(i) + range(i+1, d)
            axesT = list(range(i)) + [d] + list(range(i+1, d))
            axesD = list(range(i)) + [d+1] + list(range(i+1, d))
            #partTD = np.tensordot(partTD, iG, axes=(prodAxes, prodAxes))
            partTD = np.einsum(partTD, axesT, iG, axesD)
            res = partTD.copy()
            del partTD
            if cFlag:
                IC_stepper = np.sum(Rc**np.arange(d-1))
                IC_stepper = int(IC_stepper)
                partTC = unfold(partTC, i)[:, ::IC_stepper]
                partTC = np.dot(C[i], partTC.T)
                res -= partTC
                del partTC
        if lFlag and (s < Rl):
            pmark = i < P
            IB_stepper = int(np.sum(L[s]**np.arange(P-pmark)))
            partTB = unfold(partTB, i)[:, ::IB_stepper]
            if i < P:
                partTB = np.dot(Bkl(lro_dict, i, s, full=True, docopy=False), partTB.T)
            else:
                tmp = np.tile(B[i][:, s:s+1], [1,L[s]])
                if sqrt:
                    tmp /= (L[s]**0.5)
                partTB = np.dot(tmp, partTB.T)
            res -= partTB
            del partTB
        if (s != m) and (s < Rt):
            axesTm = list(range(i)) + [d+2] + list(range(i+1, d))
            axesTs = list(range(i)) + [d] + list(range(i+1, d))
            axesAs = [d, d+1]
            #partTT = np.tensordot(G[m2], partTT, axes=(prodAxes, prodAxes))
            partTT = np.einsum(G[s], axesTs, partTT, axesTm, Atk(A, s, i, r).T, axesAs)
            res -= partTT
            del partTT
    if orthog is None:
        pass
    else:
        if orthog == 'qr':
            u, _ = np.linalg.qr(res)
        elif orthog == 'svd':
            u, _, _ = fastSVD(res)
        else:
            raise NotImplementedError
        [uN, uR] = u.shape
        rN, rR = res.shape
        uN = min(uN, rN)
        uR = min(uR, rR)
        #result1 = np.zeros(result1.shape)
        res[:uN, :uR] = u[:uN, :uR]
    return res

def alsUpdGm(m, T, iA, canonical_dict=None, lro_dict=None, tucker_dict=None, sqrt=_SQRT_CLOBAL):
    d = T.ndim
    n = T.shape
    cFlag = canonical_dict is not None
    lFlag = lro_dict is not None
    tFlag = tucker_dict is not None
    assert tFlag
    if cFlag:
        Rc = canonical_dict['Rc']
        C = canonical_dict['C']
    Rl = 1
    if lFlag:
        L = lro_dict['L']
        Rl = len(L)
        M = int(sum(L))
        P = lro_dict['P']
        B = lro_dict['B']
        E = lro_dict['E']
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
        else:
            fmc = None
    Rt = 1
    if tFlag:
        A = tucker_dict['A']
        G = tucker_dict['G']
        r = tucker_dict['r']
        Rt = r.shape[0]
    partTD = T.copy()
    if cFlag:
        partTC = None
    for s in range(max(Rl, Rt)):
        if (s != m) and (s < Rt):
            partTG = G[s].copy()
        if lFlag and (s < Rl):
            partTB = None
        for k in range(d):
            # deal with densed tensor T
            if s == 0:
                partTD = prodTenMat(partTD, Atk(iA, m, k, r).T, k)
                # deal with CPD tensor
                if cFlag:
                    if k == 0:
                        partTC = np.dot(Atk(iA, m, k, r).T, C[k])
                    else:
                        partTC = krp_cw(np.dot(Atk(iA, m, k, r).T, C[k]), partTC)
            if lFlag and (s < Rl):
                if k == 0:
                    partTB = np.dot(Atk(iA, m, k, r).T, Bkl(lro_dict, k, s, full=True, docopy=False))
                elif k < P:
                    partTB = krp_cw(np.dot(Atk(iA, m, k, r).T, Bkl(lro_dict, k, s, full=True, docopy=False)), partTB)
                else:
                    tmp = np.tile(B[k][:, s:s+1], [1, L[s]])
                    if sqrt:
                        tmp /= (L[s]**0.5)
                    partTB = krp_cw(np.dot(Atk(iA, m, k, r).T, tmp), partTB)
            if (s != m) and (s < Rt):
                partTG = prodTenMat(partTG, np.dot(Atk(iA, m, k, r).T, Atk(A, s, k, r)), k)
        if s == 0:
            res = partTD.copy()
            del partTD
            if cFlag:
                # deal with CPD tensor
                partTC = partTC.sum(axis=1)
                res -= reshape(partTC, res.shape)
                del partTC
        if (s != m) and (s < Rt):
            res -= partTG
            del partTG
        if lFlag and (s < Rl):
            partTB = partTB.sum(axis=1)
            res -= reshape(partTB, res.shape)
            del partTB
    
    return res

def Bkl(lro_dict, modeK, blockL=None, full=False, docopy=False):
    dtype = lro_dict['B'][0].dtype
    L = lro_dict['L']
    P = lro_dict['P']
    assert modeK < P
    fmc = None
    if 'fullModesConfig' in lro_dict.keys():
        fmc = lro_dict['fullModesConfig']
    if blockL is None:
        if (fmc is not None) and (fmc[modeK] is not None):
            if full:
                tmp = lro_dict['B'][modeK]
                rv = np.zeros([tmp[0].shape[0], int(sum(L))], dtype=dtype)
                rv[:, :tmp[1].shape[0]] += np.dot(tmp[0], tmp[1].T)
                if tmp[2] is not None:
                    rv[:, tmp[1].shape[0]:] += tmp[2]
                return rv
            else:
                rv = lro_dict['B'][modeK]
        else:
            rv = lro_dict['B'][modeK]
    else:
        ind = int(sum(L[:blockL]))
        offset = L[blockL]
        if (fmc is not None) and (fmc[modeK] is not None):
            if blockL < fmc[modeK][1]:
                if full:
                    return np.dot(lro_dict['B'][modeK][0], lro_dict['B'][modeK][1][ind:ind+offset, :].T)
                rv = lro_dict['B'][modeK][0], lro_dict['B'][modeK][1][ind:ind+offset, :]
            else:
                ind = int(sum(L[fmc[modeK][1]:blockL]))
                rv = lro_dict['B'][modeK][2][:, ind:ind+offset]
        else:
            rv = lro_dict['B'][modeK][:, ind:ind+offset]
    if docopy:
        return rv.copy()
    return rv
    
'''
def BtBk_ij(AtAk, coreI, coreJ, modeK, r):
    return AtAk[int(np.sum(r[:coreI, modeK])) : int(np.sum(r[:coreI+1, modeK])),
                int(np.sum(r[:coreJ, modeK])) : int(np.sum(r[:coreJ+1, modeK]))]

def DtAk_fj(CtAk, coreJ, modeK, r):
    return CtAk[:, int(np.sum(r[:coreJ, modeK])) : int(np.sum(r[:coreJ+1, modeK]))]
'''
def alsUpdBi_old(i, T, canonical_dict=None, lro_dict=None, tucker_dict=None, fullModesRanks=None, sqrt=_SQRT_CLOBAL):
    d = T.ndim
    n = T.shape
    dtype = T.dtype
    cFlag = canonical_dict is not None
    lFlag = lro_dict is not None
    tFlag = tucker_dict is not None
    assert lFlag
    if cFlag:
        Rc = canonical_dict['Rc']
        C = canonical_dict['C']
    if lFlag:
        L = lro_dict['L']
        M = int(sum(L))
        Rl = len(L)
        P = lro_dict['P']
        B = lro_dict['B']
        E = lro_dict['E']
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
        else:
            fmc = None
    Rt = 1
    if tFlag:
        A = tucker_dict['A']
        G = tucker_dict['G']
        r = tucker_dict['r']
        Rt = r.shape[0]
    if i < P:
        result = []
        if (fmc is not None) and (fmc[i] is not None):
            invU = np.linalg.pinv(B[i][0])
        for s in range(Rl):
            H = np.ones([L[s], L[s]], dtype=dtype)
            IB_stepper = np.sum((L[s])**np.arange(P-1))
            IB_stepper = int(IB_stepper)
            partBD = T.copy()
            #Bis = Bkl(lro_dict, i, s, full=True, docopy=False)
            for m in range(max(Rt, Rl)):
                if (tFlag) and (m < Rt):
                    partBT = G[m].copy()
                if (m < Rl) and (m!=s):
                    partBB = np.ones([L[m], L[s]], dtype=dtype)
                if cFlag and (m == 0):
                    partBC = np.ones([Rc, L[s]], dtype=dtype)
                for k in range(d):
                    if k == i:
                        if (tFlag) and (m < Rt):
                            partBT = prodTenMat(partBT, Atk(A, m, k, r), k)
                        continue
                    if k < P:
                        Bks = Bkl(lro_dict, k, s, full=True, docopy=False)
                    else:
                        Bks = B[k][:, s:s+1].copy()
                        if sqrt:
                            Bks /= (L[s]**0.5) ## copy?
                    # partBB (always)
                    if (m < Rl) and (m != s):
                        if k < P:
                            partBB *= np.dot(Bkl(lro_dict, k, m, full=True, docopy=False).T, Bks)
                        else:
                            tmp = np.dot(B[k][:, m:m+1].T, Bks)
                            if sqrt:
                                tmp /= (L[m]**0.5)
                            partBB *= tmp ## sqrt
                    if (m == 0):
                        # partBC
                        if cFlag:
                            if k < P:
                                partBC *= np.dot(C[k].T, Bks)
                            else:
                                partBC *= np.dot(C[k].T, np.tile(Bks, [1, L[s]]))
                        # part BD (always)
                        if k < P:
                            partBD = prodTenMat(partBD, Bks.T, k)
                            H *= np.dot(Bks.T, Bks)
                        else:
                            partBD = prodTenMat(partBD, Bks.T, k) ## sqrt
                            #partBD = prodTenMat(partBD, np.ones([L[s], 1]), k)
                            H *= (Bks**2.).sum() ## sqrt
                    # partBT
                    if tFlag and (m < Rt):
                        if k < P:
                            partBT = prodTenMat(partBT, np.dot(Bks.T, Atk(A, m, k, r)), k)
                        else:
                            partBT = prodTenMat(partBT, np.dot(Bks.T, Atk(A, m, k, r)), k)
                            partBT = prodTenMat(partBT, np.ones([L[s], 1], dtype=dtype), k) ## sqrt
                if m == 0:
                    H = np.linalg.pinv(H)
                    partBD = unfold(partBD, i)[:, ::IB_stepper]
                    #partBD = np.dot(partBD, H.T) # actually, H = H^T
                    res = partBD.copy()
                    del partBD
                    if cFlag:
                        #partBC = np.dot(partBC, H.T)
                        res -= np.dot(C[i], partBC)
                        del partBC
                if tFlag and (m < Rt):
                    partBT = unfold(partBT, i)[:, ::IB_stepper]
                    #partBT = np.dot(partBT, H.T)
                    res -= partBT
                    del partBT
                if (m < Rl) and (m!=s):
                    #partBB = np.dot(partBB, H.T)
                    res -= np.dot(Bkl(lro_dict, i, m, full=True, docopy=False), partBB)
            res = np.dot(res, H.T)
            result.append(res.copy())
            if (fmc is not None) and (fmc[i] is not None):
                if (s < fmc[i][1]):
                    result[-1] = np.dot(invU, result[-1])
        if (fmc is not None) and (fmc[i] is not None):
            if fmc[i][1] < Rl:
                result = [None, np.hstack(result[:fmc[i][1]]).T, np.hstack(result[fmc[i][1]:])]
            else:
                result = [None, np.hstack(result[:fmc[i][1]]).T, None]
            WtVt = np.zeros([fmc[i][0], int(np.prod(n[:i]+n[i+1:]))], dtype=dtype)
            ind = 0
            for s in range(fmc[i][1]):
                #offset = n[s]
                tmp = Bkl(lro_dict, i, s, full=False, docopy=False)
                Vt = None
                for k in range(d):
                    if k == i:
                        continue
                    if Vt is None:
                        if k < P:
                            Vt = Bkl(lro_dict, k, s, full=True, docopy=False)
                        else:
                            Vt = np.tile(B[k][:, s:s+1], [1, L[s]]) ## sqrt
                            if sqrt:
                                Vt /= (L[s]**0.5)
                    else:
                        if k < P:
                            Vt = krp_cw(Bkl(lro_dict, k, s, full=True, docopy=False), Vt)
                        else:
                            tmpVt = np.tile(B[k][:, s:s+1], [1, L[s]])
                            if sqrt:
                                tmpVt /= (L[s]**0.5)
                            Vt = krp_cw(tmpVt, Vt) #sqrt
                #WtVt[:, ind:ind+offset] += np.dot(tmp[1].T, Vt.T)
                WtVt += np.dot(tmp[1].T, Vt.T)
                #ind += offset
            del Vt
            H = np.dot(WtVt, WtVt.T)
            H = np.linalg.pinv(H)                
            #partBB = np.ones([L[s], fmc[i][0]])
            IC_stepper = np.sum((L[s])**np.arange(P-1))######################
            IC_stepper = int(IC_stepper)
            shapeWtVt = [fmc[i][0]] + list(n[:i])+list(n[i+1:])
            WtVt = reshape(WtVt, shapeWtVt)
            axes = [i] + list(range(i)) + list(range(i+1, d))
            WtVt = np.transpose(WtVt, axes)
            #Bis = Bkl(lro_dict, i, s, full=True, docopy=False)
            for m in range(max(Rt, Rl)):
                if (tFlag) and (m < Rt):
                    partBT = WtVt.copy()
                if (fmc[i][1] < m < Rl):
                    partBB = WtVt.copy()
                if (cFlag) and (m == 0):
                    partBC = WtVt.copy()
                for k in range(d):
                    if k == i:
                        continue
                    # partBB (always)
                    if (fmc[i][1] < m < Rl) and (k != i):
                        if k < P:
                            partBB = prodTenMat(
                                partBB, Bkl(lro_dict, k, m, full=True, docopy=False).T, k
                            )
                        else:
                            tmpBB = B[k][:, m:m+1].copy()
                            if sqrt:
                                tmpBB /= (L[m]**0.5)
                            partBB = prodTenMat(partBB, tmpBB.T, k) ## sqrt
                    # partBT
                    if tFlag and (m < Rt):
                        partBT = prodTenMat(partBT, Atk(A, m, k, r).T, k)
                    if (m == 0):
                        # partBC
                        if cFlag:
                            partBC = prodTenMat(partBC, C[k].T, k)
                if m == 0:
                    # part BD (always)
                    axesT = list(range(i)) + [d] + list(range(i+1, d))
                    axesBD = list(range(i)) + [d+1] + list(range(i+1, d))
                    result[0] = np.einsum(T, axesT, WtVt, axesBD)
                    del axesBD, axesT
                    if cFlag:
                        IC_stepper2 = np.sum((Rl)**np.arange(d-1))
                        IC_stepper2 = int(IC_stepper2)
                        partBC = unfold(partBC, i)[:, ::IC_stepper2]
                        result[0] -= np.dot(C[i], partBC.T)
                        del partBC
                        
                if tFlag and (m < Rt):
                    axesA = [d+1, d]
                    axesBT = range(i) + [d+2] + range(i+1, d)
                    axesG = range(i) + [d] + range(i+1, d)
                    partBT = np.einsum(
                        G[m], axesG, partBT, axesBT, Atk(A, m, i, r), axesA
                    )
                    result[0] -= partBT
                    del partBT, axesG, axesBT
                if (fmc[i][1] < m < Rl):
                    partBB = unfold(partBB, i)[::IC_stepper]
                    result[0] -= np.dot(Bkl(lro_dict, i, m, full=True, docopy=False), partBB.T)
            result[0] = np.dot(result[0], H.T)
            result[0], tmp = np.linalg.qr(result[0])
            #result[1] = np.dot(result[1], tmp.T)
            del tmp
        
        else:
            result = np.hstack(result)
    else:
        result = []
        for s in range(Rl):
            H = np.ones([L[s], L[s]], dtype=dtype)
            IB_stepper = np.sum((L[s])**np.arange(P))
            IB_stepper = int(IB_stepper)
            partBD = T.copy()
            #Bis = Bkl(lro_dict, i, s, full=True, docopy=False)
            for m in range(max(Rt, Rl)):
                if (m < Rl) and (m!=s):
                    partBB = np.ones([L[m], L[s]], dtype=dtype)
                if (tFlag) and (m < Rt):
                    partBT = G[m].copy()
                if cFlag and (m==0):
                    partBC = np.ones([Rc, L[s]], dtype=dtype)
                for k in range(d):
                    if k == i:
                        if (tFlag) and (m < Rt):
                            partBT = prodTenMat(partBT, Atk(A, m, k, r), k)
                        continue
                    if k < P:
                        Bks = Bkl(lro_dict, k, s, full=True, docopy=False)
                    else:
                        Bks = B[k][:, s:s+1].copy() ## sqrt
                        if sqrt:
                            Bks /= (L[s]**0.5) ## sqrt
                    # partBB (always)
                    if (m < Rl) and (m != s):
                        if k < P:
                            partBB *= np.dot(Bkl(lro_dict, k, m, full=True, docopy=False).T, Bks)
                        else:
                            partBB *= np.dot(B[k][:, m:m+1].T, Bks) ## sqrt
                            if sqrt:
                                partBB /= (L[m]**0.5)
                    if (m == 0):
                        #print H
                        # partBC
                        if cFlag:
                            if k < P:
                                partBC *= np.dot(C[k].T, Bks)
                            else:
                                partBC *= np.dot(C[k].T, np.tile(Bks, (1, L[s]))) ## sqrt
                        # part BD (always)
                        if k < P:
                            partBD = prodTenMat(partBD, Bks.T, k)
                            H *= np.dot(Bks.T, Bks)
                        else:
                            partBD = prodTenMat(partBD, Bks.T, k) ## sqrt
                            #partBD = prodTenMat(partBD, np.ones([L[s], 1]), k)
                            H *= (Bks**2.).sum() ## sqrt
                    # partBT
                    if tFlag and (m < Rt):
                        if k < P:
                            partBT = prodTenMat(partBT, np.dot(Bks.T, Atk(A, m, k, r)), k)
                        else:
                            partBT = prodTenMat(partBT, np.dot(Bks.T, Atk(A, m, k, r)), k)
                            #partBT = prodTenMat(partBT, np.ones([L[s], 1]), k) ## sqrt
                if m == 0:
                    H = np.linalg.pinv(H)
                    partBD = unfold(partBD, i)[:, ::IB_stepper]
                    #partBD = np.dot(partBD, H.T) # actually, H = H^T
                    res = partBD.copy()
                    del partBD
                    if cFlag:
                        #partBC = np.dot(partBC, H.T)
                        res -= np.dot(C[i], partBC)
                        del partBC
                if tFlag and (m < Rt):
                    partBT = unfold(partBT, i)[:, ::IC_stepper]
                    #partBT = np.dot(partBT, H.T)
                    res -= partBT
                    del partBT
                if (m < Rl) and (m!=s):
                    #partBB = np.dot(partBB, H.T)
                    #if i < P:
                    #    res -= np.dot(Bkl(lro_dict, i, m, full=True, docopy=False), partBB)
                    #else:
                    partBB = np.dot(np.tile(B[i][:, m:m+1], [1, L[m]]), partBB)
                    if sqrt:
                        partBB /= (L[m]**0.5)
                    res -= partBB
                    del partBB
            res = np.sum(np.dot(res, H.T), axis=1, keepdims=1)
            if sqrt:
                res /= (L[s]**0.5) ## sqrt
            result.append(res.copy())
        result = np.hstack(result)
    #print np.linalg.norm(result)
    return result

def alsUpdUi_Bsi(
    i,
    T,
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None,
    fullModesRanks=None,
    sqrt=_SQRT_CLOBAL
):
    d = T.ndim
    n = T.shape
    dtype = T.dtype
    cFlag = canonical_dict is not None
    lFlag = lro_dict is not None
    tFlag = tucker_dict is not None
    assert lFlag
    assert 'fullModesConfig' in lro_dict.keys()
    fmc = copy.deepcopy(lro_dict['fullModesConfig'])
    assert (fmc is not None) and (fmc[i] is not None)
    if cFlag:
        Rc = canonical_dict['Rc']
        C = canonical_dict['C']
    if lFlag:
        L = lro_dict['L']
        M = int(sum(L))
        Rl = len(L)
        P = lro_dict['P']
        B = lro_dict['B']
        E = lro_dict['E']
        
    Rt = 1
    if tFlag:
        A = tucker_dict['A']
        G = tucker_dict['G']
        r = tucker_dict['r']
        Rt = r.shape[0]
    step_marg = i < P
    
    WtVt = np.zeros([fmc[i][0], int(np.prod(n[:i]+n[i+1:]))], dtype=dtype)
    ind = 0
    for s in range(fmc[i][1]):
        #offset = n[s]
        tmp = Bkl(lro_dict, i, s, full=False, docopy=False)
        Vt = None
        for k in range(d):
            if k == i:
                continue
            if Vt is None:
                if k < P:
                    Vt = Bkl(lro_dict, k, s, full=True, docopy=False)
                else:
                    Vt = np.tile(B[k][:, s:s+1], [1, L[s]]) ## sqrt
                    if sqrt:
                        Vt /= (L[s]**0.5)
            else:
                if k < P:
                    Vt = krp_cw(Bkl(lro_dict, k, s, full=True, docopy=False), Vt)
                else:
                    tmpVt = np.tile(B[k][:, s:s+1], [1, L[s]])
                    if sqrt:
                        tmpVt /= (L[s]**0.5)
                    Vt = krp_cw(tmpVt, Vt) #sqrt
        #WtVt[:, ind:ind+offset] += np.dot(tmp[1].T, Vt.T)
        WtVt += np.dot(tmp[1].T, Vt.T)
        #ind += offset
    del Vt
    H = np.dot(WtVt, WtVt.T)
    H = np.linalg.pinv(H)                
    #partBB = np.ones([L[s], fmc[i][0]])
    shapeWtVt = [fmc[i][0]] + list(n[:i])+list(n[i+1:])
    WtVt = reshape(WtVt, shapeWtVt)
    axes = [i] + range(i) + range(i+1, d)
    WtVt = np.transpose(WtVt, axes)
    #Bis = Bkl(lro_dict, i, s, full=True, docopy=False)
    for m in range(max(Rt, Rl)):
        IB_stepper = int(np.sum((L[m])**np.arange(P-step_marg)))
        if (tFlag) and (m < Rt):
            partBT = WtVt.copy()
        if (fmc[i][1] < m < Rl):
            partBB = WtVt.copy()
        if (cFlag) and (m == 0):
            partBC = WtVt.copy()
        for k in range(d):
            if k == i:
                continue
            # partBB (always)
            if (fmc[i][1] < m < Rl) and (k != i):
                if k < P:
                    partBB = prodTenMat(partBB, Bkl(lro_dict, k, m, full=True, docopy=False).T, k)
                else:
                    tmpBB = B[k][:, m:m+1].copy()
                    if sqrt:
                        tmpBB /= (L[m]**0.5)
                    partBB = prodTenMat(partBB, tmpBB.T, k) ## sqrt
            # partBT
            if tFlag and (m < Rt):
                partBT = prodTenMat(partBT, Atk(A, m, k, r).T, k)
            if (m == 0):
                # partBC
                if cFlag:
                    partBC = prodTenMat(partBC, C[k].T, k)
        if m == 0:
            # part BD (always)
            axesT = list(range(i)) + [d] + list(range(i+1, d))
            axesBD = list(range(i)) + [d+1] + list(range(i+1, d))
            result = np.einsum(T, axesT, WtVt, axesBD)
            del axesBD, axesT
            if cFlag:
                IC_stepper2 = np.sum((Rc)**np.arange(d-1))
                IC_stepper2 = int(IC_stepper2)
                partBC = unfold(partBC, i)[:, ::IC_stepper2]
                result -= np.dot(C[i], partBC.T)
                del partBC

        if tFlag and (m < Rt):
            axesA = [d+1, d]
            axesBT = list(range(i)) + [d+2] + list(range(i+1, d))
            axesG = list(range(i)) + [d] + list(range(i+1, d))
            partBT = np.einsum(
                G[m], axesG, partBT, axesBT, Atk(A, m, i, r), axesA
            )
            result -= partBT
            del partBT, axesG, axesBT
        if (fmc[i][1] < m < Rl):
            partBB = unfold(partBB, i)[::IB_stepper]
            result -= np.dot(Bkl(lro_dict, i, m, full=True, docopy=False), partBB.T)
    result = np.dot(result, H.T)
    result, tmp = np.linalg.qr(result)
    #result[1] = np.dot(result[1], tmp.T)
    del tmp
    return result

def alsUpdBsi(
    i,
    s,
    T,
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None,
    fullModesRanks=None,
    sqrt=_SQRT_CLOBAL
):
    d = T.ndim
    n = T.shape
    dtype = T.dtype
    cFlag = canonical_dict is not None
    lFlag = lro_dict is not None
    tFlag = tucker_dict is not None
    assert lFlag
    if cFlag:
        Rc = canonical_dict['Rc']
        C = canonical_dict['C']
    if lFlag:
        L = lro_dict['L']
        M = int(sum(L))
        Rl = len(L)
        P = lro_dict['P']
        B = lro_dict['B']
        E = lro_dict['E']
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
        else:
            fmc = None
    Rt = 1
    if tFlag:
        A = tucker_dict['A']
        G = tucker_dict['G']
        r = tucker_dict['r']
        Rt = r.shape[0]

    H = np.ones([L[s], L[s]], dtype=dtype)
    step_marg = i < P
    IB_stepper = np.sum((L[s])**np.arange(P-step_marg))
    IB_stepper = int(IB_stepper)
    partBD = T.copy()
    for m in range(max(Rt, Rl)):
        if (tFlag) and (m < Rt):
            partBT = G[m].copy()
        if (m < Rl) and (m!=s):
            partBB = np.ones([L[m], L[s]], dtype=dtype)
        if cFlag and (m == 0):
            partBC = np.ones([Rc, L[s]], dtype=dtype)
        for k in range(d):
            if k == i:
                if (tFlag) and (m < Rt):
                    partBT = prodTenMat(partBT, Atk(A, m, k, r), k)
                continue
            if k < P:
                Bks = Bkl(lro_dict, k, s, full=True, docopy=False)
            else:
                Bks = B[k][:, s:s+1].copy()
                if sqrt:
                    Bks /= (L[s]**0.5) ## copy?
            # partBB (always)
            if (m < Rl) and (m != s):
                if k < P:
                    partBB *= np.dot(Bkl(lro_dict, k, m, full=True, docopy=False).T, Bks)
                else:
                    tmp = np.dot(B[k][:, m:m+1].T, Bks)
                    if sqrt:
                        tmp /= (L[m]**0.5)
                    partBB *= tmp ## sqrt
            if (m == 0):
                # partBC
                if cFlag:
                    if k < P:
                        partBC *= np.dot(C[k].T, Bks)
                    else:
                        partBC *= np.dot(C[k].T, np.tile(Bks, [1, L[s]]))
                # part BD (always)
                if k < P:
                    partBD = prodTenMat(partBD, Bks.T, k)
                    H *= np.dot(Bks.T, Bks)
                else:
                    partBD = prodTenMat(partBD, Bks.T, k) ## sqrt
                    #partBD = prodTenMat(partBD, np.ones([L[s], 1]), k)
                    H *= (Bks**2.).sum() ## sqrt
            # partBT
            if tFlag and (m < Rt):
                if k < P:
                    partBT = prodTenMat(partBT, np.dot(Bks.T, Atk(A, m, k, r)), k)
                else:
                    partBT = prodTenMat(partBT, np.dot(Bks.T, Atk(A, m, k, r)), k)
                    #partBT = prodTenMat(partBT, np.ones([L[s], 1]), k) ## sqrt
        if m == 0:
            H = np.linalg.pinv(H)
            partBD = unfold(partBD, i)[:, ::IB_stepper]
            #partBD = np.dot(partBD, H.T) # actually, H = H^T
            res = partBD.copy()
            del partBD
            if cFlag:
                #partBC = np.dot(partBC, H.T)
                res -= np.dot(C[i], partBC)
                del partBC
        if tFlag and (m < Rt):
            partBT = unfold(partBT, i)[:, ::IB_stepper]
            #partBT = np.dot(partBT, H.T)
            res -= partBT
            del partBT
        if (m < Rl) and (m!=s):
            #partBB = np.dot(partBB, H.T)
            if i < P:
                res -= np.dot(Bkl(lro_dict, i, m, full=True, docopy=False), partBB)
            else:
                tmp = np.tile(B[i][:, m:m+1], [1, L[m]])
                if sqrt:
                    tmp /= (L[m]**0.5)
                res -= np.dot(tmp, partBB)
            del partBB
    res = np.dot(res, H.T)
    if i < P:
        # if low-rank constrained mode
        if (fmc is not None) and (fmc[i] is not None) and (s < fmc[i][1]):
            Ui, _ = Bkl(lro_dict, i, s, full=0, docopy=False)
            res = np.dot(Ui.T, res).T
    else:
        res = np.sum(res, axis=1, keepdims=1)
        if sqrt:
            res /= (L[s]**0.5) ## sqrt
    return res


def alsUpdCi(
    i,
    T,
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None,
    fullModesRanks=None,
    sqrt=_SQRT_CLOBAL
):
    d = T.ndim
    n = T.shape
    dtype = T.dtype
    cFlag = canonical_dict is not None
    lFlag = lro_dict is not None
    tFlag = tucker_dict is not None
    assert cFlag or lFlag
    if cFlag:
        Rc = canonical_dict['Rc']
        C = canonical_dict['C']
    Rl = 1
    if lFlag:
        L = lro_dict['L']
        Rl = len(L)
        M = int(sum(L))
        P = lro_dict['P']
        B = lro_dict['B']
        E = lro_dict['E']
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
        else:
            fmc = None
    Rt = 1
    if tFlag:
        A = tucker_dict['A']
        G = tucker_dict['G']
        r = tucker_dict['r']
        Rt = r.shape[0]
    partCD = T.copy()
    H = np.ones([Rc, Rc], dtype=dtype)
    IC_stepper = np.sum((Rc)**np.arange(d-1))
    IC_stepper = int(IC_stepper)
    for m in range(max(Rl, Rt)):
        if lFlag and (m < Rl):
            partCB = np.ones([L[m], Rc], dtype=dtype)
        if tFlag and (m < Rt):
            partCT = G[m].copy()
        for k in range(d):
            if k == i:
                if tFlag and (m < Rt):
                    partCT = prodTenMat(partCT, Atk(A, m, k, r), k)
                continue
            if lFlag:
                if k < P:
                    tmp = Bkl(lro_dict, k, m, full=True, docopy=False)
                    partCB *= np.dot(tmp.T, C[k])
                else:
                    tmp = np.tile(B[k][:, m:m+1], [1, L[m]])
                    if sqrt:
                        tmp /= (L[m]**0.5)
                    partCB *= np.dot(tmp.T, C[k])
            if tFlag and (m < Rt):
                partCT = prodTenMat(partCT, np.dot(C[k].T, Atk(A, m, k, r)), k)
            if m == 0:
                partCD = prodTenMat(partCD, C[k].T, k)
                H *= np.dot(C[k].T, C[k])
        if m == 0:
            H = np.linalg.pinv(H)
            partCD = unfold(partCD, i)[:, ::IC_stepper]
            #partCD = np.dot(partCD, H.T)
            res = partCD.copy()
            del partCD
        if tFlag and (m < Rt):
            partCT = unfold(partCT, i)[:, ::IC_stepper]
            #partCT = np.dot(partCT, H.T)
            res -= partCT
            del partCT
        if lFlag and (m < Rl):
            if i < P:
                partCB = np.dot(Bkl(lro_dict, i, m, full=True, docopy=False), partCB) 
            else:
                tmp = np.tile(B[i][:, m:m+1], [1, L[m]])
                if sqrt:
                    tmp /= (L[m]**0.5)
                partCB = np.dot(tmp, partCB) 
            res -= partCB
            del partCB
    res = np.dot(res, H.T)
    '''
    if lFlag and (i < P) and (fmc is not None) and (fmc[i] is not None):
        tmp = []
        if cFlag:
            tmp.append( res[:, :Rc] )
            res = res[:, Rc:]
        ind = int(sum(L[:fmc[i][1]]))
        tmp.append( np.dot(np.linalg.pinv(B[i][1]), res[:, :ind].T).T )
        tmp[0], _ = np.linalg.qr(tmp[-1])
        tmp.append( np.dot(tmp[-1].T, res[:, :ind]).T )
        if ind < M:
            tmp.append(res[:, ind:].copy())
        else:
            tmp.append(None)
        del res
        res = tmp
    '''
    return res

def als_loop(
    T,
    sweep,
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None,
    iA=None,
    projector=None,
    sqrt=_SQRT_CLOBAL
):
    d = T.ndim
    n = T.shape
    cFlag = canonical_dict is not None
    lFlag = lro_dict is not None
    tFlag = tucker_dict is not None
    assert cFlag or lFlag or tFlag    
    assert tFlag == (iA is not None)
    if cFlag:
        Rc = canonical_dict['Rc']
    if tFlag:
        r = tucker_dict['r']
        Rt = r.shape[0]
    if lFlag:
        E = lro_dict['E']
        L = lro_dict['L']
        Rl = len(L)
        M = int(sum(L))
        P = lro_dict['P']
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
        else:
            fmc = None
    #Lc = np.ones([1, R])
    # Rc = ?

    if projector is not None:
        activeModes = {}
        if cFlag:
            activeModes['c'] = []
        if lFlag:
            activeModes['l'] = []
        if tFlag:
            activeModes['t'] = {}
            activeModes['t']['c'] = []
            activeModes['t']['f'] = []
    if cFlag:
        for i in range(d):
            tmp = alsUpdCi(i, T, canonical_dict, lro_dict, tucker_dict)
            norm_i = np.linalg.norm(tmp, axis=0, keepdims=True)
            tmp /= norm_i
            canonical_dict['C'][i] = tmp.copy()
            if projector is not None:
                activeModes['c'] = [i]
                canonical_dict, lro_dict, tucker_dict = projector(canonical_dict, lro_dict, tucker_dict, activeModes)
                activeModes['c'] = []
            if i <= (d-1):
                nrm = norm_i**(1./d)
                if (nrm < np.spacing(1)).any():
                    print(f"Zero factors at C/B[{i}]")
                for j in range(d):
                    canonical_dict['C'][j] *= nrm
                    '''
                    if j == i:
                        tmp = canonical_dict['C'][j].copy()
                        CtC[j] = np.dot(tmp.T, tmp)
                    else:
                        CtC[j] *= nrm
                        CtC[j] *= nrm.T
                    '''
    if lFlag:
        for i in range(d):
            '''
            lp = LineProfiler()
            lp_wrapper = lp(alsUpdBi)
            lp_wrapper(i, T, canonical_dict, lro_dict, tucker_dict)
            lp.print_stats()
            exit()
            '''
            
            if (i < P) and (fmc is not None) and (fmc[i] is not None):
                tmp = alsUpdUi_Bsi(i, T, canonical_dict, lro_dict, tucker_dict)
                lro_dict['B'][i][0] = tmp.copy()
            for s in range(Rl):
                tmp = alsUpdBsi(i, s, T, canonical_dict, lro_dict, tucker_dict)
                #tmp = tmp2.copy()
                norm_i = np.linalg.norm(tmp, axis=0, keepdims=True)
                tmp /= norm_i
                if i < P:
                    if (fmc is not None) and (fmc[i] is not None):
                        if s < fmc[i][1]:
                            ind = int(sum(L[:s]))
                            lro_dict['B'][i][1][ind:ind+L[s], :] = tmp.copy()
                        else:
                            ind = int(sum(L[fmc[i][1]:s]))
                            lro_dict['B'][i][2][:, ind:ind+L[s]] = tmp.copy()
                    else:
                        ind = int(sum(L[:s]))
                        lro_dict['B'][i][:, ind:ind+L[s]] = tmp.copy()
                        for j in range(P):
                            lro_dict['B'][i][:, ind:ind+L[s]] *= norm_i**(1./P)
                else:
                    lro_dict['B'][i][:, s:s+1] = tmp.copy()
            
            '''
            if i < P:
                for s in range(Rl):
                    ind = int(sum(L[:s]))
                    y = tmp[:, ind:ind+L[s]].copy()
                    tmp[:, ind:ind+L[s]], _ = np.linalg.qr(y)
                #tmp, _ = np.linalg.qr(tmp)
            else:
                norm_i = np.linalg.norm(tmp, axis=0, keepdims=True)
                tmp /= norm_i
            
            if i < P:
                if (fmc is not None) and (fmc[i] is not None):
                    norm_i = np.linalg.norm(tmp[1], axis=1, keepdims=True)
                    tmp[1] /= norm_i
                    norm_i = norm_i.T
                    if tmp[2] is not None:
                        normBi2 = np.linalg.norm(tmp[2], axis=0, keepdims=True)
                        tmp[2] /= normBi2
                        norm_i = np.hstack([norm_i, normBi2])
                    lro_dict['B'][i] = [tmp[0], tmp[1], tmp[2]]
                else:
                    norm_i = np.linalg.norm(tmp, axis=0, keepdims=True)
                    lro_dict['B'][i] = tmp.copy() #/ norm_i
                    #norm_i2 = np.dot(norm_i, E*np.reshape(np.sqrt(L), [-1,1]).T)
            else:
                lro_dict['B'][i] = tmp.copy() #/ reshape(np.sqrt(L), [1, -1])#np.dot(tmp, E.T)
                #norm_i = np.linalg.norm(lro_dict['B'][i], axis=0, keepdims=True)
                #lro_dict['B'][i] /= norm_i
                #norm_i = np.ones([1, sum(L)])
                #norm_i = np.repeat(np.sqrt(L), L)
                #norm_i = np.dot(norm_i*np.sqrt(reshape(L, [1, -1])), E)
                #print norm_i
                #norm_i = np.dot(norm_i, E)
                #norm_i2 = norm_i.copy()
                #norm_i = np.repeat(norm_i, L)
                
                
                
                    
                #tmp /= norm_i

            #Lc *= normCi
            
            if i < P:
                #norm_i = np.linalg.norm(lro_dict['B'][i], axis=0, keepdims=True)
                nrm = norm_i**(1./P)
                if (nrm < np.spacing(1)).any():
                    print "Zero factors at C/B[%d]" % (i)
                for j in range(d):
                    if j < P:
                        if (fmc is not None) and (fmc[j] is not None):
                            #lro_dict['B'][j][-1] *= (norm_i)**(1./P)
                            ind = int(sum(L[:fmc[j][1]]))
                            lro_dict['B'][j][1] *= ((norm_i[:, :ind])**(1./P)).T
                            if ind < M:
                                lro_dict['B'][j][2] *= (norm_i[:, ind:])**(1./P)
                        else:
                            lro_dict['B'][j] *= nrm#(norm_i)**(1./P)
                    #elif i >=P:
                    #    lro_dict['B'][j] *= norm_i2**(1./(d-P))#np.dot(nrm, E.T)
            '''
            if projector is not None:
                activeModes['l'] = [i]
                canonical_dict, lro_dict, tucker_dict = projector(canonical_dict, lro_dict, tucker_dict, activeModes)
                activeModes['l'] = []

    if tFlag:
        for i in range(d):
            for m in range(Rt):
                tmp = alsUpdAim(i, m, T, iA, canonical_dict, lro_dict, tucker_dict, orthog='qr')#
                #tmp = alsUpdAim(i, m, T, iA, canonical_dict, lro_dict, tucker_dict, orthog=None)
                norm_i = np.linalg.norm(tmp, axis=0, keepdims=True)
                ind1 = int(sum(r[:m, i]))
                ind2 = int(sum(r[:m+1, i]))
                tucker_dict['A'][i][:, ind1:ind2] =  tmp.copy() / norm_i
                '''
                shp = [1]*(i) + [-1] + [1]*(d-i-1)
                norm_i = reshape(vec(norm_i), shp)
                tucker_dict['G'][m] *= np.sqrt(norm_i)
                '''
                if projector is not None:
                    activeModes['t']['c'] = [m]
                    activeModes['t']['f'] = [i]
                    canonical_dict, lro_dict, tucker_dict = projector(
                        canonical_dict, lro_dict, tucker_dict, activeModes
                    )
                    activeModes['t']['c'] = []
                    activeModes['t']['f'] = []
                iA[i][:, ind1:ind2] = np.linalg.pinv(tucker_dict['A'][i][:, ind1:ind2]).T
                del tmp
                if i == (d-1):
                    tucker_dict['G'][m] = alsUpdGm(m, T, iA, canonical_dict, lro_dict, tucker_dict)
                    gnrm = np.linalg.norm(tucker_dict['G'][m])
                    tucker_dict['G'][m] /= gnrm
                    '''
                    gnrm = gnrm**(1./d)
                    for j in range(d):
                        ind1 = int(sum(r[:m, j]))
                        ind2 = int(sum(r[:m+1, j]))
                        tucker_dict['A'][j][:, ind1:ind2] *= gnrm
                        iA[j][:, ind1:ind2] /= gnrm
                    '''
                    if projector is not None:
                        activeModes['t']['c'] = [m]
                        canonical_dict, lro_dict, tucker_dict = projector(
                            canonical_dict, lro_dict, tucker_dict, activeModes
                        )
                        activeModes['t']['c'] = []
    return canonical_dict, lro_dict, tucker_dict, iA

# ToDo: dictionarize backtrack hyper-parameters
def tcd(
           T,
        # dicts with parameters 
           canonical_dict=None,
           lro_dict=None,
           tucker_dict=None,
        # initial solution
           x0=None,
        # number of iterations
           maxitnum=20,
           maxInnerIt=100,
        # main solver parameters
           method = 'gd',
           constraints=None,
           doPrec=0,
        # tolerances
           tolRes=1e-5,
           tolGrad=1e-6,
           tolSwamp=1e-6,
        # backtracking
           backtrack=True,
        # verbosing
           verbose=False,
        # Levenberg-Marquardt parameters
           epsilonLM=1e-8,
           lmSetup='Quadratic',
           muInit=1.,
        # gradient descent parameters
           regTGD=1e-8,
           regPGD=1e-8,
           accelerate=None,
        # simmulated annealing
           doSA=False,
        # Trust region setup
           curvature=True,
           gammaVel=0.5,
           trStep=scg_step,
           trDelta0=1e-1,
           trDeltaMax=1e8,
           trEta=0.2,
        # CG parameters
           betaCG='fr'
    ):
    eps = np.spacing(1.)
    # check if CP/Lr-one/Tucker parameters are given
    assert (canonical_dict is not None) or (lro_dict is not None) or (tucker_dict is not None),\
        "tcd(): All-empty decomposition parameters! (canonical_dict, lro_dict, tucker_dict)"
    # retrieving information from tensor
    n = T.shape
    d = T.ndim
    dtype = T.dtype
    # scaling input tensor
    normT = np.linalg.norm(T)
    T1 = T.copy() / normT
    # dealing with parameters of decomposition
    cFlag = False
    if canonical_dict is not None:
        cFlag = True
        Rc = canonical_dict['Rc']
    lFlag = False
    if lro_dict is not None:
        lFlag = True
        P = lro_dict['P']
        assert 0 <= P < d, "You do not need (Lr, 1) model with such P: %d" % (P)
        L = lro_dict['L']
        M = int(np.sum(L))
        if 'E' in lro_dict.keys():
            E = lro_dict['E']
        else:
            E = computeE(L)
            lro_dict['E'] = E
        fmc = None
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
    tFlag = False
    if tucker_dict is not None:
        tFlag = True
        r = tucker_dict['r']
        Rt = r.shape[0]
    # setting up initialization
    if x0 is None:
        C, B, A, G = initializeCBT(
            n,
            canonical_param=canonical_dict,
            lro_param=lro_dict,
            tucker_param=tucker_dict,
            rtype='normal',
            normalize=True
        )
    else:
        if tFlag:
            assert 'A' in x0.keys(), "tcd(): you must specify x0['A'] if tucker_dict is given!"
            assert 'G' in x0.keys(), "tcd(): you must specify x0['G'] if tucker_dict is given!"
            A = x0['A']
            G = x0['G']
        if lFlag:
            assert 'B' in x0.keys(), "tcd(): you must specify x0['B'] if lro_dict is given!"
            B = x0['B']
        if cFlag:
            assert 'C' in x0.keys(), "tcd(): you must specify x0['C'] if lro_dict is given!"
            C = x0['C']
    if cFlag:
        canonical_dict['C'] = C
    if lFlag:
        lro_dict['B'] = B
    if tFlag:
        tucker_dict['A'] = A
        tucker_dict['G'] = G
    # dealing with constraints
    projector = None
    LagrangeMultipliersMethod = False
    if constraints is not None:
        if constraints['type'] == 'projected':
            if method == 'als':
                def projector(canonical_dict, lro_dict, tucker_dict, activeMode):
                    cdN, ldN, tdN = constraints['projector'](
                        canonical_dict,
                        lro_dict,
                        tucker_dict,
                        dimnums=constraints['dimnum'],
                        vect=False,
                        activeModes=activeMode
                    )
                    return cdN, ldN, tdN
            else:
                def projector(z):
                    C1, B1, A1, G1 = fvec2fcore(z, n, canonical_dict, lro_dict, tucker_dict, full_result=True)
                    cdN, ldN, tdN = refactorizeDicts(canonical_dict, lro_dict, tucker_dict, C1, B1, A1, G1)
                    rv = constraints['projector'](cdN, ldN, tdN, dimnums=constraints['dimnum'], vect=True)
                    return rv
        elif constraints['type'] == 'Lagrange':
            # HARD VALUES
            pcum = float(n[-1])
            pmin = 0.5
            multChecker = None
            if ('multChecker' in constraints.keys()) and (constraints['multChecker'] is not None):
                multChecker = lambda fv, ml: constraints['multChecker'](fv, ml, n, pmin)
            if not (method in ['tr', 'rn', 'gn', 'lm']):
                raise ValueError(
                    'Current implementation of Lagrange multipliers method'
                    'needs (qusi-)Newton approach, not %s method' % (method)
                )
            LagrangeMultipliersMethod = True
            multSize = constraints['multSize']
            if ('multInit' in constraints.keys()) and (constraints['multInit'] is not None):
                multipliersLMM = constraints['multInit'].copy()
            else:
                multipliersLMM = np.random.uniform(0.001, 0.005, size=(multSize)).astype(dtype)
            wholeDecParamNum = 0.
            if cFlag:
                wholeDecParamNum += np.sum(n)*Rc
            if lFlag:
                if fmc is not None:
                    fmcInd = np.where(np.array(fmc) != None)[0]
                    for k in range(P):
                        if not (k in fmcInd):
                            wholeDecParamNum += M*n[k]
                        else:
                            indL = int(sum(L[:fmc[k][1]]))
                            wholeDecParamNum += (n[k]+indL)*fmc[k][0]
                            if indL != M:
                                wholeDecParamNum += (M-indL)*n[k]
                else:
                    wholeDecParamNum += np.sum(n[:P])*M
                wholeDecParamNum += np.sum(n[P:])*len(L)
               
            if tFlag:
                wholeDecParamNum += np.sum(np.prod(r, axis=1)) + np.sum(np.sum(r, axis=0)*np.array(n))
            wholeDecParamNum = int(wholeDecParamNum)
            def HesMvAL(z):
                C1, B1, A1, G1 = fvec2fcore(
                    z[:wholeDecParamNum], n, canonical_dict, lro_dict, tucker_dict, full_result=True
                )
                cdN, ldN, tdN = refactorizeDicts(canonical_dict, lro_dict, tucker_dict, C1, B1, A1, G1)
                rv = constraints['hessian_matvec'](
                    z,
                    multipliersLMM,
                    canonical_dict=cdN,
                    lro_dict=ldN,
                    tucker_dict=tdN,
                    dimnums=constraints['dimnum'],
                    vect=True
                )
                return rv
            def FunAL(z, multipliers):
                
                C1, B1, A1, G1 = fvec2fcore(z, n, canonical_dict, lro_dict, tucker_dict, full_result=True)
                cdN, ldN, tdN = refactorizeDicts(canonical_dict, lro_dict, tucker_dict, C1, B1, A1, G1)
                rv = constraints['functional'](multipliers, pcum, pmin, cdN, ldN, tdN, dimnums=constraints['dimnum'])
                return rv
            if constraints['projector'] is not None:
                def projector(z):
                    C1, B1, A1, G1 = fvec2fcore(
                        z[:wholeDecParamNum], n, canonical_dict, lro_dict, tucker_dict, full_result=True
                    )
                    cdN, ldN, tdN = refactorizeDicts(canonical_dict, lro_dict, tucker_dict, C1, B1, A1, G1)
                    rv = constraints['projector'](cdN, ldN, tdN, dimnums=constraints['dimnum'], vect=True)
                    return np.concatenate([rv, z[wholeDecParamNum:]])
        if cFlag:
            C = canonical_dict['C']
        if lFlag:
            B = lro_dict['B']
        if tFlag:
            A = tucker_dict['A']
            G = tucker_dict['G']
    # computing self-, cross-products of factors
    iA = None
    if tFlag and (method == 'als'):
        iA = []
        for k in range(d):
            tmp = np.zeros([n[k], int(sum(r[:, k]))], dtype=dtype)
            ind = 0
            for m in range(Rt):
                offset = r[m, k]
                tmp[:, ind:ind+offset] = np.linalg.pinv(Atk(A, m, k, r)).T
                ind += offset
            iA.append(tmp.copy())
            del tmp
    
    # define preconditioner and (quasi-) Hessian matvecsand several other functions
    def prec(x):
        if not doPrec:
            return x
        rv = BJprecTCD(x, canonical_dict, lro_dict, tucker_dict, CtC, AtA)
        return rv
    mu_k = 0.
    def mv(x):
        rv = np.zeros(x.size, dtype=x.dtype)
        if LagrangeMultipliersMethod:
            rv[:wholeDecParamNum] += mvHes(
                x[:wholeDecParamNum], T1, canonical_dict, lro_dict, tucker_dict, curvature, return_vector=True
            )
            rv += HesMvAL(x)
        else:
            rv += mvHes(x, T1, canonical_dict, lro_dict, tucker_dict, curvature, return_vector=True)
        '''if lmSetup == 'Levenberg':
            rv += mu_k * x * GramDiag(U, E, L, P)
        else:
        '''            
        if (method == 'lm') or (method == 'rn'):
            rv += mu_k * x
        return rv
    def functional(z):
        C1, B1, A1, G1 = fvec2fcore(z, n, canonical_dict, lro_dict, tucker_dict, full_result=True)
        cdN, ldN, tdN = refactorizeDicts(canonical_dict, lro_dict, tucker_dict, C1, B1, A1, G1)
        
        rv = 0.5*np.linalg.norm(recover(n, cdN, ldN, tdN) - T1)**2.
        del C1, B1, A1, G1, cdN, ldN, tdN
        return rv
    # generating lists with information about process
    flist = [functional(fcore2fvec(n, canonical_dict, lro_dict, tucker_dict))]
    glist = []
    gnorm = np.nan
    # setting up method's parameters
    if method == 'tr':
        trDeltaK = trDelta0
        fk = flist[-1]
        g = None
        x = None
        fmk = lambda z: fk + np.sum(g*(z-x)) + 0.5*np.sum(mv(z-x)*(z-x))
        if trStep == 'dogleg':
            trStepF = lambda par1, par2, par3: dogleg_step(
                par1, par2, par3, prec=prec, maxInnerIt=maxInnerIt, tol=tolRes*0.1
            )
            curvature=False
        elif trStep == 'scg':
            trStepF = lambda par1, par2, par3: scg_step(
                par1, par2, par3, eps=tolGrad*0.1, maxInnerIt=maxInnerIt
            )
    if (method == 'tr') or (method == 'lm'):
        skippedSweep = False
    if (method == 'gn') or (method == 'lm'):
        curvature = False
    if (method == 'lm'):
        doPrec = False
        if (lmSetup == 'Quadratic') or (lmSetup == 'Nielsen'):
            jtjdiag_max = JtJ_diag(n, canonical_dict, lro_dict, tucker_dict, which='max')
            mu_k = muInit*jtjdiag_max
            nu = 2.
        else:
            raise NotImplementedError
    # if simulated annealing is used
    if doSA:
        Temperature_min = 1e-5
        Temperature_max = 1e2
        Temperature_f = -np.log(Temperature_max/Temperature_min)
    # main loop
    for sweep in range(maxitnum):
        if method == 'als':
            canonical_dict, lro_dict, tucker_dict, iA = als_loop(
                T1, sweep, canonical_dict, lro_dict, tucker_dict, iA, projector
            )
            xnew = fcore2fvec(n, canonical_dict, lro_dict, tucker_dict)
        elif method =='gd':
            x = fcore2fvec(n, canonical_dict, lro_dict, tucker_dict)
            g = jacTCD_matvec(x, T1, canonical_dict, lro_dict, tucker_dict, return_vector=True)
            gnorm = np.linalg.norm(g)
            
            if regTGD is not None:
                g += regTGD*x
            if regPGD is not None:
                xold = x.copy()
                if sweep > 0:
                    g += regPGD*(x-xold)
            if backtrack:
                _, _, alpha = backtrackArmijo(functional, x, -g, g, beta=0.999, sigma=1e-5, beta_lim=tolRes, sign=None)
            xnew = x - alpha*g
        elif method =='cg':
            x = fcore2fvec(n, canonical_dict, lro_dict, tucker_dict)
            g = jacTCD_matvec(x, T1, canonical_dict, lro_dict, tucker_dict, return_vector=True)
            gnorm = np.linalg.norm(g)
            if sweep == 0:
                p = -g
            else:
                if betaCG == 'fr':
                    beta = np.sum(g**2.) / np.sum(gold**2.)
                elif betaCG == 'pr':
                    beta = np.sum(g*(g-gold)) / np.sum(gold**2.)
                elif betaCG == 'hs':
                    beta = np.sum(g*(g-gold)) / np.sum(p*(g-gold))
                elif betaCG == 'dy':
                    beta = np.sum(g**2.) / np.sum(p*(g-gold))
                beta = max(beta, 0.)
                p = -g + beta*p
            if backtrack:
                _, _, alpha = backtrackArmijo(functional, x, p, g, beta=0.999, sigma=1e-5, beta_lim=tolRes, sign=None)
            xnew = x + alpha*p
            gold = g.copy()
        elif method == 'gn':
            x = fcore2fvec(n, canonical_dict, lro_dict, tucker_dict)
            g = jacTCD_matvec(x, T1, canonical_dict, lro_dict, tucker_dict, return_vector=True)
            gnorm = np.linalg.norm(g)
            if LagrangeMultipliersMethod:
                temp = FunAL(x, multipliersLMM)*multipliersLMM
                x = np.concatenate([x, multipliersLMM])
                g = np.concatenate([g, temp])
            
            p, infor = conjugate_gradient_method(mv, -g+mv(x), x0=x.copy(), iM=prec, maxitnum=maxInnerIt, tol=tolRes*0.1)
            p -= x
            if backtrack:
                xnew, _, _ = backtrackArmijo(
                    functional, x, p, g, beta=0.995, sigma=0.005, beta_lim=1e-11, normT=normT, sign=None
                )
            if LagrangeMultipliersMethod:
                multipliersLMM_new = xnew[wholeDecParamNum: ].copy()
                if multChecker is not None:
                    temp = FunAL(xnew[:wholeDecParamNum], multipliersLMM_new)
                    multipliersLMM_new = multChecker(temp, multipliersLMM_new)
        elif method == 'rn':
            mu_k = 0.
            mu_k = mineigen(mv, tol=1e-5)
            if mu_k > 0:
                mu_k = 0.
            else:
                border = np.log10(sweep + 1.001)
                mu_k = -min(-border, mu_k-border)
            x = fcore2fvec(n, canonical_dict, lro_dict, tucker_dict)
            g = jacTCD_matvec(x, T1, canonical_dict, lro_dict, tucker_dict, return_vector=True)            
            gnorm = np.linalg.norm(g)
            xnew, infor = conjugate_gradient_method(LA, -g+LA(x), x0=x, iM=precM, maxitnum=maxInnerIt, tol=tolRes*0.1)
            p = xnew - x
            xnew, _, _ = backtrackArmijo(functional, x, p, g, beta=0.995, sigma=0.005, beta_lim=1e-11, normT=normT, sign='pos')
        elif method == 'lm':      
            if not skippedSweep:
                x = fcore2fvec(n, canonical_dict, lro_dict, tucker_dict)
                g = jacTCD_matvec(x, T1, canonical_dict, lro_dict, tucker_dict, return_vector=True)
                gnorm = np.linalg.norm(g)
                if LagrangeMultipliersMethod:
                    temp = FunAL(x, multipliersLMM)*multipliersLMM
                    x = np.concatenate([x, multipliersLMM])
                    g = np.concatenate([g, temp])
            xnew, infor = conjugate_gradient_method(mv, -g+mv(x), x0=x, iM=prec, maxitnum=maxInnerIt, tol=tolRes*0.1)
            p = xnew - x
            if LagrangeMultipliersMethod:
                multipliersLMM_new = xnew[wholeDecParamNum: ].copy()
                if multChecker is not None:
                    temp = FunAL(xnew[:wholeDecParamNum], multipliersLMM_new)
                    multipliersLMM_new = multChecker(temp, multipliersLMM_new)
                temp = FunAL(xnew[:wholeDecParamNum], multipliersLMM_new)*multipliersLMM_new
                rho1 = flist[-1] - (functional(xnew[:wholeDecParamNum]) + np.sum(temp))
                p[wholeDecParamNum:] = multipliersLMM_new - x[wholeDecParamNum: ]
            else:
                rho1 = flist[-1] - functional(xnew)
            if lmSetup == 'Quadratic':
                alpha = np.inner(-g, p)
                alpha /= -0.5*rho1 + 2*alpha
                xnew = x + alpha*p
                if LagrangeMultipliersMethod:
                    multipliersLMM_new = xnew[wholeDecParamNum: ].copy()
                    if multChecker is not None:
                        temp = FunAL(xnew[:wholeDecParamNum], multipliersLMM_new)
                        multipliersLMM_new = multChecker(temp, multipliersLMM_new)
                    temp = FunAL(xnew[:wholeDecParamNum], multipliersLMM_new)*multipliersLMM_new
                    rho1 = flist[-1] - (functional(xnew[:wholeDecParamNum]) + np.sum(temp))
                else:
                    rho1 = flist[-1] - functional(xnew)
            
            '''if lmSetup == 'Nielsen':
                rho = rho1/np.inner(mu_k*p*GramDiag(project(Unew), E, L, P) - g, p)
            else:'''
            rho = rho1/np.inner(mu_k*p - g, p)
            if sweep > 0:
                if rho <= epsilonLM:
                    if verbose:
                        print(rho, mu_k, np.linalg.norm(g))
                    
                    if lmSetup == 'Nielsen':
                        mu_k *= nu
                        nu *= 2
                    elif lmSetup == 'Quadratic':
                        mu_k += abs(1./(2*alpha) * rho1)
                        mu_k *= nu
                        nu *= 2
                    if mu_k > 1e6:
                        print('Overflow in mu_k. end')
                        break
                    '''elif lmSetup == 'Levenberg':
                        mu_k = min(mu_k*11, 1e10)'''
                    if not skippedSweep:
                        glist.append(gnorm)
                        skippedSweep = True
                    else:
                        glist.append(glist[-1])
                    flist.append(flist[-1])
                    continue
                else:
                    skippedSweep = False
                    if lmSetup == 'Nielsen': #simple
                        mu_k *= max(1./3, 1 - (2*rho - 1)**3.)
                        nu = 2
                    elif lmSetup == 'Quadratic':
                        mu_k = max(mu_k/(1.+alpha), 1e-7)
                        nu = 2
                    '''elif lmSetup == 'Levenberg': #diag
                        mu_k = max(mu_k/9, 1e-7)'''
        elif method == 'tr':
            if not skippedSweep:                
                x = fcore2fvec(n, canonical_dict, lro_dict, tucker_dict)
                g = jacTCD_matvec(x, T1, canonical_dict, lro_dict, tucker_dict, return_vector=True)
                gnorm = np.linalg.norm(g)
                if LagrangeMultipliersMethod:
                    temp = FunAL(x, multipliersLMM)*multipliersLMM
                    x = np.concatenate([x, multipliersLMM])
                    g = np.concatenate([g, temp])
                
                mk = flist[-1]
                fk = flist[-1]
            
            p = trStepF(g, mv, trDeltaK) ##########
            xnew = x+p
            if LagrangeMultipliersMethod:
                multipliersLMM_new = xnew[wholeDecParamNum: ].copy()
                if multChecker is not None:
                    temp = FunAL(xnew[:wholeDecParamNum], multipliersLMM_new)
                    multipliersLMM_new = multChecker(temp, multipliersLMM_new) #########################
                temp = FunAL(xnew[:wholeDecParamNum], multipliersLMM_new)*multipliersLMM_new
                fxnew = functional(xnew[:wholeDecParamNum]) + np.sum(temp)
                fmkxnew = fmk(xnew)
            else:
                fxnew = functional(xnew)
                fmkxnew = fmk(xnew)
            rho = (fk - fxnew) / (mk - fmkxnew)
            if verbose:
                print(f"\t Itnum: {sweep+1}\t Rho: {rho:.3e}\t Delta: {trDeltaK:.3e}")
            if np.abs(rho) == np.inf:
                print("abs(rho) is too big. Escaping")
                break
            if rho < 0.25:
                trDeltaK = 0.25*trDeltaK
            elif rho > 0.75:
                trDeltaK = min(2*trDeltaK, trDeltaMax)
            if rho <= trEta:
                if not skippedSweep:
                    glist.append(gnorm)
                    skippedSweep = True
                else:
                    glist.append(glist[-1])
                flist.append(flist[-1])
                continue
            else:
                skippedSweep = False
        if doSA:
            Temperature = Temperature_max*np.exp(Temperature_f*(sweep+1.)/maxitnum)
            err = (2.*functional(xnew))**0.5
            coef = (flist[-1] - err) / np.linalg.norm(xnew - x)
            epk = np.random.uniform(-1, 1+np.spacing(1.), size=x.shape).astype(dtype)
            epk *= coef/(2.*np.linalg.norm(epk))
            err2 = (2.*functional(xnew + epk))**0.5
            dE = (flist[-1] - err2)*normT**2.
            random_number = np.random.random().astype(dtype)
            print(f"T={Temperature:.3e}, dE={dE:.3e}, cond={np.exp(dE / Temperature):.3e}, r={random_number:3.e}")
            if (dE >= 0.) and (np.exp(dE / Temperature) < random_number):
                pass
            else:
                xnew += epk
                del epk
        if (method != 'als') and (projector is not None):
            xnew = projector(xnew)
            
        if LagrangeMultipliersMethod:
            multipliersLMM = multipliersLMM_new.copy()
            xnew = xnew[:wholeDecParamNum]
        C, B, A, G = fvec2fcore(xnew, n, canonical_dict, lro_dict, tucker_dict, full_result=True)
        canonical_dict, lro_dict, tucker_dict = refactorizeDicts(canonical_dict, lro_dict, tucker_dict, C, B, A, G)
                    
        err = functional(xnew)
        flist.append(err)
        if method != 'als':
            glist.append(gnorm)
        if verbose:
            print(f"Itnum: {sweep+1}\t Fun.val.: {err:.3e}\t G.norm: {gnorm:.3e}")
        if (err < tolRes) or ((method != 'als') and (gnorm < tolGrad)):
            break
        if sweep > 0:
            if abs(flist[-1] - flist[-2]) < tolSwamp:
                break
    if tFlag:
        for l in range(len(G)):
            tucker_dict['G'][l] *= normT
    if cFlag:
        canonical_dict['C'][-1]*=normT
    if lFlag:
        lro_dict['B'][-1]*=normT
    info = {
        'gnorm': glist,
        'funval': flist
    }
    return canonical_dict, lro_dict, tucker_dict, info
    

                
if __name__ == '__main__':
    Nrun = 1
    maxitnum = 20
    maxInnerIt = 25
    tolRes = 1e-8
    tolGrad = 1e-8
    tolSwamp = 1e-8
    verbose = 1
    constraints = None
    fnm = 'gtcd_primary_test'
    # doSA = 0 # not available yet
    
    d = 3
    n = [20]*d
    #minn=5
    #n = range(minn, minn+d)
    Rc = 5
    Rl = 3
    P = d-1
    fset=3
    #L = [fset]*Rl
    L = list(range(fset, Rl+fset))
    fmc = [None]*d
    fmc[1] = [2, 2]
    fmc[0] = [3, 3]
    #fmc = None
    #L += [1]*Rl
    NTcores = 2
    #'''
    r = 3*np.ones([NTcores, d], dtype=np.int)
    '''
    r = np.array(
        [[2, 3, 4],
         [5, 6, 7],
         [8, 9, 10]]
    )
    '''
    '''
    r = np.array(
        [[2, 3, 4],
         [4, 2, 2]]
    )
    '''
    canonical_dict = {
        'Rc': Rc
    }
    lro_dict = {
        'L': L,
        'P': P,
        'fullModesConfig': fmc
    }
    tucker_dict = {
        'r': r
    }
    
    canonical_dict = None
    #lro_dict = None
    #tucker_dict = None
    
    def alsM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='als', verbose=verbose, 
            regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
    def gdM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad,tolSwamp=tolSwamp, method='gd', backtrack=True, 
            verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
    def gdrtM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='gd', backtrack=True,
            verbose=verbose, regTGD=1e-3, regPGD=None, doSA=0, constraints=constraints
        )
    def gdrpM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='gd', backtrack=True, 
            verbose=verbose, regTGD=None, regPGD=1e-3, doSA=0, constraints=constraints
        )
    def cgfrM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum,
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='cg', betaCG='fr', 
            verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
    def cgprM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='cg', betaCG='pr', 
            verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
    def cghsM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum,
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='cg', betaCG='hs',
            verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
                                                
    def cgdyM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='cg', betaCG='dy', 
            verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
    def gnM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes,tolGrad=tolGrad,tolSwamp=tolSwamp, method='gn', backtrack=True, 
            verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
        )
    def lmqM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='lm', epsilonLM=1e-8,
            lmSetup='Quadratic', muInit=1., verbose=verbose, regTGD=None, regPGD=None,
            doSA=0, constraints=constraints
        )
    def lmnM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='lm', epsilonLM=1e-8,
            lmSetup='Nielsen', muInit=1., verbose=verbose, regTGD=None, regPGD=None,
            doSA=0, constraints=constraints
        )
    def doglegM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            maxInnerIt=maxInnerIt, tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='tr', 
            verbose=verbose, doSA=0, constraints=constraints, trStep='dogleg',
            trDelta0=1.2,trEta=0.23
        )
    def scg_qnM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            maxInnerIt=maxInnerIt, tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='tr', 
            verbose=verbose, doSA=0, constraints=constraints, curvature=0, trStep='scg',
            trDelta0=1.2, trEta=0.23
        )
    def scg_fnM(a, x0, cdN, ldN, tdN):
        return tcd(
            a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
            maxInnerIt=maxInnerIt, tolRes=tolRes, tolGrad=tolGrad, tolSwamp=tolSwamp, method='tr', 
            verbose=verbose, doSA=0, constraints=constraints, curvature=1, trStep='scg',
            trDelta0=1.2, trEta=0.23
        )
    
    methods_names = [
        "ALS", "GD",
        #"GD-rT", "GD-rP",
        "CG-FR", "CG-PR", "CG-HS", "CG-DY", "GN", "LM-Q", "LM-N",
        "TR-DL", "SCG-QN", "SCG-FN"
    ]
    algs = [
        alsM, gdM,
        #gdrtM, gdrpM,
        cgfrM, cgprM, cghsM, cgdyM, gnM, lmqM, lmnM, doglegM, scg_qnM, scg_fnM
    ]
    #methods_names = methods_names[1:]
    #algs = algs[1:]
    resultFV = np.zeros([Nrun, len(algs), maxitnum+1])
    resultGV = np.zeros([Nrun, len(algs), maxitnum])
    resultTime = np.zeros([Nrun, len(algs)])

    for itRun in range(Nrun):
        C, B, A, G = initializeCBT(
            n, canonical_param=canonical_dict, lro_param=lro_dict, tucker_param=tucker_dict,
            rtype='uniform', normalize=True
        )
        if canonical_dict is not None:
            canonical_dict['C'] = C
        if lro_dict is not None:
            lro_dict['B'] = B
        if tucker_dict is not None:
            tucker_dict['A'] = A
            tucker_dict['G'] = G
        a = recover(n, canonical_dict=canonical_dict, lro_dict=lro_dict, tucker_dict=tucker_dict)
        Cinit, Binit, Ainit, Ginit = initializeCBT(
            n, canonical_param=canonical_dict, lro_param=lro_dict, tucker_param=tucker_dict,
            rtype='uniform', normalize=True
        )
        if canonical_dict is not None:
            del canonical_dict['C']
        if lro_dict is not None:
            del lro_dict['B']
        if tucker_dict is not None:
            del tucker_dict['A'], tucker_dict['G']
        x0 = {
            'C': Cinit,
            'B': Binit,
            'A': Ainit,
            'G': Ginit
        }
        times = []
        for itAlg in range(len(algs)):
            cdN = copy.deepcopy(canonical_dict)
            ldN = copy.deepcopy(lro_dict)
            tdN = copy.deepcopy(tucker_dict)
            t0 = time.clock()
            cdN, ldN, tdN, info = algs[itAlg](a, x0, cdN, ldN, tdN)
            t1 = time.clock()
            if len(info['gnorm']) == 0:
                info['gnorm'] = [np.nan]
            print(
                f"Run:{itRun+1}\t Alg: {methods_names[itAlg]}\t Time: {t1-t0:.2f}\t"
                f"Fv: {info['funval'][-1]:.3e}\t Gv: {info['gnorm'][-1]:.3e}"
            )
            resultTime[itRun, itAlg] += t1 - t0
            flist = info['funval']
            glist = info['gnorm']
            flist += max(0, 1+maxitnum-len(flist))*[flist[-1]]
            glist += max(0, maxitnum-len(glist))*[glist[-1]]
            resultFV[itRun, itAlg, :] += np.array(flist)
            resultGV[itRun, itAlg, :] += np.array(glist)
            np.savez_compressed(fnm, resultFV=resultFV, resultGV=resultGV,
                 resultTime=resultTime, itRun=itRun, itAlg=itAlg, lro_dict=lro_dict,
                 tucker_dict=tucker_dict, n=n, methods=methods_names
            )
    
    
    
    
    
    
    
    
    
    
    
