import numpy as np
import time
import copy
from .gtcd import refactorizeDicts, fcore2fvec, fvec2fcore, Bkl
#from tendec.utils import reshape, vec
#from tendec.btd.utils import getModeSize
#from tendec.btd.utils import recBTD, fvec2flist, flist2fvec

#from rules import constant
#from scipy.sparse import block_diag

# (J.T J + lambda*I) p = -1 * gradient

def nn_func(ulist, L, P):
    N = len(ulist)
    fv = 0.
    for k in range(N):
        fv += 0.5*(np.minimum(ulist[k], 0)**2.).sum()
    return fv

def nn_gradient(ulist, L, P):
    N = len(ulist)
    rv = []
    for k in range(N):
        rv.append(-np.minimum(ulist[k], 0))
    return flist2fvec(rv)

# https://arxiv.org/pdf/1101.6081.pdf
# https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
def projBoundedSimplex(x, alpha=0., z=1., verbose=0):
    #Efficient Projections onto the l 1 -Ball for Learning in High Dimensions
    '''
    Project vector x on bounded simplex:
    \Delta_n(\alpha, z) = \{ x \in R^n | x_i >= \alpha, \sum_{i=1}^{n} x_i = z \}
    '''
    if alpha < float(z) / x.size:
        alpha = float(z) / x.size
        if verbose:
            print(f"impossible alpha. reset to {alpha:.2e}")
    y = np.sort(x)[::-1]
    n = y.size
    sumY = 0.
    t = None
    for i in range(n-1):
        sumY += y[i]
        ti = (sumY - z + (n-i-1)*alpha) / float(i+1)
        if ti >= y[i+1]:
            t = ti
            break
    if t is None:
        t = (sumY + y[n-1] - z) / n
    y = x - t
    y[y < alpha] = alpha
    return y

def projSimplex_old(x):
    y = np.sort(x)[::-1]
    n = y.size
    sumY = 0.
    t = None
    for i in range(n-1):
        sumY += y[i]
        ti = (sumY - 1.) / float(i+1)
        if ti >= y[i+1]:
            t = ti
            break
    if t is None:
        t = (sumY + y[n-1] - 1.) / n
    y = x - t
    y[y < 0] = 0
    return y
    
    
# TODO: assertion on dimnum[i] < P
def group_constraint(n, dimnum, otype='projected'):
    types = ['projected', 'Lagrange']
    assert otype in types, "select otype from %s" % (str(types).replace('[', '').replace(']', ''))
    rv = {}
    rv['lambda'] = 5e-1
    rv['dimnum'] = dimnum
    rv['type'] = otype
    if otype == 'projected':
        rv['hessian_matvec'] = None
        #rv['gradient'] = None
        rv['functional'] = None
        #rv['rule'] = None
        rv['projector'] = projector_group_constraint
    elif otype == 'Lagrange':
        rv['multSize'] = len(rv['dimnum']) #+ 2 + 2*n[-1]
        rv['hessian_matvec'] = hessian_mv_group_constraint
        #rv['gradient'] = None
        rv['functional'] = func_group_constraint
        #rv['rule'] = None
        rv['projector'] = projector_individual
        rv['multChecker'] = check_multipliers
    return rv    

def projector_group_constraint(
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None,
    dimnums=[0],
    vect=True,
    activeModes=None
):
    assert (lro_dict is not None) or ((tucker_dict is not None) and (lro_dict is not None)),\
        "group_constraint: lro and/or tucker parts must be specified"
    lFlag = False
    Ch = None
    Gh = None
    Ah = None
    cFlag = canonical_dict is not None
    if lro_dict is not None: # always True
        Bh = copy.deepcopy(lro_dict['B'])
        def nmap(x):
            if isinstance(x, list):
                return x[0].shape[0]
            return x.shape[0]
        n = list(map(nmap, Bh))
        d = len(n)
        P = lro_dict['P']
        L = lro_dict['L']
        M = sum(L)
        lFlag = True
        fmc = None
        if 'fullModesConfig' in lro_dict.keys():
            fmc = copy.deepcopy(lro_dict['fullModesConfig'])
    tFlag = False
    if tucker_dict is not None:
        Ah = copy.deepcopy(tucker_dict['A'])
        Gh = copy.deepcopy(tucker_dict['G'])
        r = tucker_dict['r']
        Rt = r.shape[0]
        tFlag = True
    gmode = d-1
    if activeModes is None:
        activeModes = {}
        if cFlag:
            activeModes['c'] = range(d)
        if lFlag:
            activeModes['l'] = range(d)
        if tFlag:
            activeModes['t'] = {}
            activeModes['t']['f'] = range(d)
            activeModes['t']['c'] = range(Rt)
    if lFlag and tFlag:
        ind = 0
        numI = len(L)
        for dimnum in dimnums:
            if dimnum in activeModes['l']:
                U, _ = np.linalg.qr(Ah[dimnum])
                if (fmc is not None) and (fmc[dimnum] is not None):
                    Bh[dimnum][0] -= np.dot(U, np.dot(U.T, Bh[dimnum][0]))
                    if Bh[dimnum][2] is not None: # no, here is also block matrix!
                        Bh[dimnum][2] -= np.dot(U, np.dot(U.T, Bh[dimnum][2]))
                else:
                    for k in range(numI):
                        offset = L[k]
                        Bh[dimnum][:, ind:ind+offset] -= np.dot(U, np.dot(U.T, Bh[dimnum][:, ind:ind+offset]))
                        ind += offset
        if gmode in activeModes['t']['f']:
            maxDeviationAbs = 1.5
            alpha = (numI - maxDeviationAbs) / float((numI-1)*numI)
            tmp = np.diag(Ah[gmode])
            Ah[gmode] = np.zeros([numI, r[0, -1]])
            np.fill_diagonal(Ah[gmode], numI*projBoundedSimplex(tmp, alpha, z=1.))
        if gmode in activeModes['l']:
            Bh[gmode] = np.eye(numI)
    else:
        ind = 0
        numI = len(L) - 1 #Bh[-1][:, -1].size
        for dimnum in dimnums:
            if dimnum in activeModes['l']:
                if (fmc is not None) and (fmc[dimnum] is not None):
                    U, _ = np.linalg.qr(Bh[dimnum][2])
                    Bh[dimnum][0] -= np.dot(U, np.dot(U.T, Bh[dimnum][0]))
                    if (Bh[dimnum][2].shape[1] > L[-1]):
                        Bh[dimnum][2][:, :-L[-1]] -= np.dot(U, np.dot(U.T, Bh[dimnum][2][:, :-L[-1]]))
                else:
                    U, _ = np.linalg.qr(Bh[dimnum][:, int(sum(L[:-1])):])
                    for k in range(numI):
                        offset = L[k]
                        Bh[dimnum][:, ind:ind+offset] -= np.dot(U, np.dot(U.T, Bh[dimnum][:, ind:ind+offset]))
                        ind += offset
        if gmode in activeModes['l']:
            maxDeviationAbs = 1.5
            alpha = (numI - maxDeviationAbs) / float((numI-1)*numI)
            Bh[gmode][:, -1] = numI*projBoundedSimplex(Bh[gmode][:, -1], alpha, z=1.)
            Bh[gmode][:, :-1] = np.eye(numI)
    cD, lD, tD = refactorizeDicts(canonical_dict, lro_dict, tucker_dict, newC=Ch, newB=Bh, newA=Ah, newG=Gh)
    if vect:
         return fcore2fvec(n, canonical_dict, lro_dict=lD, tucker_dict=tD)
    return canonical_dict, lD, tD

def projector_individual(canonical_dict=None, lro_dict=None, tucker_dict=None, dimnums=[0], vect=True):
    cFlag = canonical_dict is not None
    lFlag = lro_dict is not None
    tFlag = tucker_dict is not None
    activeModes = {}
    if cFlag:
        activeModes['c'] = []
    if lFlag:
        d = len(lro_dict['B'])
        activeModes['l'] = [d-1]
    if tFlag:
        d = len(tucker_dict['A'])
        activeModes['t'] = {}
        activeModes['t']['f'] = [d-1]
        activeModes['t']['c'] = [False]
    return projector_group_constraint(canonical_dict, lro_dict, tucker_dict, dimnums, vect, activeModes)

def check_multipliers(evalFV, multipliers, n, pmin):
    eps = np.spacing(1.)
    mask = np.abs(evalFV[-2*n[-1]:-n[-1]] - pmin) < eps 
    multipliers[-2*n[-1]:-n[-1]] *= mask
    multipliers[:-n[-1]] = np.maximum(0., multipliers[:-n[-1]])
    return multipliers

def hessian_mv_group_constraint(
    x,
    multipliers,
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None,
    dimnums=[0],
    vect=True
):
    assert (lro_dict is not None) or ((tucker_dict is not None) and (lro_dict is not None)),\
        "group_constraint: lro and/or tucker parts must be specified"
    lFlag = False
    Ch = None
    Gh = None
    Ah = None
    if lro_dict is not None: # always True
        B = lro_dict['B']
        def mapper(x):
            if isinstance(x, list):
                return x[0].shape[0]
            return x.shape[0]
        n = list(map(mapper, B))
        d = len(n)
        P = lro_dict['P']
        L = lro_dict['L']
        M = sum(L)
        lFlag = True
        Bh = []
        fmc = None
        if 'fullModesConfig' in lro_dict.keys():
            fmc = lro_dict['fullModesConfig']
    tFlag = False
    if tucker_dict is not None:
        A = tucker_dict['A']
        G = tucker_dict['G']
        r = tucker_dict['r']
        tFlag = True
        Ah = []
        Gh = [np.zeros(r[0, :])]
    cFlag = False
    if canonical_dict is not None:
        cFlag = True
        Rc = canonical_dict['Rc']
        if not vect:
            Ch = []
    for k in range(d):
        if cFlag and (not vect):
            Ch.append(np.zeros([n[k], Rc]))
        if k < P:
            if (fmc is not None) and (fmc[k] is not None):
                tmp = []
                tmp.append( np.zeros(B[k][0].shape) )
                tmp.append( np.zeros(B[k][1].shape) )
                if B[k][2] is not None:
                    tmp.append( np.zeros(B[k][2].shape) )
                else:
                    tmp.append(None)
                Bh.append(copy.deepcopy(tmp))
            else:
                Bh.append(np.zeros([n[k], M]))
        else:
            Bh.append(np.zeros([n[k], len(L)]))
        if tFlag:
            Ah.append(np.zeros([n[k], r[0, k]]))  
    C1, B1, A1, G1 = fvec2fcore(x[:-multipliers.size], n, canonical_dict, lro_dict, tucker_dict, full_result=True)
    if lFlag and (fmc is not None):
        lD1 = copy.deepcopy(lro_dict)
        _, lD1, _ = refactorizeDicts(lro_dict=lD1, newB=B1)
    dmultipliers = x[-multipliers.size:]
    dmultMu = dmultipliers[:len(dimnums)]
    #dmultLambda, dmultTheta = dmultipliers[len(dimnums):len(dimnums)+2]
    #dmultDzeta = dmultipliers[len(dimnums)+2:len(dimnums)+2+n[-1]]
    #dmultSdzeta = dmultipliers[len(dimnums)+2+n[-1]:]
    ###################################################################################
    #dmultMu *= 0.
    #dmultLambda *= 0.
    #dmultTheta *= 0.
    #dmultDzeta *= 0.
    
    rvMult = np.zeros(dmultipliers.size)
    if lFlag and tFlag:
        for i in range(len(dimnums)):
            dimnum = dimnums[i]
            if (dimnum < P) and (fmc is not None) and (fmc[dimnum] is not None):
                Bk = Bkl(lro_dict, dimnum, full=0)
                Bk1 = Bkl(lD1, dimnum, full=0)
                tmp = np.dot(A[dimnum], np.dot(A[dimnum].T, Bk[0]))
                Bh[dimnum][0] += dmultMu[i]*tmp
                rvMult[i] += np.sum(tmp*Bk1[0])
                if Bk[2] is not None:
                    tmp = np.dot(A[dimnum], np.dot(A[dimnum].T, Bk[2]))
                    Bh[dimnum][2] += dmultMu[i]*tmp
                    rvMult[i] += np.sum(tmp*Bk1[2])
                tmp = np.dot(Bk[0], np.dot(Bk[0].T, A[dimnum]))
                if Bk[2] is not None:
                    tmp += np.dot(Bk[2], np.dot(Bk[2].T, A[dimnum]))
                Ah[dimnum] += dmultMu[i]*tmp
                rvMult[i] += np.sum(tmp*A1[dimnum])
            else:
                tmp = np.dot(A[dimnum], np.dot(A[dimnum].T, Bkl(lro_dict, dimnum, full=1)))
                Bh[dimnum] += dmultMu[i]*tmp
                rvMult[i] += np.sum(tmp*B1[dimnum])
                tmp = Bkl(lro_dict, dimnum, full=1)
                tmp = np.dot(tmp, np.dot(tmp.T, A[dimnum]))
                Ah[dimnum] += dmultMu[i]*tmp
                rvMult[i] += np.sum(tmp*A1[dimnum])
        '''Ah[-1] += dmultLambda*A[-1] # see string after next to understand why we save diagonal here
        tmp = A[-1]*A1[-1] 
        np.fill_diagonal(tmp, 0.)
        rvMult[len(dimnums)] = np.sum(tmp)
        rvMult[len(dimnums)+1] = np.sum(np.diag(A1[-1]))
        rvMult[len(dimnums)+2:len(dimnums)+2+n[-1]] = -(1.- multipliers[-n[-1]:]**2.)*np.diag(A1[-1])
        rvMult[len(dimnums)+2+n[-1]:] = 2*multipliers[-n[-1]:]*multipliers[-2*n[-1]:-n[-1]]*np.diag(A1[-1])
        tmp = -(1.-multipliers[-n[-1]:]**2.)*dmultDzeta + dmultTheta + 2*multipliers[-n[-1]:]*multipliers[-2*n[-1]:-n[-1]]*dmultSdzeta
        np.fill_diagonal(Ah[-1], tmp)'''
    else:
        for i in range(len(dimnums)):
            
            dimnum = dimnums[i]
            if (dimnum < P) and (fmc is not None) and (fmc[dimnum] is not None):
                Bk = Bkl(lro_dict, dimnum, full=0)
                Bk1 = Bkl(lD1, dimnum, full=0)
                # fmc constraint is never applied to group core
                Bkg = Bk[2][:, -L[-1]:].copy()
                Bk1g = Bk1[2][:, -L[-1]:].copy()
                tmp = np.dot(Bkg, np.dot(Bkg.T, Bk[0]))
                Bh[dimnum][0] += dmultMu[i]*tmp
                rvMult[i] += np.sum(tmp*Bk1[0])
                if (Bk[2] is not None) and (Bk[2].shape[1] > L[-1]):
                    tmp = np.dot(Bkg, np.dot(Bkg.T, Bk[2][:, :-L[-1]]))
                    Bh[dimnum][2] += dmultMu[i]*tmp
                    rvMult[i] += np.sum(tmp*Bk1[2][:, :-L[-1]])
                #tmp = np.dot(B[dimnum][:, :-L[-1]], np.dot(B[dimnum][:, :-L[-1]].T, B[dimnum][:, -L[-1]:]))
                tmp = np.dot(Bk[0], np.dot(Bk[0].T, Bkg))
                Bh[2][:, -L[-1]:] += dmultMu[i]*tmp
                rvMult[i] += np.sum(tmp*Bk1g)
            else:
                tmp = np.dot(B[dimnum][:, -L[-1]:], np.dot(B[dimnum][:, -L[-1]:].T, B[dimnum][:, :-L[-1]]))
                Bh[dimnum][:, :-L[-1]] += dmultMu[i]*tmp
                rvMult[i] += np.sum(tmp*B1[dimnum][:, :-L[-1]])
                tmp = np.dot(B[dimnum][:, :-L[-1]], np.dot(B[dimnum][:, :-L[-1]].T, B[dimnum][:, -L[-1]:]))
                Bh[dimnum][:, -L[-1]:] += dmultMu[i]*tmp
                rvMult[i] += np.sum(tmp*B1[dimnum][:, -L[-1]:])
        '''tmp = B[-1][:, :-1].copy()
        np.fill_diagonal(tmp, 0.)
        Bh[-1][:, :-1] += dmultLambda*(tmp)
        tmp *= B1[-1][:, :-1]
        rvMult[len(dimnums)] = np.sum(tmp)
        rvMult[len(dimnums)+1] = np.sum(B1[-1][:, -1])
        rvMult[len(dimnums)+2:len(dimnums)+2+n[-1]] = -(1.- multipliers[-n[-1]:]**2.)*B1[-1][:, -1]
        rvMult[len(dimnums)+2+n[-1]:] = 2*multipliers[-n[-1]:]*multipliers[-2*n[-1]:-n[-1]]*B1[-1][:, -1]
        tmp = -(1.-multipliers[-n[-1]:]**2.)*dmultDzeta + dmultTheta + 2*multipliers[-n[-1]:]*multipliers[-2*n[-1]:-n[-1]]*dmultSdzeta
        Bh[-1][:, -1] += tmp'''
    #################################################
    #rvMult[:len(dimnums)] *= 0.
    #rvMult[len(dimnums)] *= 0.
    #rvMult[len(dimnums)+1] *= 0.
    #rvMult[len(dimnums)+2:] *= 0.
    #if cFlag and (not vect):
    #    cD, lD, tD = refactorizeDicts(canonical_dict=canonical_dict, lro_dict=lro_dict, tucker_dict=tucker_dict, newC=Ch, newB=Bh, newA=Ah, newG=Gh)
    #else:
    cD, lD, tD = refactorizeDicts(
        canonical_dict=canonical_dict,
        lro_dict=lro_dict,
        tucker_dict=tucker_dict,
        newC=Ch,
        newB=Bh,
        newA=Ah,
        newG=Gh
    )
    if vect:
        rv = fcore2fvec(n, canonical_dict=cD, lro_dict=lD, tucker_dict=tD)
        if cFlag:
            sizeC = int(Rc*np.sum(n))
            rv = np.concatenate([np.zeros(sizeC), rv, rvMult])
        else:
            rv = np.concatenate([rv, rvMult])
        return rv
    return cD, lD, tD, rvMult


def func_group_constraint(
    multipliers,
    pcum,
    pmin,
    canonical_dict=None,
    lro_dict=None,
    tucker_dict=None,
    dimnums=[0],
    vect=True
):
    assert (lro_dict is not None) or ((tucker_dict is not None) and (lro_dict is not None)),\
        "group_constraint: lro and/or tucker parts must be specified"
    lFlag = False
    if lro_dict is not None: # always True
        B = lro_dict['B']
        fmc = None
        if 'fullModesConfig' in lro_dict.keys():
            fmc = lro_dict['fullModesConfig']
        def mapper(x):
            if isinstance(x, list):
                return x[0].shape[0]
            return x.shape[0]
        n = list(map(mapper, B))
        d = len(n)
        P = lro_dict['P']
        L = lro_dict['L']
        M = sum(L)
        lFlag = True
    tFlag = False
    if tucker_dict is not None:
        A = tucker_dict['A']
        r = tucker_dict['r']
        tFlag = True
    multMu = multipliers[:len(dimnums)]
    #multLambda, multTheta = multipliers[len(dimnums):len(dimnums)+2]
    #multDzeta = multipliers[len(dimnums)+2:len(dimnums)+2+n[-1]]
    #multSdzeta = multipliers[len(dimnums)+2+n[-1]:]
    rv = []
    if lFlag and tFlag:
        for i in range(len(dimnums)):
            dimnum = dimnums[i]
            if (dimnum < P) and (fmc is not None) and (fmc[dimnum] is not None):
                Bk = Bkl(lro_dict, dimnum, full=0)
                temp = np.linalg.norm(np.dot(A[dimnum].T, Bk[0]))**2.
                if (Bk[2] is not None):
                    temp += np.linalg.norm(np.dot(A[dimnum].T, Bk[2]))**2.
                temp *= 0.5
            else:
                temp = 0.5*np.linalg.norm(np.dot(A[dimnum].T, B[dimnum]))**2.
            rv.append(temp)
        '''tmp = A[-1].copy()
        np.fill_diagonal(tmp, 0.)
        rv.append( 0.5*np.linalg.norm(tmp)**2. )
        rv.append( (np.sum(np.diag(A[-1])) - pcum) )
        temp = -(np.diag(A[-1]) - pmin)
        rv += temp.tolist()
        temp = multDzeta*multSdzeta
        rv += temp.tolist()'''
    else:
        for i in range(len(dimnums)):
            dimnum = dimnums[i]
            if (dimnum < P) and (fmc is not None) and (fmc[dimnum] is not None):
                Bk = Bkl(lro_dict, dimnum, full=0)
                Bkg = Bk[2][:, -L[-1]:]
                temp = np.linalg.norm(np.dot(Bkg.T, Bk[0]))**2.
                if (Bk[2].shape[1] > L[-1]):
                    temp += np.linalg.norm(np.dot(Bkg.T, Bk[2][:, :-L[-1]]))**2.
                temp *= 0.5*multMu[i]
            else:
                temp = 0.5*multMu[i]*np.linalg.norm(np.dot(B[dimnum][:, -L[-1]:].T, B[dimnum][:, :-L[-1]]))**2.
            rv.append(temp)
        '''tmp = B[-1][:, :-1].copy()
        np.fill_diagonal(tmp, 0.)
        rv.append( 0.5*np.linalg.norm(tmp)**2. )
        rv.append( (np.sum(B[-1][:, -1] - pcum)) )
        temp = -(B[-1][:, -1] - pmin)
        rv += temp.tolist()
        temp = multDzeta*multSdzeta
        rv += temp.tolist()'''
    return np.array(rv)
    
    
    

if __name__=='__main__':
    pass
    
    