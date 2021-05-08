import numpy as np
from .utils import krp_cw

'''
def Ckl(C, k, l, L, P);
def batch_krp_bcw_fm(fmC, L, l, reverse=False);
def batch_krp_bcw_rm(rmC, l, reverse=False);
def batch_krp_bcw(lro_dict, l, skipped=None, reverse=False);
def hadamardCprTCql(lro_dict, r, l, skipped=None);
'''


def Ckl(C, k, l, L, P):
    if k < P:
        ind = int(round(sum(L[:l])))
        return C[k][:, ind:ind+L[l]]
    return C[k][:, l:l+1]

def batch_krp_bcw_fm(fmC, L, l, reverse=False):
    d = len(fmC)
    result = np.ones([1, L[l]])
    for k in range(d):
        result = krp_cw(result, Ckl(fmC, k, l, L, d), reverse)
    return result

def batch_krp_bcw_rm(rmC, l, reverse=False):
    d = len(rmC)
    result = np.ones([1, 1])
    for k in range(d):
        result = krp_cw(result, rmC[k][:, l:l+1], reverse)
    return result

def batch_krp_bcw(lro_dict, l, skipped=None, reverse=False):
    C, L, P = lro_dict['C'], lro_dict['L'], lro_dict['P']
    Rl = len(L)
    if skipped is not None:
        if skipped < P:
            krpC = batch_krp_bcw_rm(C[P:], l, reverse)
        else:
            lenCrm = len(C[P:skipped]+C[skipped+1:])
            if lenCrm > 0:
                krpC = batch_krp_bcw_rm(C[P:skipped]+C[skipped+1:], l, reverse)
            else:
                krpC = np.ones([1, 1])
    else:
        krpC = batch_krp_bcw_rm(C[P:], l, reverse)
    krpC = np.tile(krpC, [1, L[l]])
    if skipped is not None:
        if skipped < P:
            if P > 1:
                krpC = krp_cw(
                    batch_krp_bcw_fm(C[:skipped]+C[skipped+1:P], L, l, reverse),
                    krpC,
                    reverse
                )
        else:
            krpC = krp_cw(
                batch_krp_bcw_fm(C[:P], L, l, reverse),
                krpC,
                reverse
            )
    else:
        krpC = krp_cw(
            batch_krp_bcw_fm(C[:P], L, l, reverse),
            krpC,
            reverse
        )
    return krpC

def hadamardCprTCql(lro_dict, r, l, skipped=None):
    '''
    Skipped can be a list of integers if we consider 2nd order methods
    '''
    L, P, C, D = lro_dict['L'], lro_dict['P'], lro_dict['C'], lro_dict['D']
    Rl, sL = len(L), int(round(sum(L)))
    d = len(C)
    result = np.ones([L[r], L[l]])
    for k in range(d):
        if (skipped is not None) and (k == skipped):
            continue
        result *= Ckl(C, k, r, L, P).T@Ckl(C, k, l, L, P)
    return result