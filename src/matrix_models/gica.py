import numpy as np
import copy

from sklearn.decomposition import FastICA

def demean(X):
    shapeX = X.shape
    meanX = X.mean(axis=0, keepdims=True)
    Xc = X - meanX
    return Xc, meanX
    
def gica(
    Tensor,
    rPCA_ind,
    rPCA_com,
    R_com,
    feature_extractor='fast_ica',
    maxitnum=100,
    random_state=None
):
    '''
    Tensor with (Nobj, Nfeat, Nchan) shape
    Individual PCA and Group PCA steps are to be applied to the 2nd axis.
    '''
    shapeT = Tensor.shape
    [Nobj, Nfeat, Nchan] = shapeT
    indR_isint = isinstance(rPCA_ind, int)
    if indR_isint: 
        rPCA_ind = [rPCA_ind]*Nobj
    indWp = []
    meanTp = []
    indTransform = []
    for p in range(Nobj):
        Tp = Tensor[p, :, :].copy()
        demT, mean_p = demean(Tp)
        meanTp.append(mean_p)
        U, S, Vt = np.linalg.svd(np.dot(demT.T, demT))
        Vt = Vt[:rPCA_ind[p], :]
        tmp = np.dot(demT, Vt.T)
        indWp.append(tmp.copy())
        indTransform.append(Vt.T)
    stackedW = np.hstack(indWp)
    stackedW, meanStackedW = demean(stackedW)
    _, _, Vt = np.linalg.svd(np.dot(stackedW.T, stackedW))  ### not always optimal
    Vt = Vt[:rPCA_com, :]
    G = np.dot(stackedW, Vt.T)
    if feature_extractor == 'fast_ica':
        ica = FastICA(
            n_components=R_com, random_state=random_state, max_iter=maxitnum#, algorithm='deflation')
        )
        S = ica.fit_transform(G)
        A = ica.mixing_
        W = np.linalg.pinv(A)
    else:
        raise NotImplementedError
    result = {}
    result['ica'] = {'mixing': A, 'unmixing': W, 'sources': S, 'G': G}
    result['gpca'] = {'transform': Vt.T, 'mean': meanStackedW}
    result['ipca'] = {'transform': indTransform, 'mean': meanTp}
    if indR_isint:
        rPCA_ind = rPCA_ind[0]
    return result

