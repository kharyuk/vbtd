import numpy as np
import copy
import sys
sys.path.append('../')
sys.path.append('../matrix_models/')

from sklearn.base import BaseEstimator, TransformerMixin

import gica
from td.utils import reshape, prodTenMat

class GICAContrast(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        individualRank=1,
        commonRank=1,
        shapeObject=None,
        maxitnum=100,
        epsilon=1e-5,
        feature_extractor='fast_ica',
        dtype=np.float32,
        random_state=None
    ):
        
        # assertions
        # shape of input except Nsubject axis
        assert shapeObject is not None, "Unspecified shape of feature space"
        # specific parameters check (depending on method)
        self.feature_extractor = feature_extractor
        self.shapeObject = list(shapeObject)
        self.epsilon = epsilon
        self.commonRank = commonRank
        self.individualRank = individualRank
        self.maxitnum = maxitnum
        self.groupDict = None
        self.dtype = dtype
        self.random_state = random_state

    def saveParameters(self, filename):
        np.savez_compressed(
            filename, sources=self.sources, sourcesType=self.sourcesType,
            classes=self.classes, timesEstimate=self.timesEstimate
            #mixing=self.mixing
        )
        
    def loadParameters(self, filename):
        df = np.load(filename)
        del self.sources, self.classes, self.sourcesType#, self.mixing
        self.sources = df['sources']
        self.sourcesType = df['sourcesType']
        self.classes = df['classes']
        self.timesEstimate = df['timesEstimate']
        #self.mixing = df['mixing']
    
    
    def fit_transform(
        self,
        X,
        y=None,
        individualRank=None,
        commonRank=None,
        maxitnum=None,
        epsilon=None,
        random_state=None
    ):
        '''
        Learn common part AND get individual parts
        '''
        if individualRank is None:           
            individualRank = copy.deepcopy(self.individualRank)
        if commonRank is None:
            commonRank = copy.deepcopy(self.commonRank)
        if maxitnum is None:
            maxitnum = copy.deepcopy(self.maxitnum)
        if epsilon is None:
            epsilon = copy.deepcopy(self.epsilon)
        if random_state is None:
            random_state = copy.deepcopy(self.random_state)
        individualComponents = self._fit_transform_fun(
            X.astype(self.dtype),
            individualRank,
            commonRank,
            maxitnum,
            epsilon,
            random_state
        )
        return individualComponents
    
    
    def _reshape(self, X):
        shape = [-1] + self.shapeObject
        return reshape(X, shape)
    '''
    def _transform_fun(self, X):
    # understand whether sklearn allows to return tensorized output or not
        T = self._reshape(X)
        Scom = self.sources
        T1 = prodTenMat(T, Scom.T, 1)
        T = T - prodTenMat(T1, Scom, 1)
        return T
    '''
    
    def _fit_transform_fun(
        self,
        X,
        individualRank=None,
        commonRank=None,
        maxitnum=None,
        epsilon=None,
        random_state=None
    ):        
        T = self._reshape(X)
        Nsubj = T.shape[0]
        if epsilon is None:
            epsilon = self.epsilon
        if maxitnum is None:
            maxitnum = self.maxitnum
        if commonRank is None:
            commonRank = self.commonRank
        if individualRank is None:
            individualRank = self.individualRank
        if random_state is None:
            random_state = self.random_state
        parameters = {}
        #parameters['eps'] = epsilon
        parameters['maxitnum'] = maxitnum
        parameters['R_com'] = commonRank
        parameters['rPCA_com'] = commonRank
        parameters['rPCA_ind'] = individualRank
        parameters['random_state'] = random_state
        individualRank_isint = isinstance(individualRank, (int, np.integer))
        if individualRank_isint:
            parameters['rPCA_ind'] = [individualRank]*Nsubj
        else:
            assert len(parameters['rPCA_ind']) == Nsubj, "Inividual ranks must be specified for each subject"
        parameters['feature_extractor'] = self.feature_extractor
        parameters['random_state'] = None
        self.groupDict = gica.gica(T, **parameters)

        G = np.dot(self.groupDict['ica']['sources'], self.groupDict['ica']['mixing'].T)
        G = np.dot(G, self.groupDict['gpca']['transform'].T) + self.groupDict['gpca']['mean']
        
        ind = 0
        #indiv_cs = []
        '''
        if self.feature_extractor == 'fast_ica':
            ica = FastICA(
                random_state=_RANDOM_STATE,
                max_iter=self.maxitnum # should it be separated from other maxitnums?
            )
        else:
            raise NotImplementedError
        '''
        for p in range(Nsubj):
            offset = parameters['rPCA_ind'][p]
            T[p, :, :] -= np.dot(G[:, ind:ind+offset], self.groupDict['ipca']['transform'][p].T)
            T[p, :, :] -= self.groupDict['ipca']['mean'][p]
            ind += offset
            '''
            if self.feature_extractor == 'fast_ica':
                ica.n_components = indivR[p]
                S = ica.fit_transform(T[p])
                indiv_cs.append(S.copy())
            else:
                raise NotImplementedError
            '''
        return T
    
    
    '''
    def fit(self, X, y=None, individualRank=None, commonRank=None, special_parameters=None):
        '#''
        Learn the common part of data
        Nsubj is axis0!
        '#''
        if self.method is None:
            return
        if indivR is None:
            indivR = self.indivR
        if commonR is None:
            commonR = self.commonR
        shapeX = X.shape
        Nsubjects, Nother = shapeX[0], shapeX[1:]
        assert np.prod(self.n) == np.prod(Nother), "Incorrect number of features for input data X"
        n = [Nsubjects] + self.n
        self._fit_fun(X, n, indivR, commonR, special_parameters)
        '''
    
    '''
    def _fit_fun(self, X, n, indivR=None, commonR=None, special_parameters=None):
        # TODO: check commonR (vect/int)
        if indivR is None:
            indivR = self.indivR
        if commonR is None:
            commonR = self.commonR
        else: 
            self.commonR = copy.deepcopy(commonR)
        if special_parameters is None:
            special_parameters = self.special_parameters
        else:
            self.special_parameters = copy.deepcopy(special_parameters)
        T, sigma, nT = self._tens(X, n)
        parameters = {}
            parameters['eps'] = 1e-8
            parameters['maxitnum'] = self.maxitnum
            parameters['verbose'] = False
            parameters['inform'] = False
            parameters['fast'] = True
            if self.commonR == 'auto':
                Scom = cobe(T, **parameters)
            else:
                Scom = cobec(T, commonR, **parameters)
            self.groupDict = {}
            self.groupDict['Scom'] = Scom.copy()
        return
        '''
    '''
    def transform(self, X):
        '#''
        Get individual parts **knowing** the common.
        '#''
        indiv_cs = self._transform_fun(X, n, indivR, rPCA_ind=None)
        return indiv_cs
    def fit_predict(self, X, y=None):
        '#''
        Learn the common part of data AND cluster the individual
        '#''
        _predict_fun(self, X, n, indivR=None, special_parameters=None)
        pass
    ''' 
