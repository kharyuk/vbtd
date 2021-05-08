import numpy as np
import copy
import sys
sys.path.append('../')
sys.path.append('../matrix_models/')

from sklearn.base import BaseEstimator, TransformerMixin

import cobe
from td.utils import reshape, prodTenMat

class COBEContrast(BaseEstimator, TransformerMixin):
    
    def __init__(
        self,
        commonRank=1,
        shapeObject=None,
        maxitnum=100,
        epsilon=1e-5,
        random_state=None
    ):
        
        # assertions

        # shape of input except Nsubject axis
        assert shapeObject is not None, "Unspecified shape of feature space"
        # specific parameters check (depending on method)
        
        self.shapeObject = list(shapeObject)
        self.epsilon = epsilon
        self.commonRank = commonRank
        self.maxitnum = maxitnum
        self.random_state=random_state

    def saveParameters(self, path):
        np.savez_compressed(
            path,
            shapeObject=self.shapeObject,
            epsilon=self.epsilon,
            commonRank=self.commonRank,
            maxitnum=self.maxitnum,
            random_state=self.random_state,
            sources=self.sources
        )
        
    def loadParameters(self, path):
        df = np.load(path)
        #del self.sources
        self.shapeObject = df['shapeObject']
        self.epsilon = df['epsilon']
        self.commonRank = df['commonRank']
        self.maxitnum = df['maxitnum']
        self.random_state = df['random_state']
        self.sources = df['sources']
    
    
    def fit_transform(self, X, y=None, **fit_params):
        '''
        Learn common part AND get individual parts
        '''
        keys = fit_params.keys()
        if 'commonRank' in keys:
            commonRank = fit_params['commonRank']
        else:
            commonRank = copy.deepcopy(self.commonRank)
        if 'maxitnum' in keys:
            maxitnum = fit_params['maxitnum']
        else:
            maxitnum = copy.deepcopy(self.maxitnum)
        if 'epsilon' in keys:
            epsilon = fit_params['epsilon']
        else:
            epsilon = copy.deepcopy(self.epsilon)
        if 'random_state' in keys:
            random_state = fit_params['random_state']
        else:
            random_state = copy.deepcopy(self.random_state)
        individualComponents = self._fit_transform_fun(
            X, commonRank, maxitnum, epsilon, random_state
        )
        return individualComponents
    
    
    def _reshape(self, X):
        shape = [-1] + list(self.shapeObject)
        return reshape(X, shape)
        
    def _transform_fun(self, X):
    # understand whether sklearn allows to return tensorized output or not
        T = self._reshape(X)
        Scom = self.sources
        T1 = prodTenMat(T, Scom.T, 1)
        T = T - prodTenMat(T1, Scom, 1)
        return T
    
    
    def _fit_transform_fun(
        self,
        X,
        commonRank=None,
        maxitnum=None,
        epsilon=None,
        random_state=None
    ):        
        T = self._reshape(X)
        if epsilon is None:
            epsilon = self.epsilon
        if maxitnum is None:
            maxitnum = self.maxitnum
        if commonRank is None:
            commonRank = self.commonRank
        if random_state is None:
            random_state = self.random_state
        parameters = {}
        parameters['eps'] = epsilon
        parameters['maxitnum'] = maxitnum
        parameters['random_state'] = random_state
        parameters['inform'] = True
        if self.commonRank == 'auto':
            commonSources, inform = cobe.cobe(T, **parameters)
        else:
            commonSources, inform = cobe.cobec(T, commonRank, **parameters)
        self.sources = commonSources.copy()
        individualComponents = self._transform_fun(X)
        return individualComponents, inform
    
    
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
