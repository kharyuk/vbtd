import numpy as np
import copy
import time
import sys
sys.path.append('../')

from sklearn.base import BaseEstimator, TransformerMixin

from td.utils import reshape, prodTenMat

import group_tcd
#import gtcd_jit as gtcd
import gtcd

# global variables
_maxInnerIt = 15
_tolRes = 1e-8
_tolGrad = 1e-8
_tolSwamp = 1e-8

class GLROContrast(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        individualRank=1,
        commonRank=1,
        shapeObject=None,
        sourceModes=None,
        method='als',
        nShortModes=None,
        constraintMethod='projected',
        fullModesConstraint=None,
        maxitnum=100,
        epsilon=1e-5,
        random_state=None
    ):
        # assertions
        # shape of input except Nsubject axis
        assert shapeObject is not None, "Unspecified shape of feature space"
        # specific parameters check (depending on method)

        self.shapeObject = list(shapeObject)
        self.individualRank = individualRank
        self.commonRank = commonRank
        self.sources = None
        self.maxitnum = maxitnum
        self.epsilon = epsilon
        self.random_state = random_state

        self.sourceModes = list(sourceModes)
        if len(self.sourceModes) > 1:
            raise NotImplementedError('Multi source modes are not supported at the moment')
        self.fullModesConstraint = fullModesConstraint
        self.nShortModes = 1
        if nShortModes is not None:
            assert isinstance(nShortModes, int) and (0 < nShortModes < len(self.shapeObject))-1
            self.nShortModes += nShortModes
        # method
        current_method = method.lower()
        if current_method == 'als':
            self.method = alsM
        elif current_method == 'gd':
            self.method = gdM
        elif current_method == 'cg-fr':
            self.method = cgfrM
        elif current_method == 'cg-pr':
            self.method = cgprM
        elif current_method == 'cg-hs':
            self.method = cghsM
        elif current_method == 'cg-dy':
            self.method = cgdyM
        elif current_method == 'gn':
            self.method = gnM
        elif current_method == 'lm-n':
            self.method = lmnM
        elif current_method == 'lm-q':
            self.method = lmqM
        elif current_method == 'dogleg':
            self.method = doglegM
        elif current_method == 'scg-qn':
            self.method = scg_qnM
        elif current_method == 'scg-fn':
            self.method = scg_fnM
        else:
            raise NotImplementedError
            
        # constraints
        current_constraintMethod = constraintMethod.lower()
        assert current_constraintMethod in ('projected', 'lm')
        self.groupConstraintMethod = current_constraintMethod
        

    def saveParameters(self, filename):
        np.savez_compressed(
            filename, sources=self.sources, sourcesType=self.sourcesType,
            timesEstimate=self.timesEstimate
            #mixing=self.mixing
        )
        
    def loadParameters(self, filename):
        df = np.load(filename)
        del self.sources, self.classes, self.sourcesType#, self.mixing
        self.sources = df['sources']
        self.sourcesType = df['sourcesType']
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
        verbose=False,
        recover=True,
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
            X,
            individualRank,
            commonRank,
            maxitnum,
            epsilon,
            verbose,
            recover,
            random_state
        )
        return individualComponents
    
    
    def _reshape(self, X):
        shape = [-1] + self.shapeObject
        Y = reshape(X, shape)
        transposition = list(range(1, len(self.shapeObject)+1)) + [0]
        Y = np.transpose(Y, transposition)
        return Y
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
        verbose=False,
        recover=True,
        random_state=None
    ):
        if epsilon is None:
            epsilon = self.epsilon
        if random_state is None:
            random_state = self.random_state
        if maxitnum is None:
            maxitnum = self.maxitnum
        if commonRank is None:
            commonRank = self.commonRank
        if individualRank is None:
            individualRank = self.individualRank
        #self.sources = {self.sourceModes[i]: [] for i in xrange(len(self.sourceModes))}
        self.timesEstimate = None
        T = self._reshape(X)
        T = T.astype(np.float32)
        n = T.shape
        groupConstraint = group_tcd.group_constraint(
            n, self.sourceModes, self.groupConstraintMethod
        )
        normT = np.linalg.norm(T)
        T /= normT
        cdN = None
        P = len(T.shape)-self.nShortModes
        ldN = {
            'L': [self.individualRank]*n[-1] + [self.commonRank],
            'P': P,
            'fullModesConstraint': self.fullModesConstraint
        }
        tdN = None
        x0 = None
        np.random.seed(random_state)
        tic = time.clock()
        cdN, ldN, tdN, info = self.method(
            T, x0, cdN, ldN, tdN, maxitnum, groupConstraint, verbose
        )
        toc = time.clock()
        self.timesEstimate = toc-tic
        indCommon = self.commonRank
        fmc = self.fullModesConstraint
        self.sources = [None]*len(n)
        for k in range(len(n)):
            if k < P:
                if (fmc is not None) and (fmc[k] is not None):
                    self.sources[k] = ldN['B'][k][2][:, -indCommon:].copy()
                    if ldN['B'][k][2].shape[1] == indCommon:
                        ldN['B'][k][2] = None
                    else:
                        ldN['B'][k][2] = ldN['B'][k][2][:, :-indCommon].copy()
                else:
                    self.sources[k] = ldN['B'][k][:, -indCommon:].copy()
                    ldN['B'][k] = ldN['B'][k][:, :-indCommon].copy()
            else:
                self.sources[k] = ldN['B'][k][:, -1:].copy()
                ldN['B'][k] = ldN['B'][k][:, :-1].copy()
        ldN['L'] = ldN['L'][:-1]
        if recover:
            del ldN['E']
            T = gtcd.recover(n, lro_dict=ldN)
            permutation = np.roll(np.arange(T.ndim), 1)
            return np.transpose(T, permutation)
        return ldN
    
class GTLDContrast(BaseEstimator, TransformerMixin):
    '''
    Contrasting via group-independent component decompositions
    
    Input parameters:
    ----------------------
        
    Method to extract group and individual parts.
        
        n: {array-like , 1-dimensional, integer values}, default=None
    
    Shape of  input data **except** the last value associated with group axis.
    
        moi: {array-like, 1-dimensional, integer valyes}, default=[0]
        
    List, tuple or numpy array containing modes of interest to which
    separation criteria is to be applied.
    '''

    def __init__(
        self,
        individualRank=1,
        commonRank=1,
        shapeObject=None,
        sourceModes=None,
        method='als',
        nShortModes=None,
        constraintMethod='projected',
        fullModesConstraint=None,
        modeSizeFirstPriority=True,
        maxitnum=100,
        epsilon=1e-5,
        random_state=None
    ):
        # assertions
        # shape of input except Nsubject axis
        assert shapeObject is not None, "Unspecified shape of feature space"
        # specific parameters check (depending on method)
        self.modeSizeFirstPriority = modeSizeFirstPriority

        self.shapeObject = list(shapeObject)
        self.individualRank = individualRank
        self.commonRank = commonRank
        self.sources = None
        self.maxitnum = maxitnum
        self.epsilon = epsilon
        self.random_state = random_state

        self.sourceModes = list(sourceModes)
        if len(self.sourceModes) > 1:
            raise NotImplementedError('Multi source modes are not supported at the moment')
        self.fullModesConstraint = fullModesConstraint
        self.nShortModes = 1
        if nShortModes is not None:
            assert isinstance(nShortModes, int) and (0 < nShortModes < len(self.shapeObject))-1
            self.nShortModes += nShortModes
        # method
        current_method = method.lower()
        if current_method == 'als':
            self.method = alsM
        elif current_method == 'gd':
            self.method = gdM
        elif current_method == 'cg-fr':
            self.method = cgfrM
        elif current_method == 'cg-pr':
            self.method = cgprM
        elif current_method == 'cg-hs':
            self.method = cghsM
        elif current_method == 'cg-dy':
            self.method = cgdyM
        elif current_method == 'gn':
            self.method = gnM
        elif current_method == 'lm-n':
            self.method = lmnM
        elif current_method == 'lm-q':
            self.method = lmqM
        elif current_method == 'dogleg':
            self.method = doglegM
        elif current_method == 'scg-qn':
            self.method = scg_qnM
        elif current_method == 'scg-fn':
            self.method = scg_fnM
        else:
            raise NotImplementedError
            
        # constraints
        current_constraintMethod = constraintMethod.lower()
        assert current_constraintMethod in ('projected', 'lm')
        self.groupConstraintMethod = current_constraintMethod
        

    def saveParameters(self, filename):
        np.savez_compressed(
            filename, sources=self.sources, sourcesType=self.sourcesType,
            timesEstimate=self.timesEstimate
            #mixing=self.mixing
        )
        
    def loadParameters(self, filename):
        df = np.load(filename)
        del self.sources, self.classes, self.sourcesType#, self.mixing
        self.sources = df['sources']
        self.sourcesType = df['sourcesType']
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
        verbose=False,
        recover=True,
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
            X, individualRank, commonRank, maxitnum, epsilon, verbose, recover, random_state
        )
        return individualComponents
    
    
    def _reshape(self, X):
        shape = [-1] + self.shapeObject
        Y = reshape(X, shape)
        transposition = list(range(1, len(self.shapeObject)+1)) + [0]
        Y = np.transpose(Y, transposition)
        return Y
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
        verbose=False,
        recover=True,
        random_state=None
    ):
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
        #self.sources = {self.sourceModes[i]: [] for i in xrange(len(self.sourceModes))}
        self.timesEstimate = None
        T = self._reshape(X)
        T = T.astype('d')
        n = T.shape
        groupConstraint = group_tcd.group_constraint(
            n, self.sourceModes, self.groupConstraintMethod
        )
        normT = np.linalg.norm(T)
        T /= normT
        cdN = None
        P = len(T.shape)-self.nShortModes
        ldN = {
            'L': [self.individualRank]*n[-1],
            'P': P,
            'fullModesConstraint': self.fullModesConstraint
        }
        r = np.zeros([1, len(T.shape)])
        r[0, :-1] = self.commonRank
        if self.modeSizeFirstPriority:
            r[0, :-1] = np.minimum(r[0, :-1], n[:-1]) 
        # last mode - group axis
        r[:, -1] = n[-1]
        tdN = {
            'r': r.astype('i')
        }
        x0 = None
        np.random.seed(random_state)
        tic = time.clock()
        cdN, ldN, tdN, info = self.method(
            T, x0, cdN, ldN, tdN, maxitnum, groupConstraint, verbose
        )
        toc = time.clock()
        self.timesEstimate = toc-tic
        indCommon = self.commonRank
        fmc = self.fullModesConstraint
        self.sources = copy.deepcopy(tdN)
        if recover:
            del ldN['E']
            T = gtcd.recover(n, lro_dict=ldN)
            permutation = np.roll(np.arange(T.ndim), 1)
            return np.transpose(T, permutation)
        return ldN
        
# set up different algorithms
def alsM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='als', verbose=verbose, 
        regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )
def gdM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='gd', backtrack=True, 
        verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )
def gdrtM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='gd', backtrack=True,
        verbose=verbose, regTGD=1e-3, regPGD=None, doSA=0, constraints=constraints
    )
def gdrpM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='gd', backtrack=True, 
        verbose=verbose, regTGD=None, regPGD=1e-3, doSA=0, constraints=constraints
    )
def cgfrM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum,
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='cg', betaCG='fr', 
        verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )
def cgprM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='cg', betaCG='pr', 
        verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )
def cghsM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum,
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='cg', betaCG='hs',
        verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )

def cgdyM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='cg', betaCG='dy', 
        verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )
def gnM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes,tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='gn', backtrack=True, 
        verbose=verbose, regTGD=None, regPGD=None, doSA=0, constraints=constraints
    )
def lmqM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='lm', epsilonLM=1e-8,
        lmSetup='Quadratic', muInit=1., verbose=verbose, regTGD=None, regPGD=None,
        doSA=0, constraints=constraints
    )
def lmnM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='lm', epsilonLM=1e-8,
        lmSetup='Nielsen', muInit=1., verbose=verbose, regTGD=None, regPGD=None,
        doSA=0, constraints=constraints
    )
def doglegM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        maxInnerIt=_maxInnerIt, tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='tr', 
        verbose=verbose, doSA=0, constraints=constraints, trStep='dogleg',
        trDelta0=1.2,trEta=0.23
    )
def scg_qnM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        maxInnerIt=_maxInnerIt, tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='tr', 
        verbose=verbose, doSA=0, constraints=constraints, curvature=0, trStep='scg',
        trDelta0=1.2, trEta=0.23
    )
def scg_fnM(a, x0, cdN, ldN, tdN, maxitnum, constraints, verbose):
    return gtcd.tcd(
        a, x0=x0, canonical_dict=cdN, lro_dict=ldN, tucker_dict=tdN, maxitnum=maxitnum, 
        maxInnerIt=_maxInnerIt, tolRes=_tolRes, tolGrad=_tolGrad, tolSwamp=_tolSwamp, method='tr', 
        verbose=verbose, doSA=0, constraints=constraints, curvature=1, trStep='scg',
        trDelta0=1.2, trEta=0.23
    )
