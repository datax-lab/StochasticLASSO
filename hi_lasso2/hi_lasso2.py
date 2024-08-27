import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from . import util, glmnet_model
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import math

class HiLasso2:
    def __init__(self, q='auto', r=30, logistic=False,
                 random_state=None, parallel=False, n_jobs=None):
        self.q = q
        self.r = r
        self.logistic = logistic
        self.random_state = random_state
        self.parallel = parallel
        self.n_jobs = n_jobs        

    def fit(self, X, y, sample_weight=None):
        self.X = np.array(X)
        self.y = np.array(y).ravel()
        self.n, self.p = X.shape
        self.q = self.n if self.q == 'auto' else self.q
        self.sample_weight = np.ones(
            self.n) if sample_weight is None else np.asarray(sample_weight)
        self.corr = np.corrcoef(self.X.swapaxes(1,0))**2
        self.corr[np.isnan(self.corr)] = 10**(-10)
        self.corr[self.corr==0] = 10**(-10)

        b1 = self._bootstrapping()
        self.coef_ = b1
        return self

    def _bootstrapping(self):
        if self.parallel:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                results = tqdm(executor.map(self._estimate_coef,
                                            np.arange(self.r)), total=self.r)
                betas = np.array(list(results))
        else: 
            betas = np.array([list(self._estimate_coef(bootstrap_number))
                              for bootstrap_number in tqdm(np.arange(self.r))])
        return betas

    def _estimate_coef(self, bootstrap_number):
        """
        Estimate coefficients for each bootstrap samples.
        """
        beta = np.zeros((self.p))

        # Set random seed as each bootstrap_number.
        self.rs = np.random.RandomState(
            bootstrap_number + self.random_state) if self.random_state else np.random.default_rng()
        
        # Generate bootstrap index of sample.
        bst_sample_idx = self.rs.choice(np.arange(self.n), size=self.n, replace=True, p=None)
        bst_predictor_idx_list = self._variable_sampling()
        
        for bst_predictor_idx in bst_predictor_idx_list:
            # Standardization.
            X_sc, y_sc, x_std = util.standardization(self.X[bst_sample_idx, :][:, bst_predictor_idx],
                                                     self.y[bst_sample_idx])
            # Estimate coef.
            coef = glmnet_model.ElasticNet(X_sc, y_sc, logistic=self.logistic,
                                           sample_weight=self.sample_weight[bst_sample_idx], random_state=self.rs)

            beta[bst_predictor_idx] = beta[bst_predictor_idx] + (coef / x_std)
        return beta

    def _variable_sampling(self):
        idx_set = np.arange(self.p)
        bst_predictor_idx_list = []
        for i in range(math.ceil(self.p/self.q)-1):
            bst_predictor_idx = []
            sel_prop = (1/self.corr[idx_set,:][:,idx_set].sum(axis = 1))
            bst_predictor_idx.append(self.rs.choice(idx_set, 1, p = sel_prop/sel_prop.sum())[0])
            idx_set = np.setdiff1d(idx_set, bst_predictor_idx[-1])
            for j in range(self.q-1):
                sel_prop = (1/(self.corr[bst_predictor_idx,:][:,idx_set]).sum(axis = 0))
                bst_predictor_idx.append(self.rs.choice(idx_set, 1, p = sel_prop/sel_prop.sum())[0])
                idx_set = np.setdiff1d(idx_set, bst_predictor_idx[-1])
            bst_predictor_idx_list.append(np.array(bst_predictor_idx))
        bst_predictor_idx_list.append(idx_set)
        return bst_predictor_idx_list