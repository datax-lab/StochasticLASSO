import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from . import util, glmnet_model
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import math
from scipy import stats

class StochasticLasso:
    """
    Parameters
    ----------        
    q: 'auto' or int, optional [default='auto']
        The number of predictors to randomly selecting in the bootstrap sample.
        When to set 'auto', use q as number of samples.
    r: int [default=30]
       The number of times each predictors is selected in bootstrapping.            
    logistic: Boolean [default=False]
        Whether to apply logistic regression model. 
        For classification problem, Stochastic LASSO can apply the logistic regression model.       
    alpha: float [default=0.05]
       significance level used for significance test for feature selection
    random_state: int or None, optional [default=None]
        If int, random_state is the seed used by the random number generator; 
        If None, the random number generator is the RandomState instance used by np.random.default_rng
    parallel:  Boolean [default=False]
        Whether to apply parellel processing.
    n_jobs: 'None' or int, optional [default=1]
        The number of jobs to run in parallel.
        If "n_jobs is None" or "n_jobs == 0" could use the number of CPU cores returned by "multiprocessing.cpu_count()" for automatic parallelization across all available cores.
    """    
    def __init__(self, q='auto', r=30, logistic=False, alpha=0.05,
                 random_state=None, parallel=False, n_jobs=None):
        self.q = q
        self.r = r
        self.logistic = logistic
        self.alpha = alpha
        self.random_state = random_state
        self.parallel = parallel
        self.n_jobs = n_jobs        

    def fit(self, X, y, sample_weight=None):
        """
        Parameters
        ----------
        X: array-like of shape (n_samples, n_predictors)
           predictor variables    
        y: array-like of shape (n_samples,)
           response variables            
        sample_weight : array-like of shape (n_samples,), default=None
            Optional weight vector for observations. If None, then samples are equally weighted.

        Attributes
        ----------                
        p_values_ : array
            P-values of each coefficients.
        coef_ : array
            Coefficients of Stochastic LASSO.            
        intercept_: float
            Intercept of Stochastic LASSO.
        """        
        self.X = np.array(X)
        self.y = np.array(y).ravel()
        self.n, self.p = X.shape
        self.q = self.n if self.q == 'auto' else self.q
        self.sample_weight = np.ones(
            self.n) if sample_weight is None else np.asarray(sample_weight)
        self.corr = np.corrcoef(self.X.swapaxes(1,0))**2
        self.corr[(np.isnan(self.corr)) | (self.corr==0)] = 10**(-10)

        betas = self._bootstrapping()
        self.betas = betas
        self.p_values_ = self._compute_p_values(betas)
        self.coef_ = np.where(self.p_values_ < self.alpha, betas.mean(axis = 0), 0)
        self.intercept_ = np.average(self.y) - np.average(self.X, axis=0) @ self.coef_        
        return self

    def _bootstrapping(self):
        """
        Execute bootstrapping according to 'parallel' parameter.
        """        
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
        bst_predictor_idx_list = self._predictor_sampling()
        
        for bst_predictor_idx in bst_predictor_idx_list:
            # Standardization.
            X_sc, y_sc, x_std = util.standardization(self.X[bst_sample_idx, :][:, bst_predictor_idx],
                                                     self.y[bst_sample_idx])
            # Estimate coef.
            coef = glmnet_model.ElasticNet(X_sc, y_sc, logistic=self.logistic,
                                           sample_weight=self.sample_weight[bst_sample_idx], random_state=self.rs)

            beta[bst_predictor_idx] = beta[bst_predictor_idx] + (coef / x_std)
        return beta

    def _predictor_sampling(self):
        """
        Draw predictors for the bootstrap samples by Correlation Based Bootstrapping(CBB) algorithm.
        CBB penalizes predictors highly correlated with others in the bootstrapping, so that the predictors of bootstrap samples become independent.
        
        S: Set of indices of predictors that have not been drawn
        Q: Set of indices of predictors that already included in the bootstrap sample
        """                
        S = list(range(self.p))
        bst_predictor_idx_list = []
        
        # Draw q numbers of predictors.
        while len(S)>=self.q:
            Q = []
            # Draw the first predictor.
            # Update Q and S.
            Q.append(np.random.choice(S, 1)[0])
            S.remove(Q[-1])
            for j in range(self.q-1):
                # Draw q-1 numbers of predictors based on Pearson correlation coefficient between Q and S.
                sel_prop = (1/(self.corr[Q,:][:,S]).sum(axis = 0))
                # Update Q and S.
                Q.append(np.random.choice(S, 1, p = sel_prop/sel_prop.sum())[0])
                S.remove(Q[-1])
            bst_predictor_idx_list.append(Q)
        if len(S)!=0:
            bst_predictor_idx_list.append(S)
        return bst_predictor_idx_list 
    
    def _compute_p_values(self, betas):
        """
        Compute p-values of each predictor by two-stage t-test.
        """        
        # One sample t-test
        relevant = (stats.ttest_1samp(betas, 0)[1] < self.alpha)
        # Two sample t-test
        pop = abs(betas[:,relevant]).reshape(-1)
        p_values = np.array([stats.ttest_ind(abs(betas[:,i]), pop, alternative = 'greater')[1] 
                             if b else 1 for i, b in enumerate(relevant)])
        return p_values