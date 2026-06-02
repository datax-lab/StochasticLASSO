import numpy as np


def standardization(X, y, logistic = False):
    """The response is mean-corrected and the predictors are standardized

    Parameters
    ---------
    X: array-like of shape (n_samples, n_predictors)
       predictor              
    y: array-like of shape (n_samples,)
       response

    Returns
    -------
    np.ndarray
        scaled_X, scaled_y, sd_X
    """
    mean_x = X - X.mean()
    X_sc = mean_x / np.sqrt((mean_x ** 2).sum(axis=0))
    if logistic == False:
        y_sc = y - y.mean()
    else:
        y_sc = y
    return X_sc, y_sc, np.sqrt((mean_x ** 2).sum(axis=0))
