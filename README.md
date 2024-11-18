# Stochastic LASSO
Stochastic LASSO can significantly enhance the existing bootstrap-based LASSO models, providing better performance in both feature selection and coefficient estimation on extremely high-dimensional data.
Stochastic LASSO systematically addresses the drawbacks of bootstrapping by reducing multicollinearity in bootstrap samples, mitigating randomness in predictor sampling, and providing a statistical strategy for selecting statistically significant features.
Stochastic LASSO can be applied to any high-dimensional linear and logistic regression modeling.

## Installation
**Stochastic LASSO** support Python 3.6+. ``Stochastic LASSO`` can easily be installed with a pip install::

```
pip install stochastic_lasso
```

## Quick Start
```python
#Data load
import pandas as pd
X = pd.read_csv('https://raw.githubusercontent.com/datax-lab/StochasticLASSO/master/simulation_data/X.csv')
y = pd.read_csv('https://raw.githubusercontent.com/datax-lab/StochasticLASSO/master/simulation_data/y.csv')

#General Usage
from stochastic_lasso.stochastic_lasso import StochasticLasso

# Create a Stochastic LASSO model
S_Lasso = StochasticLasso(q='auto', r=30, logistic=False, alpha=0.05, random_state=None)

# Fit the model
S_Lasso.fit(X, y, sample_weight=None)

# Show the coefficients
S_Lasso.coef_

# Show the p-values
S_Lasso.p_values_

```
