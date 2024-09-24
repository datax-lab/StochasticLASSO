# Hi-LASSO2
High-Dimensional LASSO2 (Hi-LASSO2) can significantly enhance the existing bootstrap-based LASSO models, providing better performance in both feature selection and coefficient estimation on extremely high-dimensional data.
Hi-LASSO2 systematically addresses the drawbacks of bootstrapping by reducing multicollinearity in bootstrap samples, mitigating randomness in predictor sampling, and providing a statistical strategy for selecting statistically significant features.
Hi-LASSO2 can be applied to any high-dimensional linear and logistic regression modeling.

## Installation
**Hi-LASSO2** support Python 3.6+. ``Hi-LASSO2`` can easily be installed with a pip install::

```
pip install hi_lasso2
```

## Quick Start
```python
#Data load
import pandas as pd
X = pd.read_csv('https://raw.githubusercontent.com/datax-lab/Hi-LASSO2/master/simulation_data/X.csv')
y = pd.read_csv('https://raw.githubusercontent.com/datax-lab/Hi-LASSO2/master/simulation_data/y.csv')

#General Usage
from hi_lasso2.hi_lasso2 import HiLasso2

# Create a Hi-LASSO2 model
HiLasso2 = HiLasso2(q='auto', r=30, logistic=False, alpha=0.05, random_state=None)

# Fit the model
HiLasso2.fit(X, y, sample_weight=None)

# Show the coefficients
HiLasso2.coef_

# Show the p-values
HiLasso2.p_values_

```
