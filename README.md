# Hi-LASSO2
High-Dimensional LASSO (Hi-LASSO) can theoretically improves a LASSO model providing better performance of both prediction and feature selection on extremely 
high-dimensional data.  Hi-LASSO alleviates bias introduced from bootstrapping, refines importance scores, improves the performance taking advantage of 
global oracle property, provides a statistical strategy to determine the number of bootstrapping, and allows tests of significance for feature selection with 
appropriate distribution.  In Hi-LASSO will be applied to Use the pool of the python library to process parallel multiprocessing to reduce the time required for 
the model.

## Installation
**Hi-LASSO2** support Python 3.6+. ``Hi-LASSO`` can easily be installed with a pip install::

```
pip install hi_lasso2
```

## Quick Start
```python
#Data load
import pandas as pd
X = pd.read_csv('https://raw.githubusercontent.com/datax-lab/Hi-LASSO/master/simulation_data/X.csv')
y = pd.read_csv('https://raw.githubusercontent.com/datax-lab/Hi-LASSO/master/simulation_data/y.csv')

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
