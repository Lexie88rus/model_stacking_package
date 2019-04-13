# model_stacking_package
Python package for model stacking

## Overview
Repository contains Python package with implementation of stacking for sklean classification models based on
this __[Kaggle kernel](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)__.

## Package Contents
1. __setup.py__ - package setup file.
2. __stacking__ - folder, which contains package source code.
3. __init.py__  - init file, which contains shortcuts for *SklearnHelper* class and *get_oof* function.
4. __stacking_implementation.py__ - package source code.

## Package Installation
1. Download package.
2. Navigate to folder, containing package (specifically setup.py file).
3. Type
```
pip install .
```

## Package Usage Example
Example of usage of package *SklearnHelper* class and *get_oof* function 
```python
#import package
import stacking

#import sklearn model
from sklearn.ensemble import RandomForestClassifier

#define parameters for sklearn model
rf_params = {
    'n_jobs': -1,
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

#wrap model with SklearnHelper class from package
rf = SklearnHelper(clf=RandomForestClassifier, seed=42, params=rf_params)

#perform out of fold training of the model
rf_oof_train, rf_oof_test = get_oof(rf, X_train, y_train, X_test)
```

## References
1. __[Kaggle kernel](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)__
2. __[Sklearn library](https://scikit-learn.org/stable/)__
