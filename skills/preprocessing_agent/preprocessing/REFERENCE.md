# Separate preprocessing module example

Adapt to the planner; dense OHE is inappropriate for wide text. The model coder receives the declared dense output.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class Preprocessor:
    def clean(self, X):
        return X.loc[:, self.columns].replace([np.inf, -np.inf], np.nan)

    def fit_transform(self, X, y, context):
        self.columns = X.columns.tolist()
        X = self.clean(X)
        numeric = X.select_dtypes(include=['number', 'bool']).columns.tolist()
        categorical = [c for c in self.columns if c not in numeric]
        self.transformer = ColumnTransformer([
            ('number', SimpleImputer(strategy='median', keep_empty_features=True), numeric),
            ('category', Pipeline([
                ('fill', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=100))
            ]), categorical)
        ])
        return self.transformer.fit_transform(X)

    def transform(self, X):
        return self.transformer.transform(self.clean(X))

def create_preprocessor(config):
    return Preprocessor()
```

# Sparse text and numeric-coded categories
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def text_values(column):
    return ('__document__ ' + column.fillna('').astype(str)).to_numpy()

# Select exact text/numeric/category columns in fit and persist them.
text_pipe = Pipeline([('text',FunctionTransformer(text_values,validate=False)),
    ('tfidf',TfidfVectorizer(max_features=30000,ngram_range=(1,2)))])
# ColumnTransformer text selector is one column NAME (1D), not [name] (2D).
# Numeric: median imputation then StandardScaler(with_mean=False).
# Categorical: missing sentinel and OneHotEncoder(handle_unknown='ignore',sparse_output=True).
# sparse_threshold=1.0 preserves sparse concatenation; do not call toarray on wide text.
```
For native tree categories, persist train-derived levels and transform all later frames identically. Numeric Year_Factor needs an explicit cast if planned. An imputer that fills NaN contradicts a hypothesis preserving native missing routing. Fit clipping bounds/target transforms on train, apply inverse transforms before original-scale scoring and clipping explicitly in predict when planned. The generic preprocessing is a reference, not mandatory behavior.
