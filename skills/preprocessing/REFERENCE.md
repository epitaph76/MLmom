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
