# Direct statsmodels reference
```python
import statsmodels.api as sm
# Fit a saved imputer/encoder on train only, then:
values = self.preprocessor.fit_transform(X)
self.model = sm.OLS(np.asarray(y,dtype=float), sm.add_constant(values,has_constant='add')).fit(**self.params)
# predict:
values = self.preprocessor.transform(X.loc[:,self.columns])
prediction = self.model.predict(sm.add_constant(values,has_constant='add'))
```
Implement the complete Candidate and factories, not just this fragment. Reject sample weights for OLS (use an explicitly planned WLS candidate if required). Never silently drop validation rows. Persist preprocessing, columns and result with cloudpickle; keep dense widths bounded.
