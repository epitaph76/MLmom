# Runtime skills

The project has 17 English decision cards, version 1.0.0. Existing snake_case registry IDs remain stable; YAML display names use hyphens where needed.

SkillRegistry.load automatically adds candidate_sdk, validation, metrics and preprocessing, then deduplicates requested IDs. COMMON.md is prepended to candidate_sdk exactly once. The shared contract is included in its SHA256 and the 24,000-character limit. Candidate metadata records every loaded skill, including automatic ones. Manifests declare backend task compatibility and dependencies, independently of runtime availability probes.

Choose one backend for a hypothesis; add feature_engineering, imbalance, hpo or ensembles when needed. Include component library skills for ensembles. Prefer a cheap baseline, then LightGBM/CatBoost/XGBoost as appropriate. Sparse linear models need a custom sklearn candidate. Native AutoML is an alternative to shared Optuna, with patience_trials=null. A plan's preprocessing list is descriptive: generated code must implement new transformations.

The stock adapters are deliberately narrower than their libraries: sklearn selects fixed estimators; LightGBM/XGBoost use dense OHE; most boosters do not pass eval_set; statsmodels only supports unweighted single-output OLS; Torch accepts four MLP options. Cards distinguish stock behavior from custom candidate implementations. Do not copy research examples as if every parameter were an allowed model_params key.

LightAutoML's default objectives match regression/RMSE, binary/ROC-AUC and multiclass/log-loss. Its current plan validator accepts more name/task combinations than these native defaults; the skill explicitly requires an exact match. Temporal tasks remain deferred. Instructions do not add new adapter capabilities.

## Maintenance

Run tests/test_skills.py and the skill-creator quick_validate.py on each skill folder. Test registry budget/deduplication, shared-contract hashes, compatibility and flat-ZIP extraction; rerun pipeline integration when changing automatic loading. The full training environment is Python 3.12 with versions pinned in requirements.txt. Offline runtime uses installed API inspection rather than fetching these links.

## Primary API references

These sources support the cards; the project SDK, adapters, planner and evaluator remain authoritative for local behavior.

- sklearn: [API](https://scikit-learn.org/1.5/modules/classes.html)
- lightgbm: [Callback](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.early_stopping.html)
- catboost: [Fit](https://catboost.ai/en/docs/concepts/python-reference_catboost_fit)
- xgboost: [Pinned API](https://github.com/dmlc/xgboost/blob/v2.1.4/python-package/xgboost/sklearn.py)
- statsmodels: [OLS](https://www.statsmodels.org/v0.14.0/generated/statsmodels.regression.linear_model.OLS.html)
- feature_engineering: [TargetEncoder](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.TargetEncoder.html)
- flaml: [API](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/)
- autogluon: [Fit](https://auto.gluon.ai/1.3.1/api/autogluon.tabular.TabularPredictor.fit.html)
- lightautoml: [Preset](https://lightautoml.readthedocs.io/en/latest/pages/modules/generated/lightautoml.automl.presets.tabular_presets.TabularAutoML.html)
- PyTorch: [2.5 documentation](https://pytorch.org/docs/2.5/)
