# Shortcuts (Member 3). Windows users can run the commands after each target by hand.
PY ?= python

.PHONY: linear lgbm shap report test-models member3

linear:
	$(PY) -m src.models.linear

lgbm:
	$(PY) -m src.models.lgbm

shap:                ## needs lgbm first
	$(PY) -m src.explain

report:
	$(PY) scripts/member3_report.py

test-models:
	$(PY) -m pytest tests/test_models.py

member3: linear lgbm shap report
