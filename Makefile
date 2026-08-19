.PHONY: smoke test train

# Offline end-to-end smoke: build features, fit KMeans (prints inertia +
# silhouette + cluster sizes), then exercise the predict path.
smoke:
	python scripts/smoke.py

# Unit tests.
test:
	python -m pytest -q

# Full training run over the bundled dataset (elbow + silhouette sweep).
train:
	python -m src.cluster --config configs/default.yaml
