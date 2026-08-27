# Convenience targets. On Windows the interpreter is .venv/Scripts/python.exe;
# these targets assume a POSIX shell, so use the explicit commands from
# docs/REPRODUCIBILITY.md there.
PY := ./.venv/bin/python
export OMP_NUM_THREADS = 2
export MKL_NUM_THREADS = 2

.PHONY: setup test lint smoke bench experiments ablations analyse figures docs notebooks all clean

setup:
	uv python install 3.12
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) --index-url https://download.pytorch.org/whl/cpu \
		--extra-index-url https://pypi.org/simple torch
	uv pip install --python $(PY) numpy scipy pandas pillow pyyaml matplotlib \
		pytest ruff nbformat nbconvert ipykernel
	uv pip install --python $(PY) -e . --no-deps

lint:
	$(PY) -m ruff check src scripts tests

test:
	$(PY) -m pytest tests -q -m "not slow"

test-all:
	$(PY) -m pytest tests -q

smoke:
	$(PY) scripts/train.py --config configs/smoke.yaml

bench:
	$(PY) scripts/benchmark_efficiency.py

experiments:
	$(PY) scripts/run_experiments.py --seeds 0 1 2

ablations:
	$(PY) scripts/run_ablations.py --seed 0
	$(PY) scripts/run_ablations.py --seed 1
	$(PY) scripts/run_ablations.py --seed 2 --summarise

analyse:
	$(PY) scripts/analyse.py --seeds 0 1 2

figures:
	$(PY) scripts/make_figures.py

docs:
	$(PY) scripts/render_docs.py

notebooks:
	$(PY) scripts/build_notebooks.py --execute

all:
	$(PY) scripts/run_all.py

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
