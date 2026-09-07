# HAT+TGA — convenience targets for development and testing
# Usage: make <target>

PYTHON     = uv run python
PYTEST     = uv run pytest
CKPT      ?= $(shell ls outputs/checkpoints/hat_tga/best*.ckpt 2>/dev/null | head -1)
# Allow override from environment
HATTGA_CKPT_PATH ?= $(CKPT)

.PHONY: help install install-data setup-data test test-env test-model test-results \
        train evaluate figures clean

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "HAT+TGA targets"
	@echo "  make install          Install core dependencies (uv sync)"
	@echo "  make install-data     Install [data] extras (download/process tools)"
	@echo "  make setup-data       Symlink Project_Wind/data/ into hat-tga/data/"
	@echo "  make test             Run full test suite"
	@echo "  make test-env         Test 01: package imports"
	@echo "  make test-model       Test 02: model forward pass (no data needed)"
	@echo "  make test-results     Test 03: reproduce manuscript numbers"
	@echo "  make train            Train HAT+TGA from scratch"
	@echo "  make evaluate CKPT=<path>  Evaluate checkpoint, write eval_results.json"
	@echo "  make figures  CKPT=<path>  Generate manuscript-style figures"
	@echo "  make clean            Remove outputs/ and __pycache__/"
	@echo ""

# ── Environment ───────────────────────────────────────────────────────────────
install:
	uv sync

install-data:
	uv sync --extra data

# ── Data ──────────────────────────────────────────────────────────────────────
setup-data:
	$(PYTHON) scripts/setup_test_data.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test: setup-data
	HATTGA_CKPT_PATH=$(HATTGA_CKPT_PATH) $(PYTEST) tests/ -v --tb=short 2>&1 | tee outputs/test_report.txt

test-env:
	$(PYTEST) tests/test_01_environment.py -v

test-model:
	$(PYTEST) tests/test_02_model.py -v

test-results: setup-data
	HATTGA_CKPT_PATH=$(HATTGA_CKPT_PATH) $(PYTEST) tests/test_03_results.py -v -s

# ── Workflows ─────────────────────────────────────────────────────────────────
train:
	$(PYTHON) scripts/train.py

evaluate:
	@if [ -z "$(HATTGA_CKPT_PATH)" ]; then \
		echo "Usage: make evaluate CKPT=outputs/checkpoints/hat_tga/best.ckpt"; exit 1; fi
	$(PYTHON) scripts/evaluate.py ckpt_path=$(HATTGA_CKPT_PATH)

figures:
	@if [ -z "$(HATTGA_CKPT_PATH)" ]; then \
		echo "Usage: make figures CKPT=outputs/checkpoints/hat_tga/best.ckpt"; exit 1; fi
	$(PYTHON) scripts/generate_figures.py ckpt_path=$(HATTGA_CKPT_PATH)

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
