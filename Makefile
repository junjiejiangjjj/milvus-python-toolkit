.PHONY: help submodules deps deps-test install install-fast build build-native test lint check smoke wheel-contents clean

PYTHON ?= python
PIP := $(PYTHON) -m pip
DIST_DIR := dist
WHEEL := $(shell ls -t $(DIST_DIR)/milvus_toolkit-*.whl 2>/dev/null | head -n 1)

help:
	@printf "Milvus Python Toolkit targets:\n"
	@printf "  make install        Install with bundled milvus_storage from submodule\n"
	@printf "  make install-fast   Editable install for unit tests without native build\n"
	@printf "  make build          Build the bundled milvus-toolkit wheel\n"
	@printf "  make build-native   Build and stage milvus-storage native library only\n"
	@printf "  make test           Run pytest\n"
	@printf "  make lint           Run ruff\n"
	@printf "  make check          Run test and lint\n"
	@printf "  make smoke          Run bundled milvus_storage smoke test\n"
	@printf "  make wheel-contents List bundled milvus_storage files in latest wheel\n"
	@printf "  make clean          Remove local build/test artifacts\n"

submodules:
	git submodule update --init --recursive

deps:
	$(PIP) install --upgrade pip
	$(PIP) install "build>=1" "conan>=2"

deps-test:
	$(PIP) install "pyarrow>=17.0.0" "numpy>=1.20.0" "cffi>=1.17.1" "pandas>=2.3.3" "pytest>=7" "ruff>=0.4"

install: deps submodules
	$(PIP) install .

install-fast: deps-test
	$(PIP) install -e . --no-deps

build: deps submodules
	$(PYTHON) -m build --wheel

build-native: submodules
	$(PYTHON) scripts/build_milvus_storage.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

check: test lint

smoke:
	MILVUS_STORAGE_IMPORT_SMOKE=1 $(PYTHON) -m pytest tests/integration/test_milvus_storage_install.py

wheel-contents:
	@if [ -z "$(WHEEL)" ]; then \
		printf "No milvus_toolkit wheel found in $(DIST_DIR). Run make build first.\n"; \
		exit 1; \
	fi
	$(PYTHON) -m zipfile -l "$(WHEEL)" | grep 'milvus_toolkit/_vendor/milvus_storage'

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
