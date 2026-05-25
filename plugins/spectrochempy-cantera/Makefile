.PHONY: install test clean

install:
	pip install -e .

test:
	python -m pytest tests/ -v

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
