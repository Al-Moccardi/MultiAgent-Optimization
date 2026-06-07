.PHONY: install test dry clean
install:
	pip install -r requirements.txt
test:
	PYTHONPATH=. python -m part1_allocation.tests.test_milp
dry:
	PYTHONPATH=. python -m part1_allocation.pipeline.run_all --mode mock
clean:
	rm -rf shared/pareto/*.json shared/pareto/*.parquet
