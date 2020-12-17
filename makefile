all:
	@echo env or clean

env:
	pip3 install numpy
	pip3 install pandas

clean:
	rm -rf ft_linear_regression/linear_regression_lib/__pycache__
