all:
	@echo env or clean

env:
	pip3 install numpy
	pip3 install pandas
	pip3 install matplotlib

clean:
	rm -rf ft_linear_regression/linear_regression_lib/__pycache__
