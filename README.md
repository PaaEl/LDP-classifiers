# Required installation

To be able to run our code, python 3.9.1 is required with the following modules installed:

- numpy
- pandas
- json
- sklearn
- scipy
- pure_ldp
- statsmodels
  
Installing these modules in a linux terminal enviroment can be done using:

`pip install [module_name]`

# Running tests

The test can be run using the `testrun.py` file.
This file contains a `testrun()` function where the names of the databases, epsilon values and classifiers can be changed. 
The function will automatically run all tests on all possible combinations.
The results are written in a .csv file that is stored in the `Experiments` folder.

Terminal command to run the file: `python3 testrun.py`