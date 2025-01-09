This code contains code for reproducing empirical results found in [Iterative Methods for Private Synthetic Data: Unifying Framework and New Methods](https://arxiv.org/abs/2106.07153). We have released a more user-friendly version of this code for running our algorithms [here](https://github.com/terranceliu/dp-query-release).

# Setup

Requires Python3. To install the necesssary packages, please run:
````
pip install -r requirements.txt
````

To run DualQuery, please install gurobipy for Anaconda
* Windows: https://www.gurobi.com/gurobi-and-anaconda-for-windows/
* MacOS: https://www.gurobi.com/gurobi-and-anaconda-for-mac/
* Linux: https://www.gurobi.com/gurobi-and-anaconda-for-linux/

# Data

The ADULT dataset has already been preprocessed and added to this repository. We also include proprocessed ADULT* (denoted as adult_orig in this repository) and LOANS datasets.

To obtain the ACS data used in our work, please follow these steps:

1) Go to the IPUMS USA website (https://usa.ipums.org/) and add the following variables to your data cart:
````
ACREHOUS, AGE, AVAILBLE, CITIZEN, CLASSWKR, DIFFCARE, DIFFEYE, DIFFHEAR, DIFFMOB, 
DIFFPHYS, DIFFREM, DIFFSENS, DIVINYR, EDUC, EMPSTAT, FERTYR, FOODSTMP, GRADEATT, 
HCOVANY, HCOVPRIV, HINSCAID, HINSCARE, HINSVA, HISPAN, LABFORCE, LOOKING, MARRINYR, 
MARRNO, MARST, METRO, MIGRATE1, MIGTYPE1, MORTGAGE, MULTGEN, NCHILD, NCHLT5, 
NCOUPLES, NFATHERS, NMOTHERS, NSIBS, OWNERSHP, RACAMIND, RACASIAN, RACBLK, RACE, 
RACOTHER, RACPACIS, RACWHT, RELATE, SCHLTYPE, SCHOOL, SEX, SPEAKENG, VACANCY, 
VEHICLES, VET01LTR, VET47X50, VET55X64, VET75X90, VET90X01, VETDISAB, VETKOREA, 
VETSTAT, VETVIETN, VETWWII, WIDINYR, WORKEDYR
````
2) Submit separate extract requests for the 1-yr samples (denoted as "ACS") for the years 2010, 2014, and 2018.
3) Rename the .dat and .xml files to acs_YEAR_1yr.dat and acs_YEAR_1yr.xml (for example: acs_2010_1yr.dat)
4) Move the files (6 in total) to ./Datasets/acs/
5) Run the following command**
````
python Util/process_data/process_ipums.py --fn acs_2010_1yr acs_2014_1yr acs_2018_1yr
````

# Execution

To reproduce the results found in the paper, please run the scripts found in ./scripts/<DATASET_NAME>/

For example, to run GEM on the 2018 ACS dataset for PA, use the following command:

````
./scripts/acs_PA/run_gem.sh
````

Note that to run GEM<sup>Pub</sup>, you must first pretrain a generator network via the code found in gem_nondp.py.

This can be done using the following scripts for the ACS and ADULT dataset:

````
./scripts/acs_PA/run_gem_nondp.sh
./scripts/adult/run_gem_nondp.sh
````

# Acknowledgements

We adapt code from

1) https://github.com/sdv-dev/CTGAN
2) https://github.com/ryan112358/private-pgm
3) https://github.com/terranceliu/pmw-pub
