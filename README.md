# Modules
## pyphi
Phi toolbox for multivariate analysis by Sal Garcia (salvadorgarciamunoz@gmail.com, sgarciam@ic.ac.uk)

Version 1.0 includes: Principal Components Analysis (PCA), Projection to Latent Structures (PLS), Locally Weighted PLS (LWPLS), Savitzy-Golay derivative and Standard Normal Variate pre-processing for spectra.

## pyphi_plots
A variety of plotting tools for models created with pyphi. 

# Getting Started
Pyphi requires the following python packages: numpy, scipy, pandas, xlrd, bokeh, matplotlib, pyomo. These can be installed via setup.py below, or manually using pip/conda and the ```requirements.txt``` file.

## Installation
1) Download this repository via ```git clone``` or manually using the download zip button at the top of the page.
2) Install the pyphi and pyphi_plots modules by opening a terminal window, navigating to the root of this repository, and typing ```python setup.py install```
   - You may want to strongly consider using a virtual environment (create with conda, ```conda create -n yourenv python``` or venv ```python -m venv yourenv```, then activate ```conda activate yourenv``` or venv Windows ```yourenv\Scripts\activate.bat``` or venv Linux/mac ```source yourenv/bin/activate``` ) to avoid interfering with your system python installation. 
 
To confirm you have a working installation, navigate to the ```Examples``` directory and run ```python Example_Script_testing_MD_by_NLP.py```, verifying there are no errors logged to the console.

## Optional External Dependencies
- IPOPT as an executable in your system path or GAMS python module or GAMS executable in yoru system path. (Otherwise, will solve pyomo NLPs remotely using the NEOS server)
  - Windows: ```conda install -c conda-forge IPOPT=3.11.1``` or download from [IPOPT releases page](https://github.com/coin-or/Ipopt/releases), extract and add the IPOPT\bin folder to your system path or add all files to your working directory
  - Mac/Linux: ```conda install -c conda-forge IPOPT```, download from [IPOPT releases page](https://github.com/coin-or/Ipopt/releases), or [Compile using coinbrew](https://coin-or.github.io/Ipopt/INSTALL.html#COINBREW)
- libhsl with ma57 within library loading path or in the same directory as IPOPT executable
   - Speeds up IPOPT for large problems but requires a free academic or paid industrial license and a local IPOPT installation
   - Must request in advance and building the source code is nontrivial. Expert use only.

Adding a folder to your system path:
 - Windows: temporary ```set PATH=C:\Path\To\ipopt\bin;%PATH%``` or persistent ```setx PATH=C:\Path\To\ipopt\bin;%PATH%```
 - Mac/Linux: ```export PATH=/path/to/ipopt:$PATH```, add to .profile/.*rc file to make persistent
 - Both via Conda: after activating your environment, use ```conda env config vars set``` and your OS-specific set or export command


# What is New
#### May 28th
* Enhanced clean_low_variances function to return a list with columns removed from dataframe
#### May 27th
* PLS model estimation using Non-linear programming as described in Journal of Chemometrics, 28(7), pp.575-584.
#### March 30th
* PCA model estimation using Non-linear programming as described in Lopez-Negrete et al. J. Chemometrics 2010; 24: 301–311
