# Modules
## pyphi
phi toolbox for multivariate analysis by Sal Garcia (salvadorgarciamunoz@gmail.com , sgarciam@ic.ac.uk )
version 1.0 includes: Principal Components Analysis (PCA), Projection to Latent Structures (PLS), Locally Weighted PLS (LWPLS), Savitzy-Golay derivative and Standard Normal Variate pre-processing for spectra.

## pyphi_plots
A variety of plotting tools for models created with pyphi. 

# Getting Started
## Dependencies
numpy, scipy, pandas, xlrd, bokeh, matplotlib, pyomo

## Optional External Dependencies
- IPOPT as an executable in system path or GAMS python module or GAMS executable in system path. (Otherwise, solve NLPs using the NEOS server)
  - All platforms: ```conda install -c conda-forge  IPOPT```
  - Windows: [IPOPT releases page](https://github.com/coin-or/Ipopt/releases)
  - Mac/Linux: [Compile using coinbrew](https://coin-or.github.io/Ipopt/INSTALL.html#COINBREW)
- libhsl with ma57 within library loading path (Speeds up IPOPT for large problems but requires a free academic or paid industrial license and a local IPOPT installation)

## Installation
1) Download this repository via ```git clone``` or manually using the download zip button at the top of the page.
2) Install python dependencies using either ```pip install -r requirements.txt``` or ```conda install -c conda-forge -file requirements```. You may wish to create a new virtual environment using conda or venv before installing.
3) Add pyphi to your python path so it can be imported in python code
	- Windows: In the cmd terminal enter ```set PYTHONPATH=C:\Path\To\pyphi;%PYTHONPATH%``` to  set for that session or ```setx PYTHONPATH C:\Path\To\pyphi;%PYTHONPATH%``` to set it system wide.
	- Mac/Linux: In the terminal, use ```export PYTHONPATH=/path/to/pyphi:$PYTHONPATH``` to set for that session or add to your *rc/profile file to automatically set it.
	- Using conda: After you have activated your virtual environment via ```conda activate yourenv```, call ```conda env config vars set PYTHONPATH=C:\Path\To\pyphi;%PYTHONPATH%``` (Windows) or ```conda env config vars set PYTHONPATH=C:\Path\To\pyphi;%PYTHONPATH%``` (Mac/Linux). Then reactivate your environment with ```conda activate yourenv``` . This will update your PYTHONPATH automatically within your environment.
4) Download listed external dependencies, putting libhsl into the same directory as IPOPT. Then add to path
	- Windows: ```set PATH=C:\Path\To\ipopt;%PATH%```
	- Mac/Linux: ```export PATH=/path/to/ipopt:$PATH```
	- Using conda: Use the OS specific command with ```conda env config vars set``` as in step 3 to automatically set each time.

To confirm you have a working installation, copy the file ```Examples/Example_Script_testing_MD_by_NLP.py``` to a new directory, run it using ```python Example_Script_testing_MD_by_NLP.py```, and verity that there are no errors logged to the console.
	
# What is New
#### May 28th
* Enhanced clean_low_variances function to return a list with columns removed from dataframe
#### May 27th
* PLS model estimation using Non-linear programming as described in Journal of Chemometrics, 28(7), pp.575-584.
#### March 30th
* PCA model estimation using Non-linear programming as described in Lopez-Negrete et al. J. Chemometrics 2010; 24: 301–311
