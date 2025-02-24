# Modules
## pyphi Release 4.0
Phi toolbox for multivariate analysis by Sal Garcia (salvadorgarciamunoz@gmail.com, sgarciam@ic.ac.uk)

Documentation: https://salvadorgarciamunoz.github.io/pyphi/index.html

## pyphi_plots
A variety of plotting tools for models created with pyphi.

## pyphi_batch
Batch analysis toolbox to perform batch alightment and Multi-way models for batch data

# Getting Started
Pyphi requires the  python packages listed in the  ```requirements.txt``` file.

## Installation
1) Ensure you have Python 3 installed and accessible via your terminal ("python" command).
2) Download this repository via ```git clone``` or manually using the download zip button at the top of the page.
3) Install the pyphi, pyphi_plots and pyphi_batch modules by opening a terminal window, navigating to the root of this repository, and typing 
```pip install -r requirements.txt```.

I find the use of Anaconda the easiest using Spyder, just download the code and add the path to Spyder

To confirm you have a working installation, navigate to the ```Examples``` folder and copy the ```Example_Script.py``` to the directory of your choice. Run ```python Example_Script.py```, verifying there are no errors logged to the console.


## Optional External Dependencies
- IPOPT as an executable in your system path or GAMS python module or GAMS executable in yoru system path.
  - Windows: ```conda install -c conda-forge IPOPT=3.11.1``` or download from [IPOPT releases page](https://github.com/coin-or/Ipopt/releases), extract and add the IPOPT\bin folder to your system path or add all files to your working directory.
  - Mac/Linux: ```conda install -c conda-forge IPOPT```, download from [IPOPT releases page](https://github.com/coin-or/Ipopt/releases), or [Compile using coinbrew](https://coin-or.github.io/Ipopt/INSTALL.html#COINBREW).
  
  - if GAMS is installed, pyphi will run ipopt via GAMS, make sure the GAMS executables are reachable through the system PATH

- If IPOPT is not detected, pyphi will submit the pyomo models to the NEOS server to solve them remotely.
  - To use the NEOS server, the environment variable "NEOS_EMAIL" must be assigned a valid email. This can be done outside of python using set/set/export or use ```import os
  os.environ["NEOS_EMAIL"] = youremail@domain.com```
  in your code.


Run the script '''Example_Script_testing_MD_by_NLP.py''' to verify that pyphi can execute IPOPT

Adding a folder to your system path:
 - Windows: temporary ```set PATH=C:\Path\To\ipopt\bin;%PATH%``` or persistent ```setx PATH=C:\Path\To\ipopt\bin;%PATH%```.
 - Mac/Linux: ```export PATH=/path/to/ipopt:$PATH```, add to .profile/.*rc file to make persistent.
 - Both via Conda: after activating your environment, use ```conda env config vars set``` and your OS-specific set or export command.

This is Release 4.0 the lastest added in terms of computation is the calculation of the Covariant scores and loadings (equivalent to what you get with OPLS) with the '''cca''' flag in the PLS routing
