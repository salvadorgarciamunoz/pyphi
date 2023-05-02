import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyphi",
    version="3",
    author="Sal Garcia",
    author_email="sgarciam@ic.ac.uk",
    description="A Python toolbox for multivariate analysis using PCA and PLS methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/salvadorgarciamunoz/pyphi",
    py_modules = ["pyphi","pyphi_plots"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "xlrd",
        "bokeh",
        "matplotlib",
        "pyomo"
    ],
) 
