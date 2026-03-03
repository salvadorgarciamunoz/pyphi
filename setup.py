import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyphi-mvda",
    version="6.0",
    license="MIT",
    author="Sal Garcia",
    author_email="sgarciam@ic.ac.uk",
    description="A Python toolbox for multivariate analysis using PCA and PLS methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/salvadorgarciamunoz/pyphi",
    packages=setuptools.find_packages(),   # replaces py_modules
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "bokeh",
        "matplotlib",
        "numpy",
        "openpyxl",
        "pandas",
        "pyomo",
        "scipy",
        "statsmodels",
    ],
)
