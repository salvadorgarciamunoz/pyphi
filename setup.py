import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyphi",
    version="1.0.0",
    author="Sal Garcia",
    author_email="salvadorgarciamunoz@gmail.com",
    description="A Python toolbox for multivariate analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/salvadorgarciamunoz/pyphi",
    packages=setuptools.find_packages(),
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