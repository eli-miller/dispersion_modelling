# Setup to install VRPM_functions as a package
from setuptools import setup, find_packages

setup(
    name="VRPM_functions",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "cmcrameri",
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "scipy",
    ],
    entry_points={
        "console_scripts": [
            # If there are any scripts you want to run from the command line, add them here
            # For example:
            # 'my_script = my_package.my_module:my_function',
        ],
    },
)
