from setuptools import setup, find_packages

setup(
    name='stochastic_lasso',
    version='0.0.1',
    description='Stochastic LASSO',
    author='Beomsu Baek',
    author_email='qorqjatn9145@gnu.ac.kr',
    url='https://github.com/datax-lab/StochasticLASSO',
    install_requires=['glmnet', 'tqdm'],
    packages=find_packages(exclude=[]),
    keywords=['variable selection', 'feature selection', 'lasso', 'high-dimensional data'],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)