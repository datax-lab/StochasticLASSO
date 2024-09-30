from setuptools import setup, find_packages

setup(
    name='hi_lasso2',
    version='0.0.1',
    description='High-Demensional LASSO2',
    author='Beomsu Baek',
    author_email='qorqjatn1388@gnu.ac.kr',
    url='https://github.com/datax-lab/Hi-LASSO2',
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