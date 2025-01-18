from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="megaprofiler",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib",
        "seaborn",
        "numpy",
        "tabulate",
        "scikit-learn",
        "imbalanced-learn",
        "statsmodels",
        "scipy"
    ],
    entry_points={
        'console_scripts': [
            'megaprofiler=megaprofiler.cli:main',  
        ],
    },
    description=(
        "megaprofiler is a highly customizable and extensible data profiling library "
        "designed to help data scientists and engineers understand their datasets "
        "before performing analysis or building models."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Siddharth Krishnan",
    author_email="sid@sidkrishnan.com",
    url="https://github.com/sidkris/megaprofiler",  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
