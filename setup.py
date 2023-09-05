from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='mlstructuremining',
    version='04.01.00',
    author='Emil T. S. Kjaer',
    author_email='etsk@chem.ku.dk',
    url='https://github.com/EmilSkaaning/MLstructureMining',
    description='Predicts suitable crystal structures from Pair Distribution Function (PDF) data.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=['mlstructuremining'],
    packages=['mlstructuremining'],

    entry_points={'console_scripts': [
        'mlstructuremining=mlstructuremining.cli:main',
    ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
    package_data={'mlstructuremining': ['model/*']},  # Include all files in the model directory
    include_package_data = True,
    zip_safe=False,

    install_requires=[
        'scipy==1.7.3',
        'pandas==1.3.5',
        'xgboost==1.5.2',
        'tqdm==4.62.3',
        'numpy==1.21.5',
        'matplotlib==3.5.1',
        'h5py==3.6.0',
    ],
)

