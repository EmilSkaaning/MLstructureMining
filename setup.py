from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ciff',
    version='1.0.0',
    author='Emil T. S. Kjaer',
    author_email='etsk@chem.ku.dk',
    url='https://github.com/EmilSkaaning/ciff',
    description='Finds crystallographic information files!',
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=['ciff'],
    packages=['ciff'],

    entry_points={'console_scripts': [
        'ciff=ciff.cli:main',
    ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent',
    ],
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

# requirements.txt for deployment on machines that you control.
# pip freeze to genereate requirements.txt file.
