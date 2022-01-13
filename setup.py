from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ciff',
    version='0.1.0',
    author='Emil T. S. Kjaer',
    author_email='etsk@chem.ku.dk',
    url='',
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

    install_requires = [
        'scipy',
        'pandas',
        'xgboost',
        'tqdm',
        'matplotlib'
    ],
    #extras_require = {
    #    'dev': [
    #        'pytest>=3.7',
    #    ],
    #},
)

# requirements.txt for deployment on machines that you control.
# pip freeze to genereate requirements.txt file.