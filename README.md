# MLstructureMining
Welcome to MLstructureMining!
This is a simple machine learning tool for structure characterization of metal-oxides using total scattering Pair 
Distribution Function (PDF) analysis.  
Simply provide a PDF and the model will output best best structural models from its structure catalog which contain approximately 11.000 crystal structures. 

1. [Install](#install)
2. [Usage](#usage)
3. [Authors](#authors)
4. [Cite](#cite)
5. [Acknowledgments](#acknowledgments)
6. [License](#license)
7. [Develop](#develop)

## Install
To install MLstructureMining you will need to have [Python](https://www.python.org/downloads/) or 
[Anaconda](https://www.anaconda.com/products/individual) installed. It is recommended to run MLstructureMining on Python version
3.7 or higher. If you have installed Anaconda you can create a new environment and activate it. 
```
conda create --name mlsm_env python=3.7
conda activate mlsm_env
```
Download the wheel from releases and install it with pip.
```
pip install mlstructuremining-<version>-py3-none-any.whl
```
To verify that MLstructureMining have been installed properly try calling the help argument.
```
mlstructuremining --help

>>> usage: mlstructuremining [-h] -d DATA [-n N_CPU] [-s SHOW] [-f FILE_NAME]
>>>                   
>>> This is a package which takes a directory of PDF files 
>>> or a specific PDF file. It then determines the best structural 
>>> candidates based of a metal oxide catalog. Results can
>>> be compared with precomputed PDF through Pearson analysis. 
```  
This should output a list of possible arguments for running MLstructureMining and indicates that it could find the package! 

## Usage
Now that MLstructureMining is installed and ready to use, lets discuss the possible arguments. The arguments are described in 
greater detail at the end of this section.

| Arg | Description | Default |  
| --- | --- |  --- |  
|  | __Optional arguments__ | |  
| `-h` or `--help` | Prints help message. |    
| `-n` or `--n_cpu_` | Number of cpus used by model. __int__ | `-n 1` 
| `-s` or `--show` | Number of best predictions printed. __int__ | `-s 5` 
| `-f` or `--file_name` | Name of the output file. __str__ | `-o ''` 
|  | __Required argument__ | | 
| `-d` or `--data` | A directory of PDFs or a specific PDF file. __str__ | `-d 5` 

# Authors
__Emil T. S. Kjær__<sup>1</sup>  
__Andy S. Anker__<sup>1</sup>   
__Kirsten M. Ø. Jensen__<sup>1</sup>    
 
<sup>1</sup> Department of Chemistry and Nano-Science Center, University of Copenhagen, 2100 Copenhagen Ø, Denmark.   

Should there be any question, desired improvement or bugs please contact us on GitHub or 
through email: __emil.thyge.kjaer@gmail.com__.

# Cite
If you use our code or our results, please consider citing our paper. Thanks in advance!
```
```

# Acknowledgments
Our code is developed based on the the following publication:
```
```

# License
This project is licensed under the Apache License Version 2.0, January 2004 - see the [LICENSE](LICENSE) file for details.

# Develop
Instal in developer mode.
```
$ pip install -e .[dev]
```
Build wheel from source distribution.
```
python setup.py sdist bdist_wheel
```
