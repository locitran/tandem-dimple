[![Build Status](https://travis-ci.org/smsaladi/pyRONN.svg?branch=master)](https://travis-ci.org/smsaladi/pyRONN)
[![Coverage](https://img.shields.io/codecov/c/github/smsaladi/pyRONN/master.svg)](https://codecov.io/github/smsaladi/pyRONN/)

RONN
====

This is a repackaged version of RONN 3.2 after significant rewrite (3.3) and
wrapping to be directly called as a Python extension. The codebase is thoroughly
regression-tested; disorder scores are the same as predicted by RONN 3.2.

## Installation

Stable releases version from PyPI
```shell
pip install pyRONN
```

Development version from Github
```shell
pip install git+https://github.com/smsaladi/pyRONN.git
```


## History

### RONN 3.2

This program is designed for for detecting disordered residues in proteins.

Date:	March 2012

Designer:	Zheng Rong Yang (Exeter University)

Programmer:	Varun Ramraj (University of Oxford)

Contact:	varun@strubi.ox.ac.uk


### RONN 3.3
Substantial modifications to speed up code as well as facilitate wrapping
with Python (or any other language). Can now also handle multiple
FASTA-formatted records in an input file. Path to `data` directory must be
specified as a command-line argument if not located at `../data` with respect
to the executable file.

Date:    Sept 2016

Author:  Shyam Saladi (California Institute of Technology)

Contact: saladi@caltech.edu
