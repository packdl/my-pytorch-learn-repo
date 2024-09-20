# my-pytorch-learn-repo

This is a repository containing Jupyter notebooks for CPSC Homework 1

## Installation

There is a requirements.txt file containing packages necessary to run the included Jupyter notebooks. These packages should be installed in a virtual environment using the following commands:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure the ipython kernel is part of the virtual environment.

```
pip install ipykernel
python -m ipykernel --name=hwenv
```

In Jupyter, set the kernel to hwenv.

Afterwards, you should be able to install and rerun any of the Jupyter notebooks.

## Files
- [Homework-part1](homework-part1.ipynb). This contains all section of HW 1-1
- [Homework-part2](homework-part2.ipynb). 
    - Visualize the optimization process.
    - Observe gradient norm during training
- [Homework-part2-gradientzero](homework-part2-gradzero.ipynb). What happens when gradient is almost zero.
- [Homework-part3-1](homework-part3-1.ipynb).  Can network fit random labels
- [Homework-part3-2](homework-part3-2.ipynb). Number of parameters vs generalization
- [Homework-part3-3-1](homework-part3-3-1.ipynb) Flatness vs generalization part 1
- [Homework-part3-3-2](homework-part3-3-2.ipynb) Flatness vs generalization part 2