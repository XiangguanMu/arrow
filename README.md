# ARROW

The `ARROW` repository contains code for *Arrow: Accelerator for Time Series Causal Discovery with Time Weaving* .

## Installation

After cloning the repository, please run

```shell
conda create --name arrow python=3.10.15
conda activate arrow
pip install -r requirements.txt
```

## Usage

To run the synthetic experiments, please run

```shell
cd code
python code/synthetic_test.py --lag constant --data patched --dataset linear --n 10 --model pcmci
```
### Parameter discription

1. **lag** (*str*, default: `multiple`) :  

    The lag type between variables.

    - `constant` : Represents a unified delay among all variables
    - `multiple` : Represents different delays between variables
2. **data** (*str*, default: `patched`) : 

    Data type passed into algorithm.

    - `raw` : The original synthetic data
    - `patched`: The patched data encoding with trend information
3. **dataset** (*str*, default: `linear`) : 

    Dataset used with `linear` or `nonlinear` setting.
4. **n** (*int*, default: `10`) : 

    The number of variables, i.e., time series.
5. **model** (*str*, default: `pcmci`) : 

    Select which method to be used. The choice range is `{pcmci, surd, varlingam, ngc}` .

    

