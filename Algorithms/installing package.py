# Installing necessary packages and algorithms

!pip install scikit-quant
!pip install -U pymoo
!pip install gpytorch
!pip install disjoint-set
!pip install HEBO

from skquant.opt import minimize as sk_minimize

# other python packages
import copy
import numpy as np
import scipy
from numpy import mean
from scipy.optimize import (
    minimize,
    differential_evolution,
    basinhopping,
    direct,
    Bounds
)
from tqdm import tqdm
import matplotlib.pyplot as plt
