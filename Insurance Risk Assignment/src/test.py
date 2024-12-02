# Import packages
import numpy as np 
import matplotlib.pyplot as plt  
import pandas as pd
import scipy.stats as stats 
import sympy as sy 
from scipy.optimize import fsolve
import time
import os

# Import my code 
from functions.plot import *
from models.model import model, model_gamma_4_16, model_gamma_02_08, model_question_6
from models.run_models import run_question

# Constants
LAMBDA_ARRIVAL= 4
CLAIMSIZE_MEAN = 16000 
TIME_HORIZON = 2000  

# Define parameters 
u0List = [16000 * i for i in [0,1,2,3,4,5]]
thetaList = [0.01, 0.1, 0.5, 0.9, 1 ,2]

LAMBDA_ARRIVAL= 4
CLAIMSIZE_MEAN = 16000 
TIME_HORIZON = 2000 

if __name__ == "__main__": 

    result = run_question(6)
    [print(i) for i in result]