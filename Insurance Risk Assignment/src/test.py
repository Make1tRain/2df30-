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
from models.model import model, model_gamma_4_16, model_gamma_02_08
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

# m = model_gamma_4_16(TIME_HORIZON, LAMBDA_ARRIVAL, CLAIMSIZE_MEAN)
# m.simulate_one_pair(16000, 0.01, n=5)

# model_gamma_02_08()

q1_df = pd.read_csv("./results/q1_model_simulation_results.csv", index_col=0)
print(q1_df)

plt_effect_of_theta(q1_df, "q1")
plt_effect_of_u(q1_df, "q1")