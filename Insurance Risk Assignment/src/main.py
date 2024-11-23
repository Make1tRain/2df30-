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

# Question 1
print("[i] Running Simulations for question 1")
q1_df = run_question(1)

plt_effect_of_theta(q1_df, "q1")
plt_effect_of_u(q1_df, "q1")

# Question 2 

# Question 3 

# Question 4

# Question 5

# Gamma(4, 16)
print("[i] Running Simulations for question 1")
q5_1_df = run_question(5.1)

plt_effect_of_theta(q5_1_df, "q5_1")
plt_effect_of_u(q5_1_df, "q5_1")

# Gamma(0.2, 0.8)
print("[i] Running Simulations for question 1")
q5_2_df = run_question(5.2)

plt_effect_of_theta(q5_2_df, "q5_2")
plt_effect_of_u(q5_2_df, "q5_2")