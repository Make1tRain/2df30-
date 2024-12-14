import numpy as np 
import random 


def calculate_x(pd:float) -> bool: 
    return 1 if random.random() < pd else 0 




# -------------------- ABN AMOR -------------------- 
# 8000 mortgages 
# EAD = 1000 
# LGD = 0.25 
# PD = 0.01 (Probability of default) 

def abn_amor(): 
    n = 8000 
    EAD = 1000 
    LGD = 0.25 
    PD = 0.01 
    
    pdArray = np.ones()


print(abn_amor())

# -------------------- NSN BANK -------------------- 
# NOTE: Exposure at default is the full amount 
# 1000 mortgages of 1k EUR 
# 10 mortgages of 10k EUR 
# 4 mortgages of 50k EUR
# 2 mortgages of 100k EUR 
# 1 mortgage of 500k EUR 

# LGD = 0.5
# PD = 0.02 (Probability of default) 


# -------------------- ROBA BANK -------------------- 
# NOTE: Exposure at default is the full amount 
# 1000 mortgages of 1k EUR  (PD = 0.0275)
# 10 mortgages of 10k EUR   (PD = 0.02)
# 4 mortgages of 50k EUR  (PD = 0.0175)
# 2 mortgages of 100k EUR   (PD = 0.015)
# 1 mortgage of 500k EUR  (PD = 0.008)
# LGD = 0.5



def bank(EAD:np.array, LGD:float, PD:np.array): 
    n = len(EAD)

    pdArray = np.array(list(map(calculate_x, np.ones(n) * PD)))
    Li = EAD * LGD * pdArray
    Ln = Li.sum()
    
    return Ln













