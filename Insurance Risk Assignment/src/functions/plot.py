import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

def plt_effect_of_theta(df, name): 
    fig, axs = plt.subplots(nrows=2, ncols=3)
    axs = axs.flatten() 

    for i in range(len(df.index)): 
        axs[i].scatter(x=df.columns, y=df[df.index==df.index[i]])

    fig.savefig(f"./rsults/{name}")

def plt_effect_of_u(df, name): 
    fig, axs = plt.subplots(nrows=2, ncols=3)
    axs = axs.flatten() 

    for i in range(len(df.index)): 
        axs[i].scatter(x=df.index, y=df[df.index==df.index[i]])

    fig.savefig(f"./rsults/{name}")
