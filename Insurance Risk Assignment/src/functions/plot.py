# import matplotlib.pyplot as plt 
# import numpy as np 
# import pandas as pd 

# def plt_effect_of_theta(df, name): 
#     fig, axs = plt.subplots(nrows=2, ncols=3)
#     axs = axs.flatten() 

#     for i in range(len(df.index)): 
#         axs[i].scatter(x=df.columns, y=df[df.index==df.index[i]])

#     fig.savefig(f"./results/{name}_theta.png")

# def plt_effect_of_u(df, name): 
#     fig, axs = plt.subplots(nrows=2, ncols=3)
#     axs = axs.flatten() 

#     for i in range(len(df.index)): 
#         axs[i].scatter(x=df.index, y=df[df.index==df.index[i]])

#     fig.savefig(f"./results/{name}_u.png")


import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

# def plt_effect_of_theta(df, name):
#     """
#     Plot the effect of theta on the ruin probability, with theta as columns
#     and u as rows in a heatmap.
#     """
#     # Create a heatmap of the ruin probability, with u as rows and theta as columns
#     plt.figure(figsize=(10, 6))
#     plt.imshow(df, aspect='auto', cmap='viridis', origin='lower', 
#                extent=[df.columns.min(), df.columns.max(), df.index.min(), df.index.max()])
#     plt.colorbar(label="Ruin Probability")
#     plt.xlabel(r"$\theta$")
#     plt.ylabel(r"$u$")
#     plt.title(f"Effect of $\theta$ on Ruin Probability: {name}")
#     plt.tight_layout()
#     plt.savefig(f"./results/{name}_theta.png")
#     # plt.show()

# def plt_effect_of_u(df, name):
#     """
#     Plot the effect of u on the ruin probability, with u as rows
#     and theta as columns in a heatmap.
#     """
#     # Create a heatmap of the ruin probability, with u as rows and theta as columns
#     plt.figure(figsize=(10, 6))
#     plt.imshow(df.T, aspect='auto', cmap='viridis', origin='lower', 
#                extent=[df.columns.min(), df.columns.max(), df.index.min(), df.index.max()])
#     plt.colorbar(label="Ruin Probability")
#     plt.xlabel(r"$\theta$")
#     plt.ylabel(r"$u$")
#     plt.title(f"Effect of $u$ on Ruin Probability: {name}")
#     plt.tight_layout()
#     plt.savefig(f"./results/{name}_u.png")
#     # plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plt_effect_of_theta(df, name):
    """
    Plot the effect of theta on the ruin probability for fixed u values.
    """
    # Create a figure for the line plot
    plt.figure(figsize=(10, 6))
    
    # Iterate through the rows (different u values)
    for u in df.index:
        plt.plot(df.columns, df.loc[u], label=f"u = {u}")

    # Add labels and title
    plt.xlabel(r"$\theta$")
    plt.ylabel("Ruin Probability")
    plt.title(f"Effect of $\theta$ on Ruin Probability: {name}")
    plt.legend(title=r"$u$", loc="best")  # Legend with u values
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"./results/{name}_effect_of_theta.png")
    plt.show()

def plt_effect_of_u(df, name):
    """
    Plot the effect of u on the ruin probability for fixed theta values.
    """
    # Create a figure for the line plot
    plt.figure(figsize=(10, 6))
    
    # Iterate through the columns (different theta values)
    for theta in df.columns:
        plt.plot(df.index, df[theta], label=f"$\theta = {theta}$")

    # Add labels and title
    plt.xlabel(r"$u$")
    plt.ylabel("Ruin Probability")
    plt.title(f"Effect of $u$ on Ruin Probability: {name}")
    plt.legend(title=r"$\theta$", loc="best")  # Legend with theta values
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"./results/{name}_effect_of_u.png")
    plt.show()
