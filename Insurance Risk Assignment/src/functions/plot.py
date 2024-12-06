import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

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
