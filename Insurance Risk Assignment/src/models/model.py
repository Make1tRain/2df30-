import numpy as np 
import matplotlib.pyplot as plt  
import pandas as pd
import scipy.stats as stats 
import sympy as sy 
from scipy.optimize import fsolve
import time
import os
import multiprocessing


# Question 1
class _model_functions(): 
    def set_name(self, name="model"):
        self.name = name 

    def plot(self): 
        self.df.plot()

    def savefig(self):
        if not os.path.isdir("./results"): 
            os.mkdir("./results")

        self.df.to_csv(f'./results/{self.name}_simulation_results.csv')
        
    def simulate_once_and_plot(self, u0, theta):
        c = (1 + theta) * self.lambda_arrival * self.claimSize_mean  # self.claimSize_mean is E(X_i)
        ruin, D, recoveryTime = self.run(u0, c)

        # plt.figure()
        # plt.plot(arrivalTimes, levels, "b") 
        # plt.plot(arrivalTimes, [0 for i in range(len(arrivalTimes))], "r")

    
    def _print_info(self, u0List, thetaList): 
        print(
            f"""
Time Horizon: {self.time_horizon} days.
u0 List: {u0List}
Safety Loading List: {thetaList}

"""
        )

class _model_disributions(): 
    def set_interArrival_time(self): 
        self.interArrivalDist =  stats.expon(scale=1 / self.lambda_arrival)
    
    def set_number_of_claims_dist(self, time_horizon, lam):
        # 1.2 - Number of Claims (Poisson(lambda * t)) (lambda is 4 per day)

        # return stats.poisson(mu=time_horizon * lam)
        self.numberOfClaimsDist = time_horizon * lam 
    
    def set_claim_sizes_dist(self): 
        # 1.1 - Claim sizes (Uniform distribution with a mean of 16,000 and a variance of 12,000,000)
        # X ~ Uniform(10000, 22000)
        claimSizeDistribution = stats.uniform(loc=10000, scale=22000 - 10000)

        self.claimSizeDist = claimSizeDistribution


class model(_model_functions, _model_disributions):
    def __init__(self, time_horizon, lambda_arrival, claimSize_mean):
        self.time_horizon = time_horizon 
        self.lambda_arrival = lambda_arrival
        self.claimSize_mean = claimSize_mean 

        self.set_number_of_claims_dist(self.time_horizon, self.lambda_arrival)
        self.set_interArrival_time()
        self.set_claim_sizes_dist()
        self.set_name()
        

    def run(self, u0, c):
        # we first get a sample from the number of claims RV N(t) ~ Pois(lam * t)
        N = round(self.time_horizon*self.lambda_arrival)

        # We get interarrival times ~ Expon(lambda) which is a property of PP
        interArrivals = self.interArrivalDist.rvs(N)
        arrivalTimes = np.cumsum(interArrivals)

        # We sample the claim sizes
        claimSizes = self.claimSizeDist.rvs(N)
        cumulativeClaims = np.cumsum(claimSizes)

        # Calculate U(t) at every t
        levels = u0 + arrivalTimes * c - cumulativeClaims

        if len(levels) == 0:
            print(levels, N) 

        minimumPoint = min(levels)
        ruin = minimumPoint < 0
        
        return ruin

    def simulate_one_pair(self, u0, theta, n):
        start = time.time()
        result = np.zeros(n)
        c = (1 + theta) * self.lambda_arrival * self.claimSize_mean  # self.claimSize_mean is E(X_i)

        for i in range(n):
            ruin = self.run(u0, c)
            result[i] = 1 if ruin == True else 0
            print(f"[i] (u={u0},theta={theta}) Run: {i}")

        end = time.time()
        print(
            f"[i] (u={u0},theta={theta}) Time taken for {n} simulations: {end - start} seconds, this is equal to {(end - start) / n} seconds per run."
        )
        return np.mean(result)
    
    def simulate(self, u0List, thetaList, n=1000):

        self._print_info(u0List, thetaList)

        result = [[[]] * len(thetaList)] * len(u0List)

        for i, u0 in enumerate(u0List):
            for j, theta in enumerate(thetaList):
                result[i][j] = self.simulate_one_pair(u0, theta, n)

        self.df = pd.DataFrame(data=result, columns=thetaList, index=u0List)
        self.savefig() 
        return self.df
    
    def simulate_multiprocessing(self, u0List, thetaList, n=1000):

        result_queue = multiprocessing.Queue()
        processes = []

        # Create a multiprocessing pool to run the simulation for each (u, theta) pair
        for i, u in enumerate(u0List):
            for j, theta in enumerate(thetaList):
                # Each process will call process_pair
                p = multiprocessing.Process(target=process_pair, args=(u, theta, (i, j), result_queue, self, n))
                processes.append(p)
                p.start()

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Gather the results from the queue
        result = [[None] * len(thetaList) for _ in range(len(u0List))]
        while not result_queue.empty():
            index_pair, simulation_result = result_queue.get()
            i, j = index_pair
            result[i][j] = simulation_result

        # Create DataFrame with the result
        self.df = pd.DataFrame(columns=thetaList, data=result)
        self.df.index = u0List
        self.savefig()

        return self.df

def process_pair(u, theta, index_pair, result_queue, model, n):
    """Function to be run in each process to simulate the pair (u, theta)."""
    print(f"({u},{theta}) Process Created")
    result_queue.put((index_pair, model.simulate_one_pair(u, theta, n)))




# You are free to choose several probability distributions,
# but we ask you to always include the gamma(4, 16) and the gamma(0.2, 0.8) distributions.

# Define a new model based on the previous one 

# Question 5
class model_gamma_4_16(model): 
    def __init__(self, time_horizon, lambda_arrival, claimSize_mean):
        super().__init__(time_horizon, lambda_arrival, claimSize_mean)

    def set_interArrival_time(self):     
        self.interArrivalDist =  stats.gamma(a=4, scale=1/16)

class model_gamma_02_08(model): 
    def __init__(self, time_horizon, lambda_arrival, claimSize_mean):
        super().__init__(time_horizon, lambda_arrival, claimSize_mean)

    def set_interArrival_time(self):     
        self.interArrivalDist = stats.gamma(a=0.2, scale=1/0.8)

# Question 6

class model_question_6(model): 
    def run(self, u0, c):
        # we first get a sample from the number of claims RV N(t) ~ Pois(lam * t)
        N = round(self.time_horizon*self.lambda_arrival)

        # We get interarrival times ~ Expon(lambda) which is a property of PP
        interArrivals = self.interArrivalDist.rvs(N)
        arrivalTimes = np.cumsum(interArrivals)

        # We sample the claim sizes
        claimSizes = self.claimSizeDist.rvs(N)
        cumulativeClaims = np.cumsum(claimSizes)

        # Calculate U(t) at every t
        levels = u0 + arrivalTimes * c - cumulativeClaims

        if len(levels) == 0:
            print(levels, N) 

        minimumPoint = min(levels)
        ruin = minimumPoint < 0

        defecitAtRuin, recoveryTime = None, None 

        # if ruin happens: 
        if bool(ruin): 
            # We need T: moment of first ruin, if ruin occurs; and U(T): capital at ruin         
            ruinLevels = [(i, val) for i,val in enumerate(levels) if val < 0] # every item is (index of ruin, u(t))
            firstRuin = ruinLevels[0] # (index of ruin, u(T))
            
            ruinIndex = firstRuin[0]
            defecitAtRuin = - firstRuin[1]

            recoveryIndex = None
            for i, val in list(enumerate(levels))[firstRuin[0]:]: 
                # we find the moment where it recovers 
                if val >= 0: 
                    recoveryIndex = i
                    break  

            if ruinIndex and recoveryIndex:  
                recoveryTime = arrivalTimes[recoveryIndex] - arrivalTimes[ruinIndex]

        # if ruins: ruin = 1, defecitAtRuin = number, recoveryTimes depends if it recovers or not  
        # if not ruins: ruin = 0, defecitAtRuin = None, recoveryTimes = None
        return ruin, defecitAtRuin, recoveryTime
    
    def simulate_one_pair(self, u0, theta, n):
        start = time.time()
        ruinList = np.zeros(n)
        DList = [0] * n
        RList = [0] * n
        c = (1 + theta) * self.lambda_arrival * self.claimSize_mean  # self.claimSize_mean is E(X_i)

        for i in range(n):
            ruin, D, recoveryTime = self.run(u0, c)
            ruinList[i] = 1 if ruin == True else 0
            DList[i] = D
            RList[i] = recoveryTime
            print(f"[i] (u={u0},theta={theta}) Run: {i}")

        end = time.time()
        print(
            f"[i] (u={u0},theta={theta}) Time taken for {n} simulations: {end - start} seconds, this is equal to {(end - start) / n} seconds per run."
        )
        
        # ruin, D, R
        DList = [round(i, 4) for i in DList if i != None]
        RList = [round(i, 4) for i in RList if i != None]

        return np.mean(ruinList), np.mean(DList), np.mean(RList) 

    def simulate_multiprocessing(self, u0List, thetaList, n=1000):
        # Create a multiprocessing pool to run the simulation for each (u, theta) pair asynchronously
        with multiprocessing.Pool() as pool:
            # Prepare a list of tasks (each task corresponds to a (u, theta) pair)
            tasks = [
                (u, theta, (i, j), self, n)  # Prepare arguments for each simulation pair
                for i, u in enumerate(u0List)
                for j, theta in enumerate(thetaList)
            ]
            
            # Use pool.map_async to run simulations in parallel (asynchronously)
            results = pool.starmap_async(process_pair_q6, tasks)

            # Wait for all processes to complete
            pool.close()
            pool.join()

        # Prepare the results: Create 3 dataframes for Ruin Probability, D, and R
        ruin_result = [[None] * len(thetaList) for _ in range(len(u0List))]
        D_result = [[None] * len(thetaList) for _ in range(len(u0List))]
        R_result = [[None] * len(thetaList) for _ in range(len(u0List))]

        # Collect results from the pool
        for result in results.get():  # Retrieve the results
            index_pair, ruin_prob, D, R = result
            i, j = index_pair
            ruin_result[i][j] = ruin_prob
            D_result[i][j] = D
            R_result[i][j] = R

        # Create DataFrames for ruin probability, D, and R
        ruin_df = pd.DataFrame(columns=thetaList, data=ruin_result)
        D_df = pd.DataFrame(columns=thetaList, data=D_result)
        R_df = pd.DataFrame(columns=thetaList, data=R_result)

        # Set the index of all DataFrames to the u0List
        ruin_df.index = u0List
        D_df.index = u0List
        R_df.index = u0List

        ruin_df.to_csv("./results/q6_ruin_simulation_results.csv")
        D_df.to_csv("./results/q6_D_simulation_results.csv")
        R_df.to_csv("./results/q6_R_simulation_results.csv")

        # Return the three DataFrames
        return ruin_df, D_df, R_df


def process_pair_q6(u, theta, index_pair, model, n):
    """Function to be run in each process to simulate the pair (u, theta)."""
    print(f"({u},{theta}) Process Created")
    
    # Get the three values (ruin probability, D, R) from simulate_one_pair
    ruin_prob, D, R = model.simulate_one_pair(u, theta, n)
    
    # Return the result (index_pair, ruin_prob, D, R)
    return (index_pair, ruin_prob, D, R)