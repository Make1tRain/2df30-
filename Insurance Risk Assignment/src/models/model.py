import numpy as np 
import matplotlib.pyplot as plt  
import pandas as pd
import scipy.stats as stats 
import sympy as sy 
from scipy.optimize import fsolve
import time
import os

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
        # plt.step(arrivalTimes, cumulativeClaims, "b", where="post")
        # plt.plot(arrivalTimes, u0 + c * arrivalTimes, "r")
        # plt.show()

        plt.figure()
        plt.plot(arrivalTimes, levels, "b") 
        plt.plot(arrivalTimes, [0 for i in range(len(arrivalTimes))], "r")

    
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
        # NOTE: In the example, the expected value of N(t) was taken by saying time horizon * lambda however,
        # we decided to sample the number of claims as well since it is an RV.

        # return stats.poisson(mu=time_horizon * lam)
        self.numberOfClaimsDist = time_horizon * lam 
    
    def set_claim_sizes_dist(self): 
        # 1.1 - Claim sizes (Uniform distribution with a mean of 16,000 and a variance of 12,000,000)
        # 1.1.1 - Calculate the a and b of the continuous uniform distribution
        a, b, mu, var = sy.symbols("a, b, mu, sigma^2")

        eq1 = sy.Eq((a + b) / 2, mu)
        eq2 = sy.Eq(((a - b) ** 2) / 12, var)

        # by using the two equations which are the definitions of mean and variance of continuous uniform distribution,
        # we find the values of a and b
        bValue = sy.solve(eq1.subs({a: sy.solve(eq2, a)[0], mu: self.claimSize_mean, var: 12000000}))[0]
        aValue = sy.solve(eq1.subs({b: bValue, mu: self.claimSize_mean}), a)[0]

        print(f"Claim Sizes ~ U(a={aValue},b={bValue})")

        # 1.1.2 Create the claim size distribution
        claimSizeDistribution = stats.uniform(loc=aValue, scale=bValue - aValue)

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
        MaxAggLoss = u0 - minimumPoint 
        D, recoveryTime = None, None 

        """
        3D = None # TODO: Remove None's

        recoveryTime = None 

        if bool(ruin): 
            l = [i,val for i,val in enumerate(levels) if val < 0]
            ruinIndex = l[0][0]
            D = l[0][1]

            recoveryIndex = None
            for i,val in enumerate(levels)[ruinIndex:]: 
                if val >= 0: 
                    recoveryIndex = i
                    break 
            
            if ruinIndex and recoveryIndex:  
                recoveryTime = arrivalTimes[recoveryIndex] - arrivalTimes[ruinIndex]
        
        """
        
        return ruin, D, recoveryTime
        # return ruin 

    def simulate_one_pair(self, u0, theta, n):
        start = time.time()
        result = np.zeros(n)
        DList = np.zeros(n)
        RList = np.zeros(n)
        c = (1 + theta) * self.lambda_arrival * self.claimSize_mean  # self.claimSize_mean is E(X_i)

        for i in range(n):
            ruin, D, recoveryTime = self.run(u0, c)
            result[i] = 1 if ruin == True else 0
            # DList[i] = D 
            # RList[i] = recoveryTime
            print(f"[i] (u={u0},theta={theta}) Run: {i}")

        end = time.time()
        print(
            f"[i] (u={u0},theta={theta}) Time taken for {n} simulations: {end - start} seconds, this is equal to {(end - start) / n} seconds per run."
        )
        
        # ruin, D, R
        return np.mean(result), np.mean([i for i in DList if i != None]), np.mean([i for i in RList if i != None])
    
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
        import multiprocessing

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

