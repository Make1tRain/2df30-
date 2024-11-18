from scipy import stats
from numpy import cumsum, append, zeros, mean
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW

# The Cramer-Lundberg model (The compound Poisson risk model)

lam = 0.5             # Rate of the Poisson process
meanClaimSize = 1    # mean claim size
theta = 0.1          # safety loading factor
c = (1 + theta)*lam*meanClaimSize  # premium per time unit

u0 = 20              # initial capital
TimeHorizon = 1000             # Time horizon
N = round(TimeHorizon*lam)

#simulate a sample path
def simCLModelSamplePath():
    interArrivals = stats.expon(scale=1/lam).rvs(N)
    arrivalTimes = cumsum(interArrivals)
    claimSizes = stats.expon(scale=meanClaimSize).rvs(N)
    cumulativeClaim = cumsum(claimSizes)
    levels = u0 + arrivalTimes * c - cumulativeClaim
    ruin = min(levels) < 0
    return arrivalTimes,cumulativeClaim,levels,ruin

arr,cl,lev,ruin = simCLModelSamplePath()
print(ruin)

#Plot cumulative claim size and premium level
plt.figure()
plt.step(arr,cl,'b', where='post')
plt.plot(arr,u0+c*arr,'r')
plt.show()

#Plot of capital level
def plotCapitalLevel():
    xpoints = [0]
    ypoints = [u0]
    currentLevel =0
    n=len(arr)
    for i in range(n):
        t = arr[i]
        xpoints.append(t)
        xpoints.append(t)
        ypoints.append(u0+c*t-currentLevel)
        currentLevel =cl[i]
        ypoints.append(u0+c*t-currentLevel)
    plt.figure()
    plt.plot(xpoints,ypoints,'b')
    plt.hlines(xmin=0,xmax=t,y=0,color = 'red')

plotCapitalLevel()
plt.show()


#Estimate ruin probability
# Estimate the probability on ruin
nrRuns = 5000
sim = zeros(nrRuns)
for i in range(nrRuns):
    arr,cl,lev,ruin = simCLModelSamplePath()
    sim[i] = ruin

#print(sim)
print(mean(sim))

ci = DescrStatsW(sim).tconfint_mean(alpha=0.05)
print(ci)