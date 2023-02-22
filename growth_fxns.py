import pandas as pd
import numpy as np
import random

def flatten_matrix(matrix):
    out = []
    for sublist in matrix:
        for val in sublist:
            out.append(val)
    return(out)

def gillespie_growth(init_cell_count=200, div_rate=0.04, death_rate=0.001, 
                     max_time=126, output_time_step=6, seed=0):
    if seed != 0:
        random.seed(seed)
    else:
        random.random()

    X = [init_cell_count]
    t = [0]
    while t[-1] < max_time:

        current_X = X[-1]
        
        rates = [div_rate * current_X, death_rate * current_X]
        rate_sum = sum(rates)

        tau = np.random.exponential(scale=1/rate_sum)

        t.append(t[-1] + tau)

        rand = random.uniform(0,1)

        # production event
        if rand * rate_sum > 0 and rand * rate_sum <= rates[0]:
                X.append(X[-1] + 1)

        # decay event
        elif rand * rate_sum > rates[0] and rand * rate_sum <= rates[0] + rates[1]:
                X.append(X[-1] - 1)
  
    # Gillespie simulations generate many time points
    # subsample times closest to time intervals based on output_time_step
    source = np.array(t)
    reference = np.arange(0,126,output_time_step)
    dt = [abs(reference - i) for i in source]
    dt = pd.DataFrame(dt)
    i = dt.idxmin()
    times = pd.Series(t)
    times = times.iloc[i]
    cell_count = pd.Series(X)
    cell_count = cell_count.iloc[i]
    out = {'times':times,'cell_count':cell_count}
    out = pd.DataFrame.from_dict(out)
    return(out)

def exp_growth(x, P0=1, rate=0.04, log2=True):
    if log2:
        return(round(P0*np.exp(x*rate)/np.exp(2),4))
    else:
        return(round(P0*np.exp(x*rate),4))


def mylogistic(t, P0=100, rate=0.04, K=1000):
    """
    Logistic equation with parameters:
    t = time
    P0 = initial size of population
    rate = population growth rate
    K = carrying capacity
    NOTE: computing exponent using NumPy
    """
    return(K*P0*np.exp(rate*t) / (K + P0*(np.exp(rate*t)-1)))


def gompertz(t, P0=100, rate=0.04, K=1000):
    """
    Gompertz equation with parameters:
    t = time
    P0 = initial size of population
    rate = population growth rate
    K = carrying capacity
    NOTE: computing exponent using NumPy
    """
    return(K*np.exp( np.log(P0/K)*np.exp(-rate*t)))

