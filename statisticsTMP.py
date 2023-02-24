import math
import numpy as np
import scipy.stats as stats

# Basic Statistics Functions
def mean(values):
    return sum(values) / len(values)

def median(values):
    n = len(values)
    if n % 2 == 0:
        return (values[n//2-1] + values[n//2]) / 2
    else:
        return values[n//2]

def mode(values):
    return max(set(values), key = values.count)

def variance(values):
    return np.var(values)

def standard_deviation(values):
    return np.std(values)

def covariance(values_x, values_y):
    return np.cov(values_x, values_y)[0][1]

def correlation(values_x, values_y):
    return np.corrcoef(values_x, values_y)[0][1]

# Hypothesis Testing Functions
def z_test(sample_mean, population_mean, sample_std, sample_size):
    z = (sample_mean - population_mean) / (sample_std / math.sqrt(sample_size))
    p_value = stats.norm.sf(abs(z))*2
    return z, p_value

def t_test(sample_mean, population_mean, sample_std, sample_size, degree_of_freedom):
    t = (sample_mean - population_mean) / (sample_std / math.sqrt(sample_size))
    p_value = stats.t.sf(np.abs(t), degree_of_freedom)*2
    return t, p_value

def chi_square_test(observed, expected):
    chi_squared_stat = 0
    for i in range(len(observed)):
        numerator = ((observed[i] - expected[i]) ** 2)
        denominator = expected[i]
        chi_squared_stat += numerator / denominator
    return chi_squared_stat, 1 - stats.chi2.cdf(chi_squared_stat, len(observed)-1)

# Regression Functions
def linear_regression(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return slope, intercept, r_value, p_value, std_err

# Probability Density Function and Cumulative Density Function
def pdf(x, mean, std):
    return stats.norm.pdf(x, mean, std)

def cdf(x, mean, std):
    return stats.norm.cdf(x, mean, std)
