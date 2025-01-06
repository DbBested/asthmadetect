import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit


def spherical_model(h, nugget, sill, range_):
    return nugget + (sill - nugget) * np.piecewise(h, [h <= range_, h > range_],
                                                   [lambda h: 1.5 * (h / range_) - 0.5 * (h / range_)**3, 1])

def exponential_model(h, nugget, sill, range_):
    return nugget + (sill - nugget) * (1 - np.exp(-h / range_))

def gaussian_model(h, nugget, sill, range_):
    return nugget + (sill - nugget) * (1 - np.exp(-(h**2) / (range_**2)))


file_path = r"C:\Users\dbbes\OneDrive\Documents\Workspace\asthma\asthma_points_withpos.csv"
df = pd.read_csv(file_path)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df = df.drop_duplicates(subset=['Longitude_Last', 'Latitude_Last'])


coords = df[['Longitude_Last', 'Latitude_Last']].values
values = df['prevalence_int'].values

distances = squareform(pdist(coords))
differences = squareform(pdist(values[:, None]))

num_bins = min(len(df), 500)


bins = np.linspace(20000, distances.max(), num_bins) 
bin_center = 0.5 * (bins[1:] + bins[:-1])


exp_variogram = []
for i in range(len(bins)-1):
    bin_differences = differences[(distances >= bins[i]) & (distances < bins[i+1])]
    if len(bin_differences) > 0:
        exp_variogram.append(np.nanmean(bin_differences))
    else:
        exp_variogram.append(np.nan)

exp_variogram = np.array(exp_variogram)
finite_indices = np.isfinite(exp_variogram)
exp_variogram = exp_variogram[finite_indices]
bin_center = bin_center[finite_indices]

def robust_initial_parameters(exp_variogram, bin_center):
    nugget_estimate = np.min(exp_variogram)
    sill_estimate = np.percentile(exp_variogram, 75)
    range_estimate = bin_center[np.argmax(exp_variogram >= sill_estimate)] if np.any(exp_variogram >= sill_estimate) else bin_center[-1]
    return nugget_estimate, sill_estimate, range_estimate

initial_params = robust_initial_parameters(exp_variogram, bin_center)
bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

params_spherical, _ = curve_fit(spherical_model, bin_center, exp_variogram, p0=initial_params, bounds=bounds, maxfev=10000)
params_exponential, _ = curve_fit(exponential_model, bin_center, exp_variogram, p0=initial_params, bounds=bounds, maxfev=10000)
initial_params_gaussian = list(initial_params)
params_gaussian, _ = curve_fit(gaussian_model, bin_center, exp_variogram, p0=initial_params_gaussian, bounds=bounds, maxfev=10000)

plt.figure(figsize=(10, 6))
plt.plot(bin_center, exp_variogram, 'o', label='Experimental Variogram')
plt.plot(bin_center, spherical_model(bin_center, *params_spherical), label='Spherical Model', color='orange')
plt.plot(bin_center, exponential_model(bin_center, *params_exponential), label='Exponential Model', color='green')
plt.plot(bin_center, gaussian_model(bin_center, *params_gaussian), label='Gaussian Model', color='red')
plt.xlabel('Distance')
plt.ylabel('Semivariance')
plt.title('Variogram with Multiple Models')
plt.legend()
plt.show()
