import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit

# Define the theoretical variogram model functions
def spherical_model(h, nugget, sill, range_):
    return nugget + (sill - nugget) * np.piecewise(h, [h <= range_, h > range_],
                                                   [lambda h: 1.5 * (h / range_) - 0.5 * (h / range_)**3, 1])

def exponential_model(h, nugget, sill, range_):
    return nugget + (sill - nugget) * (1 - np.exp(-h / range_))

def gaussian_model(h, nugget, sill, range_):
    return nugget + (sill - nugget) * (1 - np.exp(-(h**2) / (range_**2)))

# Load the data
file_path = 'pm25points.csv'  # Assuming the file is in the same directory as the script
df = pd.read_csv(file_path)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Ensure no duplicates or identical points that could lead to zero distances
df = df.drop_duplicates(subset=['Longitude', 'Latitude'])

# Extract coordinates and values
coords = df[['Longitude', 'Latitude']].values
values = df['F1ST_MAX_24_HR'].values

# Calculate the pairwise distances and differences
distances = squareform(pdist(coords))
differences = squareform(pdist(values[:, None]))

# Function to estimate initial parameters for the variogram model
def estimate_initial_parameters(exp_variogram, bin_center):
    sill_estimate = np.max(exp_variogram) * 0.9
    range_estimate = bin_center[np.argmax(exp_variogram >= sill_estimate)] if np.any(exp_variogram >= sill_estimate) else bin_center[-1]
    return (0, sill_estimate, range_estimate)

# Since the number of points is small, use fewer bins for the variogram
num_bins = len(df)  # Using half the number of points to ensure we have enough data in each bin
bins = np.linspace(0, distances.max(), num_bins)
bin_center = 0.5 * (bins[:-1] + bins[1:])

# Calculate the experimental variogram
exp_variogram = [np.nanmean(differences[(distances >= bins[i]) & (distances < bins[i+1])])
                 for i in range(len(bins)-1)]

# Convert the list to a numpy array and ensure there are no NaNs
exp_variogram = np.array(exp_variogram)
finite_indices = np.isfinite(exp_variogram)
exp_variogram = exp_variogram[finite_indices]
bin_center = bin_center[finite_indices]

# Estimate initial parameters using the finite values of the experimental variogram
initial_params = estimate_initial_parameters(exp_variogram, bin_center)

# Fit the models to the experimental variogram using the initial parameter guesses
params_spherical, _ = curve_fit(spherical_model, bin_center, exp_variogram, p0=initial_params, maxfev=10000)
params_exponential, _ = curve_fit(exponential_model, bin_center, exp_variogram, p0=initial_params, maxfev=10000)
params_gaussian, _ = curve_fit(gaussian_model, bin_center, exp_variogram, p0=initial_params, maxfev=10000)

# Plot the experimental variogram and the models
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
