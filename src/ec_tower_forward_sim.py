# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from matplotlib import style

from NNA_data_analysis import nna_tools
from VRPM_functions import *
import seaborn as sns
import warnings
import numpy as np
warnings.filterwarnings('ignore', category=RuntimeWarning)

style.use('eli_default')

ec_path = os.path.join(os.path.expanduser("~"),
                       "UCB-O365",
                       "Precision Laser Diagnostics Lab-Data sharing - NNA",
                       "ec_data_July.csv",
                       )
ec_data = nna_tools.create_ec_dataframe(ec_path)

ec_data['stability'] = ec_data.L.apply(l_to_PG)
# %%
Q_source = 62  # g/m^2/year  ELder paper 7983.69 mg m-2 d-1 but then sees 10s of ppm at .5m
BTL_AREA = 600  # m^2
# convert to kg/s
Q_source = Q_source * BTL_AREA / (1000 * 365 * 24 * 60 * 60) # kg/s
# Artifically increase Q_source to match anecdotally observed concentrations
Q_source *= 10

# sources_x, sources_y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
sources_x, sources_y = [np.array(0),np.array(0)]
sources = pd.DataFrame({'x': sources_x.ravel(),
                        'y': sources_y.ravel(),
                        'z': 0,
                        'strength': Q_source / len(sources_x.ravel())})

# Create array of receptors
thetas = np.linspace(0, 2 * np.pi, 50)
rs = np.linspace(1, 50, 20)
Rs, Thetas = np.meshgrid(rs, thetas)
Xs = Rs * np.cos(Thetas)
Ys = Rs * np.sin(Thetas)

Zs = 2 + np.zeros_like(Xs)

receptors = np.array([Xs.ravel(), Ys.ravel(), Zs.ravel()]).T

u_mag = 1
u_dir = 0
stability = 'A'


params = ec_data[['wind_speed', 'stability', 'wind_dir', 'ch4_flux', 'ch4_mixing_ratio']].dropna()
params = params.query('ch4_flux > 0')
params['flux_kg_s'] = params.ch4_flux * BTL_AREA * 1e-6 / 1000 * BTL_AREA

# params = params.iloc[0:500]
# params = params.query('stability != "F"')
#%%

test = []

for i in tqdm.tqdm(range(len(params))):
# for i in tqdm.tqdm(range(100)):
    point_measurement = np.zeros_like(Xs.ravel())
    for i_source in range(len(sources)):
        point_measurement += general_concentration(
            receptors,
            sources[['x', 'y', 'z']].iloc[i_source],
            Q=params['flux_kg_s'].iloc[i],
            u_mag=ec_data.wind_speed.iloc[i],
            u_dir=ec_data.wind_dir.iloc[i],
            stability=stability)
    # params['measurements'].iloc[i] = point_measurement
    test.append(point_measurement)

test = np.array(test)

#%%
plt.figure()

background = params.ch4_mixing_ratio.iloc[0:len(test)].min() # subtract off the background concentration (lowest value)
params['enhancement'] = params.ch4_mixing_ratio - background

# Out of the 100 potential receptors, determine the closest matching for each measurement
idx_best_measurement = np.argmin(np.abs(test - np.array(params.enhancement).reshape(len(params), 1)), axis=1)
# turn this index into the actual measurement
best_measurement = test[np.arange(0, len(test)), idx_best_measurement]




# plt.plot(test, alpha=.01, color='C1')
# plt.plot(test.mean(axis=1), color='C1', label='mean measurement')
# plt.plot(test.max(axis=1), color='C1', linestyle='--', label='max measurement')
# plt.plot(test.min(axis=1), color='C1', linestyle=':', label='min measurement')
plt.plot(params.index, best_measurement, color='C1', label='best measurement')
# plt.plot(second_best_measurement, color='C4', label='second best measurement')

test_nonzero = test.copy()
test_nonzero[test_nonzero == 0] = np.nan
# plt.plot(np.nanmean(test_nonzero, axis=1), color='C2', label='mean measurement (nonzero)')
plt.plot(params.index, params.enhancement, color='C0', label='EC Data')
plt.legend()
# Turn x axis labels 45 degrees
plt.xticks(rotation=45)
plt.ylabel('CH4 Enhancement (ppm)')



# Add inset between 5/1/23 and 5/8/23
ax = plt.gca()
axins = ax.inset_axes([.6, .6, .35, .35])
axins.plot(params.index, best_measurement, color='C1', label='best measurement')
axins.plot(params.index, params.enhancement, color='C0', label='EC Data')
axins.set_xlim(pd.datetime(2023, 5, 1), pd.datetime(2023, 5, 8))
# Indicate the inset location with a rectangle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# turn inset x axis labels 45 degrees
axins.tick_params(axis='x', rotation=45)

#%%
plt.figure()
plt.scatter(params.enhancement, best_measurement, alpha=.1)
plt.xlabel('EC Data')
plt.ylabel('Best Model Prediction')
# Add 1-1 line for reference
plt.plot([0, params.enhancement.max()], [0, params.enhancement.max()], color='black', linestyle='--')
plt.axis('square')
plt.title('Model Performance')

#%%
fig, axs = plt.subplots(2,2, sharey='row', gridspec_kw={'width_ratios': [3,1]})

# plot the R and Theta value of the best measurement
axs[0,0].plot(params.index, Thetas.ravel()[idx_best_measurement], '-o', alpha=.1)
axs[1,0].plot(params.index, Rs.ravel()[idx_best_measurement], '-o', alpha=.1)

# Add histograms to the right of the plots
axs[0,1].hist(Thetas.ravel()[idx_best_measurement], bins=len(thetas), orientation='horizontal')
axs[1,1].hist(Rs.ravel()[idx_best_measurement], bins=len(rs), orientation='horizontal')

axs[0,0].set_ylabel('Theta (rad)')
axs[1,0].set_ylabel('R (m)')
fig.suptitle("Locations of Best Measurements")
axs[1,1].set_xlabel('Count')

# remove x axis labels from [0,0] and turn [1,0] x axis labels 45 degrees
axs[0,0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axs[1,0].tick_params(axis='x', rotation=45)

#%%

# Make schematic of the sources and receptors.
plt.figure()
plt.scatter(receptors[:,0], receptors[:,1], color='C0', label='Receptor')
plt.scatter(sources.x, sources.y, color='C1', label='Source')
plt.legend()
plt.axis('equal')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Schematic of Sources and Receptors')

#%%
test = np.nan_to_num(test)
data = np.array(params.enhancement)

from scipy.optimize import nnls

coefficients, residuals = nnls(test, data)
best_combination = np.dot(test, coefficients)

#%%

# Regularization strength (adjust as needed)
alpha = 0.1

# Define the objective function for Lasso regression
def objective_function(coefficients):
    predicted_data = np.dot(test, coefficients)
    l1_regularization = alpha * np.sum(np.abs(coefficients))
    mse = np.mean((predicted_data - data) ** 2)
    return mse + l1_regularization

# Constraints: coefficients must be non-negative
constraints = [{'type': 'ineq', 'fun': lambda x: x}]

# Initial guess for coefficients (all zeros)
initial_guess = np.zeros(test.shape[1])

# Perform the optimization
result = minimize(objective_function, initial_guess, constraints=constraints)

# Get the optimized coefficients
coefficients = result.x

print("Coefficients:")
print(coefficients)

# Get the best linear combination using the computed coefficients
best_combination = np.dot(test, coefficients)

print("Best Linear Combination:")
print(best_combination)


#%%
plt.plot(params.index, best_combination, color='C1', label='best combination')
plt.legend()
#%%
import seaborn as sns

# ... (Previous code to define data, test, and other variables)

# Create a DataFrame to hold the best measurement values and indices
best_measurement_df = pd.DataFrame({'stability': params.stability,
                                    'theta': Thetas.ravel()[idx_best_measurement],
                                    'r': Rs.ravel()[idx_best_measurement]})

# Create a figure and axes using Seaborn's scatterplot and histplot functions
fig, axs = plt.subplots(2, 2, gridspec_kw={'width_ratios': [3, 1]})

# Plot Theta and R values of the best measurement with hue set to 'stability'
sns.scatterplot(data=best_measurement_df, x=params.index, y='theta', hue='stability', ax=axs[0, 0], alpha=0.1)
sns.scatterplot(data=best_measurement_df, x=params.index, y='r', hue='stability', ax=axs[1, 0], alpha=0.1)

# Add histograms to the right of the plots with hue set to 'stability'
sns.histplot(data=best_measurement_df, y='theta', bins=len(thetas), hue='stability', ax=axs[0, 1], multiple='stack', stat='density')
sns.histplot(data=best_measurement_df, y='r', bins=len(rs), hue='stability', ax=axs[1, 1], multiple='stack', stat='density', legend=False)

axs[0, 0].set_ylabel('Theta (rad)')
axs[1, 0].set_ylabel('R (m)')
fig.suptitle("Locations of Best Measurements")
axs[1, 1].set_xlabel('Count')

# Remove x-axis labels from [0, 0] and turn [1, 0] x-axis labels 45 degrees
axs[0, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
axs[1, 0].tick_params(axis='x', rotation=45)

plt.show()
