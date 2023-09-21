#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:52:40 2022

@author: elimiller
"""
import importlib
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt

# from VRPM_functions import *  # TODO: Fix this import
import VRPM_functions
import seaborn as sns
importlib.reload(VRPM_functions)
from VRPM_functions import create_measurement_geometry, read_measurement_geometry, simulate_stacked_retros, get_meas_field, get_contour_coordinates

# %%
# Set up options
plt.close('all')
matplotlib.use('macosx')
matplotlib.style.use('eli_default')
matplotlib.rcParams.update({'font.size': 16})
warnings.filterwarnings("ignore")

config = read_measurement_geometry('geometry_config.yaml')
sources, retros = create_measurement_geometry(config)

z_vals = config['span_retro_z']
stabilitys = ['A', 'B', 'C', 'D', 'E', 'F']
us = [1.5, 2.5, 3.5, 4.5, 5.5]

# stabilitys = ['A', 'F']
# stabilitys = ['C']
us = [1.5]
summary_store = []

fig, ax = plt.subplots(figsize=(8, 6))
for stability in stabilitys:
    for u in us:
        summary = simulate_stacked_retros(u, stability, plot=False, config_file='geometry_config.yaml')
        meas_field, x, y = get_meas_field(summary, z_vals[0])
        contours = get_contour_coordinates(x, y, meas_field, contour_level=3e-3)

        # Plot the contours
        ax.plot(contours[:, 0], contours[:, 1] - config['source_loc_y'], label=f'{stability}, {u} [m/s]', linestyle='--')

    ax.legend()



# summary_store.append(summary)

# save_fig_dir = '/Users/elimiller/Documents/1Research/python_project/dispersion_modelling/plots/stacked_retro_plots'
# save_fig_name = 'delta_z_%s_%.1f mps.png' % (stability, u)
# summary_store = pd.concat(summary_store)
# %%
# sns.scatterplot(data=summary_store, x='measurement', y='max_local_concentration', label=stability, marker='o')


# %%


# plt.figure()
# sns.boxplot(data=summary, x='z', hue='y', y='measurement')
# plt.figure()
# sns.boxplot(data=summary, x='z', hue='y', y='max_local_concentration')
#
# # %%
# # lot curves of concentration vs z for each retro. Use groupby to plot each x,y, pair as a line
# plt.figure()
# sns.boxplot(x='z', y='measurement', data=retros)
#
# # %%
# sns.lineplot(data=summary, x='x', y='measurement', hue='z', estimator=np.max, ci=None)
# sns.lineplot(data=summary, x='x', y='max_local_concentration', hue='z', estimator=np.max, ci=None, linestyle=':')
# sns.lineplot(data=summary, x='x', y='measurement', hue='z', estimator=np.mean, ci=None, linestyle='--')
