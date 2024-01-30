#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:52:40 2022

@author: elimiller
"""
import importlib
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from VRPM_functions import *  # TODO: Fix this import
import VRPM_functions

importlib.reload(VRPM_functions)
from VRPM_functions import (
    create_measurement_geometry,
    read_measurement_geometry,
    simulate_stacked_retros,
    get_meas_field,
    get_contour_coordinates,
    plot_simulation_domain,
    get_max_contour_locations,
)

# %%
# Set up options
plt.close("all")
matplotlib.use("macosx")
matplotlib.style.use("eli_default")
matplotlib.rcParams.update({"font.size": 16})
warnings.filterwarnings("ignore")

config_name = "high_wind_config.yaml"
# config_name = "geometry_config_for_plots.yaml"
config = read_measurement_geometry(config_name)
sources, retros = create_measurement_geometry(config)

plot_simulation_domain(sources, retros, config["origin"], draw_beams=True)
# %%
z_vals = config["span_retro_z"]
wind_speeds = config["wind_speeds"]
stabilitys = config["stability_classes"]

wind_direction = VRPM_functions.convert_wind_direction(-90)  # coming from the east

contour_df = pd.DataFrame()


for stability in stabilitys:
    for wind_speed in wind_speeds:
        fig, ax = plt.subplots(figsize=(8, 6))
        # First thing is to get the measurement field and overall metrics at each retro
        summary = simulate_stacked_retros(
            wind_speed,
            stability,
            u_dir=wind_direction,
            plot=False,
            config_file=config_name,
        )

        # Then we move into more "detection limit" type analysis.

        meas_field, x, y = get_meas_field(summary, z_vals[0])
        # TODO: expand to allow for multiple contour levels?
        contours = get_contour_coordinates(x, y, meas_field, contour_level=3e-3)

        # Plot the contours
        ax.plot(
            contours[:, 0] - config["source_loc_x"],
            contours[:, 1] - config["source_loc_y"],  # shift to source location
            label=f"{stability}, {wind_speed} [m/s]",
            linestyle="solid",
        )

        x_max_coords, y_max_coords = get_max_contour_locations(contours)

        contour_df = pd.concat(
            [
                contour_df,
                pd.DataFrame(
                    {
                        "stability": stability,
                        "wind_speed": wind_speed,
                        "x_max": x_max_coords[0],
                        "y_max": y_max_coords[1],
                        "y_at_max_x": x_max_coords[1],
                        "x_at_max_y": y_max_coords[0],
                    }
                ),
            ]
        )

    ax.legend()
    # plt.xlim(0, 215)
    # plt.ylim(-15, 70)
    # plt.savefig(
    #     f"plots/stacked_retro_plots/z=1.5_max_flux/stacked_retro_{stability}.png",
    #     transparent=True,
    #     dpi=300,
    # )


# summary_store.append(summary)

# save_fig_dir = '/Users/elimiller/Documents/1Research/python_project/dispersion_modelling/plots/stacked_retro_plots'
# save_fig_name = 'delta_z_%s_%.1f mps.png' % (stability, wind_speed)
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
