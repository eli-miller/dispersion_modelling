#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:52:40 2022

@author: elimiller
"""
import importlib
import warnings
import itertools as it
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm

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
    plot_contours,
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

# plot_simulation_domain(sources, retros, config["origin"], draw_beams=True)
# %%
z_vals = config["span_retro_z"]
wind_speeds = config["wind_speeds"]
stabilitys = config["stability_classes"]
wind_directions = config["wind_directions"]
source_strengths = config["source_strengths"]

wind_directions = [VRPM_functions.convert_wind_direction(wd) for wd in wind_directions]


contour_df = pd.DataFrame()

combinations = it.product(wind_directions, wind_speeds, stabilitys, source_strengths)
total_iterations = (
    len(wind_directions) * len(wind_speeds) * len(stabilitys) * len(source_strengths)
)


for wind_direction, wind_speed, stability, source_strength in tqdm(
    combinations, total=total_iterations
):
    # First thing is to get the measurement field and overall metrics at each retro
    summary = simulate_stacked_retros(
        wind_speed,
        stability,
        u_dir=wind_direction,
        Q=source_strength,
        plot=False,
        config_file=config_name,
    )

    # Then we move into more "detection limit" type analysis.

    meas_field, x, y = get_meas_field(summary, z_vals[0])
    # TODO: expand to allow for multiple contour levels?
    contours = get_contour_coordinates(x, y, meas_field, contour_level=3e-3)

    contour_df = pd.concat(
        [
            contour_df,
            pd.DataFrame(
                {
                    "stability": stability,
                    "wind_speed": wind_speed,
                    "wind_direction": wind_direction,
                    "source_strength": source_strength,
                    "contours": [contours.tolist()],
                }
            ),
        ]
    )

    # Add the max contour locations to the dataframe after the fact. (Allows plotting earlier to find bugs)

    # This is how it was before: x_max_coords, y_max_coords = get_max_contour_locations(contours)
    # Let's use a .apply or something
contour_df.to_csv("detection_limit.csv", index=False)
# %%
plot_contours(contour_df, group_by=["stability", "wind_direction"])

# %%
# Remove D and F stability classes
# contour_df = contour_df[~contour_df["stability"].isin(["E", "F"])]

# contour_df = pd.read_csv("detection_limit_store.csv")


def safe_get_max_contour_locations(contour):
    try:
        return get_max_contour_locations(contour)
    except:
        return ((0, 0), (0, 0))


contour_df["max_coords"] = contour_df["contours"].apply(safe_get_max_contour_locations)

contour_df["x_max"] = contour_df["max_coords"].apply(lambda x: x[0][0])
contour_df["y_max"] = contour_df["max_coords"].apply(lambda x: x[1][1])
contour_df["y_at_x_max"] = contour_df["max_coords"].apply(lambda x: x[0][1])
contour_df["x_at_y_max"] = contour_df["max_coords"].apply(lambda x: x[1][0])

contour_df.y_max -= 2500  # shift relative to source
# %%


for field in ["x_max", "y_max"]:
    plt.figure()
    sns.scatterplot(
        data=contour_df,
        x="wind_speed",
        y=field,
        hue="stability",
        style="wind_direction",
        s=100,
    )

# Plot the effect of source strength
plt.figure()
