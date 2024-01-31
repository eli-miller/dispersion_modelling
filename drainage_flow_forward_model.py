import pandas as pd
import matplotlib.pyplot as plt
import VRPM_functions
import seaborn as sns


plt.style.use("eli_default")

# TODO: Restructure to make the "retros" df more full-featured.
#  Currently, there is a lot of fragile unpacking in the VRPM_functions that
#  Should be avoided.
#  c. Fix unpacking in functions to be more robust
#  a. Add a column for the retro name
#  b. Add a column for the pathlenth
#  d. Reach: Use the deployment file to do all of this automatically


# TODO: parallelize calculations in VRPM_functions.get_synthetic_measurement

# TODO: rethink setup.  Can we reduce down to much less data strucutres and variables?
#  Can / should we be using xarray?


config_name = "drainage_flow.yaml"
config = VRPM_functions.read_measurement_geometry(config_name)
sources, retros, point_sensors = VRPM_functions.create_measurement_geometry(
    config, simulation_type="real"
)
sources = VRPM_functions.assign_source_strength(
    sources,
    total_source_strength=config["Q_source_total"],
    source_area=config["source_area"],
)

retros_for_simulation = (
    retros.copy()
)  # TODO: This is a hack to get around the fact that the retros df is not full-featured


retros["retro_names"] = config["retro_names"]
retros["pathlength"] = VRPM_functions.calculate_pathlength(retros, config["origin"])
point_sensor_names = config["point_sensor_names"]
origin = config["origin"]

input_met_data = pd.read_csv(
    "input_met_data.csv", index_col=0, parse_dates=True
).dropna()

# # Interpolate input_met_data onto a 5 minute interval. For stability, use the
# # previous stability value
# input_met_data = input_met_data.resample("5T").interpolate(method="linear")
# input_met_data["stability"] = input_met_data["stability"].fillna(
#     method="ffill"
# )  # fill in the first value with the first value

fig, axs = VRPM_functions.plot_simulation_domain(
    sources, retros, config["origin"], draw_beams=True
)

fig, axs = VRPM_functions.add_point_sensors_to_domain(point_sensors, axs)
# For each entry in the input_met_data, run the simulation and save the output
# to a csv file that contains inputs and results for each retro and point sensor measurement

# %%
# Create a dataframe to store the results
simulation_results = pd.DataFrame()

# Loop over the input_met_data
for index, row in input_met_data.iterrows():
    converted_wind_directon = VRPM_functions.convert_wind_direction(row["wind_dir"])
    # Note PIC is returned as ppm*m.  Need to normalize by pathlength
    pics, sim_info = VRPM_functions.get_synthetic_measurement(
        retros=retros_for_simulation,
        sources=sources,
        origin=config["origin"],
        u_mag=row["wind_speed"],
        u_dir=converted_wind_directon,
        stability=row["stability"],
        plot=False,
    )

    # Get the measurements at each point sensor using general_concentration

    point_sensor_measurement = VRPM_functions.general_concentration(
        beam_loc=point_sensors,
        source_loc=list(
            sources[["x", "y", "z"]].iloc[0]
        ),  # Assumption that there is one source
        Q=sources.strength.iloc[0],  # Assumption that there is one source
        u_mag=row["wind_speed"],
        u_dir=converted_wind_directon,
        stability=row["stability"],
    )
    single_run_df = pd.DataFrame(
        {
            "datetime": [index],
            "wind_speed": [row["wind_speed"]],
            "wind_dir": [row["wind_dir"]],
            "stability": [row["stability"]],
            "simulation_info": [sim_info],
            # 'point_sensor_measurements': point_sensor_measurements}
        }
    )

    # add the pics to the dataframe using the retro names from the config file
    for retro_name, pic in zip(retros["retro_names"], pics):
        # # Determine pathlength for each retro. This would be the elegant way to implement, but have to fix the
        # # retro dataframe to include the pathlength and not break the rest of the code (sloppy unpacking)
        pathlength = retros[retros["retro_names"] == retro_name]["pathlength"]
        pathlength = pathlength.iloc[0]
        single_run_df[retro_name] = pic / pathlength

    # add the point sensor measurements to the dataframe using the point sensor names from the config file
    for point_sensor_name, point_sensor_measurement in zip(
        point_sensor_names, point_sensor_measurement
    ):
        single_run_df[point_sensor_name] = point_sensor_measurement
    # Add the measurements to the dataframe. unpack pics with the names from the config file
    simulation_results = pd.concat(
        [
            simulation_results,
            single_run_df,
        ],
        ignore_index=True,
    )


# %%
def plot_simulation_results(simulation_results, retro_names, point_sensor_names):
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(9, 9),
        sharex="col",
        sharey="row",
        gridspec_kw={"width_ratios": [2, 1], "height_ratios": [2, 1]},
    )

    # Plot the retro measurements vs time
    for retro_name in retro_names:
        sns.lineplot(
            data=simulation_results,
            x="datetime",
            y=retro_name,
            label=retro_name,
            ax=axs[0, 0],
            # alpha=0.5,
        )

        sns.scatterplot(
            data=simulation_results,
            x="wind_dir",
            y=retro_name,
            # label=retro_name,
            ax=axs[0, 1],
            size="wind_speed",
        )

    # Plot the point sensor measurements vs time
    for point_sensor_name in point_sensor_names:
        sns.lineplot(
            data=simulation_results,
            x="datetime",
            y=point_sensor_name,
            label=point_sensor_name,
            ax=axs[1, 0],
            # alpha=0.5,
        )

        sns.scatterplot(
            data=simulation_results,
            x="wind_dir",
            y=point_sensor_name,
            # label=point_sensor_name,
            ax=axs[1, 1],
            size="wind_speed",
        )

    axs[0, 0].set_title("Retro Measurements vs Time")
    axs[0, 1].set_title("Retro Measurements vs Wind Direction")
    axs[1, 0].set_title("Point Sensor Measurements vs Time")
    axs[1, 1].set_title("Point Sensor Measurements vs Wind Direction")

    axs[0, 0].set_ylabel("Path-averaged enhancement (ppm)")
    axs[1, 0].set_ylabel("Point sensor enhancement (ppm)")

    plt.tight_layout()
    plt.show()

    # Turn x axis labels 45 degrees for axs[1,0]
    for tick in axs[1, 0].get_xticklabels():
        tick.set_rotation(45)


retros_to_plot = retros["retro_names"]

# remove any retros or point sensors that begin with "NorthEastBracket"
retros_to_plot = [
    retro for retro in retros_to_plot if not retro.startswith("NorthEastBracket")
]
point_sensor_names = [
    point_sensor_name
    for point_sensor_name in point_sensor_names
    if not point_sensor_name.startswith("NorthEastBracket")
]

plot_simulation_results(simulation_results, retros_to_plot, point_sensor_names)
