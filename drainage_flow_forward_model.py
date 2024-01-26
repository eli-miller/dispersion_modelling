import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import VRPM_functions
from icecream import ic
import seaborn as sns

config_name = "drainage_flow.yaml"
config = VRPM_functions.read_measurement_geometry(config_name)
sources, retros, point_sensors = VRPM_functions.create_measurement_geometry(
    config, simulation_type="real"
)


retro_names = config["retro_names"]
point_sensor_names = config["point_sensor_names"]


input_met_data = pd.read_csv(
    "input_met_data.csv", index_col=0, parse_dates=True
).dropna()

# Replace input_met_data with a synthetic dataset. Sweep through wind speed, wind direction, and stability
input_met_data = pd.DataFrame()
input_met_data["wind_dir"] = np.linspace(0, 360, 6)
input_met_data["wind_speed"] = np.ones_like(input_met_data["wind_dir"])
input_met_data["stability"] = "C"
input_met_data["datetime"] = pd.date_range(
    start="2024-01-01 00:00:00", periods=len(input_met_data), freq="H"
)

fig, axs = VRPM_functions.plot_simulation_domain(
    sources, retros, config["origin"], draw_beams=True
)

fig, axs = VRPM_functions.add_point_sensors_to_domain(point_sensors, axs)
plt.figure()
# For each entry in the input_met_data, run the simulation and save the output
# to a csv file that contains inputs and results for each retro and point sensor measurement

# %%
# Create a dataframe to store the results
simulation_results = pd.DataFrame()

# Loop over the input_met_data
for index, row in input_met_data.iterrows():
    converted_wind_directon = VRPM_functions.convert_wind_direction(row["wind_dir"])

    pics, sim_info = VRPM_functions.get_synthetic_measurement(
        retros=retros,
        sources=sources,
        origin=config["origin"],
        u_mag=row["wind_speed"],
        u_dir=converted_wind_directon,
        stability=row["stability"],
        plot=True,
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
    for retro_name, pic in zip(retro_names, pics):
        # Use scientific notation for the pic
        single_run_df[retro_name] = pic

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

# Plot the point measurement vs wind direction

plt.figure()
[
    sns.scatterplot(data=simulation_results, x="wind_dir", y=point_sensor_name)
    for point_sensor_name in point_sensor_names
]


# %%
# Compute the cartesian distance between each retro and the source and print
# Add retro names to the retros df
retros["name"] = retro_names

for retro_name in retro_names:
    retro_loc = retros[retros.name == retro_name][["x", "y", "z"]].iloc[0]
    source_loc = sources[["x", "y", "z"]].iloc[0]
    distance = np.linalg.norm(retro_loc - source_loc)
    print(f"Distance between {retro_name} and source: {distance} m")

# Add point sensor names to the point_sensors df
point_sensors["name"] = point_sensor_names
# Repeat for the point sensors
for point_sensor_name in point_sensor_names:
    point_sensor_loc = point_sensors[point_sensors.name == point_sensor_name][
        ["x", "y", "z"]
    ].iloc[0]
    source_loc = sources[["x", "y", "z"]].iloc[0]
    distance = np.linalg.norm(point_sensor_loc - source_loc)
    print(f"Distance between {point_sensor_name} and source: {distance} m")
