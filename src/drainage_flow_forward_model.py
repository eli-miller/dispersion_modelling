import os

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import VRPM_functions
import argparse
import seaborn as sns

try:
    plt.style.use("eli_default")
except:
    pass

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

#
# config_name = "drainage_flow.yaml"
# input_met_data_path = "input_met_data.csv"
# plot_domain = False
# plot_results = True
# save_path = "simulation_results.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a forward model simulation of drainage flow "
        "and output synthetic retro measurements and "
        "point sensor measurements. Run from root of the project."
    )
    parser.add_argument(
        "--config_name",
        "-c",
        type=str,
        required=True,
        help="The name of the configuration file to use",
    )
    parser.add_argument(
        "--input_met_data_path",
        "-i",
        type=str,
        default="data/input_met_data.csv",
        help="The path to the input meteorological data",
    )
    parser.add_argument(
        "--plot_domain",
        default=False,
        action="store_true",
        help="Whether to plot the domain",
    )
    parser.add_argument(
        "--plot_results",
        action="store_true",
        default=True,
        help="Whether to plot the results",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default=None,
        help="The filename to save the simulation results. Don't include path "
        "or extension as this will be used for both plot and csv files "
        "in separate locations",
    )
    parser.add_argument(
        "--resample_frequency",
        type=str,
        default=None,
        help="The frequency to resample the input met data to. Should be a higher resolution than the input data.",
        required=False,
    )
    return parser.parse_args()


args = parse_args()
config_name = args.config_name
input_met_data_path = args.input_met_data_path
plot_domain = args.plot_domain
plot_results = args.plot_results
save_name = args.save_name
resample_frequency = args.resample_frequency


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
    input_met_data_path, index_col=0, parse_dates=True
).dropna()

if resample_frequency is not None:
    # Ensure that the resample frequency is higher than the input data
    try:
        assert (
            pd.to_timedelta(resample_frequency)
            < input_met_data.index[1] - input_met_data.index[0]
        )
    except AssertionError:
        print("Resample frequency must be higher than the input data frequency")
        raise AssertionError

    input_met_data = input_met_data.resample(resample_frequency).interpolate()


if plot_domain:
    fig, axs = VRPM_functions.plot_simulation_domain(
        sources, retros, config["origin"], draw_beams=True
    )
    fig, axs = VRPM_functions.add_point_sensors_to_domain(point_sensors, axs)


simulation_results = pd.DataFrame()

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
    # TODO: is this just the row?
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
        # Determine pathlength for each retro. This would be the elegant way to implement, but have to fix the
        # retro dataframe to include the pathlength and not break the rest of the code (sloppy unpacking)
        pathlength = retros[retros["retro_names"] == retro_name]["pathlength"]
        pathlength = pathlength.iloc[0]
        single_run_df[retro_name] = pic / pathlength

    for point_sensor_name, point_sensor_measurement in zip(
        point_sensor_names, point_sensor_measurement
    ):
        single_run_df[point_sensor_name] = point_sensor_measurement

    simulation_results = pd.concat(
        [
            simulation_results,
            single_run_df,
        ],
        ignore_index=True,
    )


retros_to_plot = retros["retro_names"]

if plot_results:
    VRPM_functions.plot_simulation_results(
        simulation_results, retros_to_plot, point_sensor_names
    )

if save_name is not None:
    simulation_results.to_csv(
        os.path.join("output_data", "forward_model", save_name + ".csv")
    )
    plt.savefig(os.path.join("plots", "forward_model", save_name + ".png"))
    plt.close()
else:
    plt.close()
