import os
import subprocess

from tqdm import tqdm

input_dir = "data"
# Get a list of all files in the input directory
all_files = os.listdir(input_dir)

# Filter this list to include only .csv files
input_files = [
    file for file in all_files if file.startswith("met_data") and file.endswith(".csv")
]

# Loop over the input files
for input_file in tqdm(input_files):
    # Construct the command to run drainage_flow_forward_model.py
    stability_case = input_file.strip().split("_")[-1].split(".")[0]

    command = (
        f"python src/drainage_flow_forward_model.py "
        f"--config_name configs/drainage_flow.yaml "
        f"--input_met_data_path {os.path.join(input_dir, input_file)} "
        f"--save_name stability_case_{stability_case} "
    )

    # Use subprocess to run the command
    subprocess.run(command, shell=True)
