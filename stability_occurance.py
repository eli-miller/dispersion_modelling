## Plot heatmap of conditions that we expect to experience using ec_data.
# Add to path to point to create_ec_data function within nna_tools
import sys

import numpy as np

sys.path.append('/Users/elimiller/Documents/1Research/python_project/NNA_data_analysis/')
from nna_tools import create_ec_dataframe
import VRPM_functions
import os
import matplotlib.pyplot as plt
import seaborn as sns

ec_data = create_ec_dataframe(os.path.join(os.getcwd(), '*.txt'))
ec_data['stability'] = ec_data['L'].apply(VRPM_functions.l_to_PG)
ec_data['wind_speed_rounded'] = np.round(ec_data['wind_speed'] * 2) / 2

heatmap_data = ec_data.pivot_table(index='stability', columns='wind_speed_rounded', aggfunc='size', fill_value=0)

# sns.heatmap(data=heatmap_data)

plt.figure()
sns.jointplot(data=ec_data, x='wind_speed', y='(z-d)/L', hue='stability')
