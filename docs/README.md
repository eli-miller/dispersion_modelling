# dispersion_modelling
Collection of code to model dispersion of methane in the Goldstream Valley

### Description
`VRPM_functions.py` contains functions called to simulate long-path laser measurements within a gaussian plume.
`stacked_retro_analysis.py` is an example file (in progress) which uses these functions to simulate measurements. 

#### Examples
```sources, retros = read_measurement_geometry('geometry_config.yaml')```
reads in the measurement geometry from a yaml file. The yaml file should be formatted as follows:
```Q_source_total: 620 # 62  # g/m^2/year 10x to get ground level conc. to match
source_area: 600  # m^2

n_source_x: 1 #10
n_source_y: 1 #10
span_source_x: [0,0] #[-5, 5]
span_source_y: [0,0] #[-5, 5]

source_loc_y: 2500
source_loc_x: 0
source_loc_z: 0 # Keep zero

n_retro_x: 30
n_retro_y: 10
n_retro_z: 5

span_retro_x: [-5, 100]
span_retro_y: [-25, 25]
span_retro_z: [0, 3]

retro_loc_x : 0
retro_loc_y : 2500
```
