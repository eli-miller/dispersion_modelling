import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime

# Create arrays of wind speeds and wind directions using numpy
wind_speeds = np.arange(0.5, 10.5, 0.5)
wind_directions = np.arange(0, 385, 1)
wind_sources = ["synthetic"]
stability_classes = ["A", "B", "C", "D", "E", "F"]

combinations = list(
    product(wind_speeds, wind_directions, wind_sources, stability_classes)
)

datetimes = pd.date_range(start="2024-01-01", periods=len(combinations), freq="30min")

df = pd.DataFrame(
    combinations, columns=["wind_speed", "wind_dir", "wind_source", "stability"]
)
df.insert(0, "datetime", datetimes)

df.to_csv("data/met_data_synthetic.csv", index=False)
