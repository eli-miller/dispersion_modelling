#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:52:40 2022

@author: elimiller
"""
import warnings

import cmcrameri.cm as cm
import matplotlib
import pandas as pd
import seaborn as sns

from VRPM_functions import *

# Set up options
plt.close("all")
matplotlib.use("macosx")
matplotlib.style.use("eli_default")
matplotlib.rcParams.update({"font.size": 16})
warnings.filterwarnings("ignore")

# Configs
PLOT = True
RECONSTRUCT = False

# Inputs
# Q_source = 7983.69 / 1000 / (24 * 60)  # g/m^2/day #TODO: Check per day vs per second units....
Q_source = (
    62  # g/m^2/year  ELder paper 7983.69 mg m-2 d-1 but then sees 10s of ppm at .5m
)

BTL_AREA = 600  # m^2
# convert to kg/s
Q_source = Q_source * BTL_AREA / (1000 * 365 * 24 * 60 * 60)
Q_source *= 10

retro_df = pd.read_csv("real_retro_locations.csv", skiprows=0)

beam_lengths = retro_df["r"]

sources_df = retro_df[retro_df.name == "BTL_EC"]

sources_x, sources_y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
sources = pd.DataFrame(
    {
        "x": float(sources_df.DeltaY) + sources_x.ravel(),
        "y": float(sources_df.DeltaX) + sources_y.ravel(),
        "z": 0,
        "strength": Q_source / len(sources_x.ravel()),
    }
)

# Manually take out variations in AGL
retros_real = pd.DataFrame(
    {"x": retro_df["DeltaY"], "y": retro_df["DeltaX"], "z": retro_df["DeltaZ"]}
)

# Create synthetic retros
x_synthetic = np.linspace(-1, 100, 10)
y_synthetic = np.linspace(-10, 10, 5)
y_synthetic = np.array([0, 20], dtype="float")

X_synthetic, Y_synthetic = np.meshgrid(x_synthetic, y_synthetic)
Z_synthetic = 1.5 + np.zeros_like(X_synthetic)

X_synthetic += float(sources_df.DeltaY)
Y_synthetic += float(sources_df.DeltaX)

retros_synthetic = pd.DataFrame(
    {"x": X_synthetic.ravel(), "y": Y_synthetic.ravel(), "z": Z_synthetic.ravel()}
)

# retros = pd.concat([retros_real, retros_synthetic])
retros = retro  #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:52:40 2022

@author: elimiller
"""
import warnings

import cmcrameri.cm as cm
import matplotlib
import pandas as pd
import seaborn as sns

from VRPM_functions import *

# Set up options
plt.close("all")
matplotlib.use("macosx")
matplotlib.style.use("eli_default")
matplotlib.rcParams.update({"font.size": 16})
warnings.filterwarnings("ignore")

# Configs
PLOT = True
RECONSTRUCT = False

# Inputs
# Q_source = 7983.69 / 1000 / (24 * 60)  # g/m^2/day #TODO: Check per day vs per second units....
Q_source = (
    62  # g/m^2/year  ELder paper 7983.69 mg m-2 d-1 but then sees 10s of ppm at .5m
)

BTL_AREA = 600  # m^2
# convert to kg/s
Q_source = Q_source * BTL_AREA / (1000 * 365 * 24 * 60 * 60)
Q_source *= 10

retro_df = pd.read_csv("real_retro_locations.csv", skiprows=0)

beam_lengths = retro_df["r"]

sources_df = retro_df[retro_df.name == "BTL_EC"]

sources_x, sources_y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
sources = pd.DataFrame(
    {
        "x": float(sources_df.DeltaY) + sources_x.ravel(),
        "y": float(sources_df.DeltaX) + sources_y.ravel(),
        "z": 0,
        "strength": Q_source / len(sources_x.ravel()),
    }
)

# Manually take out variations in AGL
retros_real = pd.DataFrame(
    {"x": retro_df["DeltaY"], "y": retro_df["DeltaX"], "z": retro_df["DeltaZ"]}
)

# Create synthetic retros
x_synthetic = np.linspace(-1, 100, 10)
y_synthetic = np.linspace(-10, 10, 5)
y_synthetic = np.array([0, 20], dtype="float")

X_synthetic, Y_synthetic = np.meshgrid(x_synthetic, y_synthetic)
Z_synthetic = 1.5 + np.zeros_like(X_synthetic)

X_synthetic += float(sources_df.DeltaY)
Y_synthetic += float(sources_df.DeltaX)

retros_synthetic = pd.DataFrame(
    {"x": X_synthetic.ravel(), "y": Y_synthetic.ravel(), "z": Z_synthetic.ravel()}
)

# retros = pd.concat([retros_real, retros_synthetic])
retros = retros_real
retros = retros_synthetic
retros.z = 0

retros_elevated = retros.copy()
retros_elevated.z += 1.5

retros = pd.concat([retros, retros_elevated])

lg = np.array(np.sqrt(retros.x**2 + retros.y**2))  # approx just for testing

origin = [0, 0, 230]  # Everything in "tower-centric coordinates"
# origin = [0,0,1.5]
# origin = [0,0,0]

Us = np.linspace(0.5, 4, 20)
Us = Us[::-1]
Us = [0.75]
Dirs = [0]
stablities = ["A", "B", "C", "D", "E"]
stablities = ["D"]

# Pre-assign for memory
# estimate_store = np.zeros(len(Us))
# results = np.zeros((len(Us), 4))
# # pic_store = np.zeros((len(Us), len(len(retros))))

store_df = pd.DataFrame(
    columns=["wind_speed", "dir", "stability", "estimates", "C", "C_max"]
)

for stability in stablities:
    for i in tqdm.tqdm(range((len(Us)))):
        u = Us[i]
        u_dir = np.radians(0)  # np.radians(Dirs[i])

        pics = get_synthetic_measurement(
            retros,
            sources,
            origin,
            u_mag=u,
            u_dir=u_dir,
            stability=stability,
            plot=True,
        )

        # pic_store[i, :] = pics
        avg_conc = pics / lg
        # store results in store_df
        store_df = store_df.append(
            {
                "wind_speed": u,
                "dir": u_dir,
                "stability": stability,
                "estimates": avg_conc,
            },
            ignore_index=True,
        )

        if False:
            a = plt.gcf().get_axes()[0]
            visualize_bivarpiiate(results[i, :], ax=a)

store_df["maximum_concentration"] = store_df.estimates.apply(lambda x: np.nanmax(x))
store_df["average_concentration"] = store_df.estimates.apply(lambda x: np.nanmean(x))

# avg_conc = pic_store / lg
# avg_conc = avg_conc[0]  # TODO fix this indexing issue
# %%
# Plot everything.
# TODO: Wrap plotting code into function

plt.figure()

plt.axis("equal")

# Create grid for concentration visualization
side_length = 500
slice_height = 0.50
X, Y = np.meshgrid(
    np.linspace(sources.x.min() - 1000, retros.x.max(), num=side_length),
    np.linspace(origin[1], retros.y.max() + 100, num=side_length),
)
Z = slice_height + np.zeros_like(X)
C = np.zeros_like(X.ravel())

test_pts = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T

for i in range(len(sources)):
    C += general_concentration(
        test_pts, sources.iloc[i, 0:3], sources.strength.iloc[i], u, u_dir, stability
    )

plt.pcolormesh(
    X.reshape(side_length, side_length),
    Y.reshape(side_length, side_length),
    C.reshape(side_length, side_length),
    alpha=0.9,
    cmap=cm.lajolla,
    norm=colors.LogNorm(vmin=1e-4, vmax=10),
)

plt.colorbar(cmap=cm.lajolla, label="PPM enhancement at %.1f meters AGL" % slice_height)

plt.scatter(origin[0], origin[1], c="orange", label="Laser")

# plt.scatter(retros.x, retros.y, c='red', label='Retros', s=pic_store[0,:] / np.max(pic_store) * 50 )

plt.scatter(sources.x, sources.y, c="blue", label="Big Trail Lake")
plt.plot(
    sources.x.mean() + (np.cos(u_dir) * np.linspace(0, 1000)),
    sources.y.mean() + (np.sin(u_dir) * np.linspace(0, 1000)),
    ":k",
    label="Wind Direction",
)
plt.legend()

sns.scatterplot(
    x="x",
    y="y",
    data=retros,
    hue=pics,
    # size=pic_store[0, :],
    legend=False,
)

ppm_threshold = 1e-3

for i in range((avg_conc > ppm_threshold).sum()):
    x = retros.x[avg_conc > ppm_threshold].iloc[i]
    y = retros.y[avg_conc > ppm_threshold].iloc[i]
    s = str(np.round(1e3 * avg_conc[avg_conc > ppm_threshold][i], 1)) + " ppb"
    # plt.text(x=x-10, y=y+10, s=s, fontdict={'fontsize': 'xx-small'})
    plt.plot([origin[0], x], [origin[1], y], color="darkgreen", alpha=0.95)

plt.ylabel("Northing (m)")
plt.xlabel("Easting (m)")
# plt.ylim(bottom=0)
# plt.xlim(right=0)
plt.title('Wind: %.2f m/s, Stability: "%s"' % (u, stability))

plt.figure()
sns.scatterplot(
    data=store_df, x="wind_speed", y="maximum_concentration", s=100, hue="stability"
)
plt.ylabel("Maximum Path-integrated concentration [ppm]")
plt.xlabel("U [m/s]")
plt.title("10x BTL Mean Emission Scenario")
# %%

plt.figure()
plt.plot(x_synthetic, avg_conc.reshape((10, 4))[:, 0])
plt.plot(x_synthetic, avg_conc.reshape((10, 4))[:, 1])
plt.ylabel("Concentretaion measurement [ppm]")
plt.xlabel("distance from emissions center [m]")
plt.xlabel("distance from emissions center [m]")
plt.title("Near-Feature Measurements ")
C.max()
plt.text(30, 0.0025, "Maximum .5m concentration: %.2f ppm" % C.max())
s_real
retros = retros_synthetic
retros.z = 0

retros_elevated = retros.copy()
retros_elevated.z += 1.5

retros = pd.concat([retros, retros_elevated])

lg = np.array(np.sqrt(retros.x**2 + retros.y**2))  # approx just for testing

origin = [0, 0, 230]  # Everything in "tower-centric coordinates"
# origin = [0,0,1.5]
# origin = [0,0,0]

Us = np.linspace(0.5, 4, 20)
Us = Us[::-1]
Us = [0.75]
Dirs = [0]
stablities = ["A", "B", "C", "D", "E"]
stablities = ["D"]

# Pre-assign for memory
# estimate_store = np.zeros(len(Us))
# results = np.zeros((len(Us), 4))
# # pic_store = np.zeros((len(Us), len(len(retros))))

store_df = pd.DataFrame(
    columns=["wind_speed", "dir", "stability", "estimates", "C", "C_max"]
)

for stability in stablities:
    for i in tqdm.tqdm(range((len(Us)))):
        u = Us[i]
        u_dir = np.radians(0)  # np.radians(Dirs[i])

        pics = get_synthetic_measurement(
            retros,
            sources,
            origin,
            u_mag=u,
            u_dir=u_dir,
            stability=stability,
            plot=True,
        )

        # pic_store[i, :] = pics
        avg_conc = pics / lg
        # store results in store_df
        store_df = store_df.append(
            {
                "wind_speed": u,
                "dir": u_dir,
                "stability": stability,
                "estimates": avg_conc,
            },
            ignore_index=True,
        )

        if False:
            a = plt.gcf().get_axes()[0]
            visualize_bivarpiiate(results[i, :], ax=a)

store_df["maximum_concentration"] = store_df.estimates.apply(lambda x: np.nanmax(x))
store_df["average_concentration"] = store_df.estimates.apply(lambda x: np.nanmean(x))

# avg_conc = pic_store / lg
# avg_conc = avg_conc[0]  # TODO fix this indexing issue
# %%
# Plot everything.
# TODO: Wrap plotting code into function

plt.figure()

plt.axis("equal")

# Create grid for concentration visualization
side_length = 500
slice_height = 0.50
X, Y = np.meshgrid(
    np.linspace(sources.x.min() - 1000, retros.x.max(), num=side_length),
    np.linspace(origin[1], retros.y.max() + 100, num=side_length),
)
Z = slice_height + np.zeros_like(X)
C = np.zeros_like(X.ravel())

test_pts = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T

for i in range(len(sources)):
    C += general_concentration(
        test_pts, sources.iloc[i, 0:3], sources.strength.iloc[i], u, u_dir, stability
    )

plt.pcolormesh(
    X.reshape(side_length, side_length),
    Y.reshape(side_length, side_length),
    C.reshape(side_length, side_length),
    alpha=0.9,
    cmap=cm.lajolla,
    norm=colors.LogNorm(vmin=1e-4, vmax=10),
)

plt.colorbar(cmap=cm.lajolla, label="PPM enhancement at %.1f meters AGL" % slice_height)

plt.scatter(origin[0], origin[1], c="orange", label="Laser")

# plt.scatter(retros.x, retros.y, c='red', label='Retros', s=pic_store[0,:] / np.max(pic_store) * 50 )

plt.scatter(sources.x, sources.y, c="blue", label="Big Trail Lake")
plt.plot(
    sources.x.mean() + (np.cos(u_dir) * np.linspace(0, 1000)),
    sources.y.mean() + (np.sin(u_dir) * np.linspace(0, 1000)),
    ":k",
    label="Wind Direction",
)
plt.legend()

sns.scatterplot(
    x="x",
    y="y",
    data=retros,
    hue=pics,
    # size=pic_store[0, :],
    legend=False,
)

ppm_threshold = 1e-3

for i in range((avg_conc > ppm_threshold).sum()):
    x = retros.x[avg_conc > ppm_threshold].iloc[i]
    y = retros.y[avg_conc > ppm_threshold].iloc[i]
    s = str(np.round(1e3 * avg_conc[avg_conc > ppm_threshold][i], 1)) + " ppb"
    # plt.text(x=x-10, y=y+10, s=s, fontdict={'fontsize': 'xx-small'})
    plt.plot([origin[0], x], [origin[1], y], color="darkgreen", alpha=0.95)

plt.ylabel("Northing (m)")
plt.xlabel("Easting (m)")
# plt.ylim(bottom=0)
# plt.xlim(right=0)
plt.title('Wind: %.2f m/s, Stability: "%s"' % (u, stability))

plt.figure()
sns.scatterplot(
    data=store_df, x="wind_speed", y="maximum_concentration", s=100, hue="stability"
)
plt.ylabel("Maximum Path-integrated concentration [ppm]")
plt.xlabel("U [m/s]")
plt.title("10x BTL Mean Emission Scenario")
# %%

plt.figure()
plt.plot(x_synthetic, avg_conc.reshape((10, 4))[:, 0])
plt.plot(x_synthetic, avg_conc.reshape((10, 4))[:, 1])
plt.ylabel("Concentretaion measurement [ppm]")
plt.xlabel("distance from emissions center [m]")
plt.xlabel("distance from emissions center [m]")
plt.title("Near-Feature Measurements ")
C.max()
plt.text(30, 0.0025, "Maximum .5m concentration: %.2f ppm" % C.max())
