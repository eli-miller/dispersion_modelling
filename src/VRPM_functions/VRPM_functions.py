#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:28:22 2022

@author: elimiller
"""
import math

import cmcrameri.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import yaml
from matplotlib import colors
from scipy.optimize import minimize, Bounds
import xml.etree.ElementTree as ET

# from shapely import MultiPoint
# from shapely.geometry import Polygon, Point
# import pyproj


############################################################################################################
# Functions for simulating a single retroreflector
############################################################################################################
def line_points(start, end, num_pts=5000):
    """start and end 3x1 array-like of xyz coords."""
    x1, y1, z1 = start
    (
        x2,
        y2,
        z2,
    ) = end

    t = np.linspace(0, 1, num=num_pts)
    xs = x1 + (x2 - x1) * t
    ys = y1 + (y2 - y1) * t
    zs = z1 + (z2 - z1) * t
    return np.array([xs, ys, zs]).T


def l_to_PG(l, method="Foken"):
    """Turns obukhov length into Pasquill-Gifford stability class based on table 1 from Munoz Esparaza
    Note that our function for dispersion coefficcients below does not have class G, so we will use F instead.
    Also this paper does not have a class E (slightly stable)... TODO: figure this out.
    l: obukhov length in m

    # TODO: See Foken Micrometerology Page 317 (Table 8.3) for another classification scheme
    # TODO: See Table A.5 in the appendix of the below paper:Estimating the Flammable Mass of a Vapor Cloud
    # by John L. Woodward 1998
    https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470935361.app1
    """

    if method == "Munoz":
        if 0 < l < 200:  # Very Stable:
            return "F"
        elif 200 < l < 1000:  # Stable
            return "F"
        elif np.abs(l) > 1000:  # Near Neutral
            return "D"
        elif -1000 < l < -200:  # Unstable
            return "B"
        elif -200 < l < 0:  # Very Unstable
            return "A"

    elif method == "Foken":
        if -30 <= l < 0:
            return "A"
        elif -100 <= l < -30:
            return "B"
        elif -3000 <= l < -100:
            return "C"
        elif 0 < l <= 60:
            return "F"
        elif 60 < l <= 250:
            return "E"
        elif 250 < l < 5000:
            return "D"
        else:
            return "H"  # This is a test. Not real but we aren't catching extremely large values of L (Which I think should be neutral?)

    elif method == "Woodward":
        raise NotImplementedError


def dispersion_coeffs(stability):
    """
    Dispersion Coefficients for gaussian plume model, Table 6.2 in De Visscher.

    Parameters
    ----------
    stability : str
        Stability class for the atmosphere. Can be "A", "B", "C", "D", "E", or "F".

    Returns
    -------
    a, b, c, d, e, f : float
        Coefficients for the dispersion model.

    """
    # Class A
    if stability == "A":  # Very unstable
        a, b, c, d, e, f = 0.22, 0.0001, 0.5, 0.2, 0, 0
    elif stability == "B":  # Moderately unstable
        a, b, c, d, e, f = 0.16, 0.0001, 0.5, 0.12, 0, 0
    elif stability == "C":  # Slightly unstable
        a, b, c, d, e, f = 0.11, 0.0001, 0.5, 0.08, 0.0002, 0.5
    elif stability == "D":  # Neutral
        a, b, c, d, e, f = 0.08, 0.0001, 0.5, 0.06, 0.0015, 0.5
    elif stability == "E":  # Slightly stable
        a, b, c, d, e, f = 0.06, 0.0001, 0.5, 0.03, 0.0003, 1
    elif stability == "F":  # Stable
        a, b, c, d, e, f = 0.04, 0.0001, 0.5, 0.016, 0.0003, 1
    else:
        raise ValueError("stability must be A, B, C, D, E, or F")
    return a, b, c, d, e, f


# def general_concentration(x,y,z, Q=1, wind_speed=1, stability="E"):
def general_concentration(beam_loc, source_loc, Q, u_mag, u_dir, stability="E"):
    """Q [kg/s] emission rate and returns a concentration [ppm] at that point"""
    """ With reflection. Page """

    a, b, c, d, e, f = dispersion_coeffs(stability)
    (
        x0,
        y0,
        z0,
    ) = source_loc

    try:  # just leaving for back-compatibility to try to not break things
        # TODO: Rewrite everything. This is a mess.
        x = beam_loc[:, 0]
        y = beam_loc[:, 1]
        z = beam_loc[:, 2]
    except:
        x = beam_loc["x"]
        y = beam_loc["y"]
        z = beam_loc["z"]

    # Transform into global coordinates
    x_glob = x - x0
    y_glob = y - y0
    z_glob = z - z0  # Not necessary for ground-based sources

    # Figure out how much to rotate in the xy plane from GP formulation
    phi = np.arctan2(y_glob, x_glob)  # Angle of laser point from east axis from x0y0

    l_xy = np.sqrt(x_glob**2 + y_glob**2)
    # New coordinates
    x_prime = l_xy * np.cos(u_dir - phi)
    y_prime = l_xy * np.sin(u_dir - phi)

    sigma_y = a * x_prime / (1 + b * x_prime) ** c
    sigma_z = d * x_prime / (1 + e * x_prime) ** f

    # C = Q/(2*pi*sigma_y*sigma_z*wind_speed)  * exp(-y_prime**2/(2*sigma_y**2)) * (exp(-(z_glob)**2/(2*sigma_z**2))+exp(-(z_glob)**2/(2*sigma_z**2)))
    h = 0
    C = (
        Q
        / (2 * np.pi * u_mag * sigma_y * sigma_z)
        * np.exp(-0.5 * y_prime**2 / sigma_y**2)
        * (
            np.exp(-0.5 * (z - h) ** 2 / sigma_z**2)
            + np.exp(-0.5 * (z + h) ** 2 / sigma_z**2)
        )
    )

    C *= 24.45 / 16.04 * 1e6  # Convert from kg/m^3 to ppm

    C[x_prime < 0] = 0

    return C


def plume_contribution(beam_loc, source_loc, Q, u_mag, u_dir):
    """Returns: concentration [g/m^3] given plume and POI params.

    Inputs:
      beam_loc: x,y,z location of beam point. easting, northing, elev.= m]
      source_loc: x,y,z, location of source. Easting, northing, elevation
      Q: emmission Rate [g/s]
      u_mag: wind speed [m/s]
      u_dir: wind direction [angle from east axis, radians]

      Using plume formula from Elder 2021 Supplemental info (Bio. geo. chem....

      Want to convert this fuction to more sophisticated dependent on stability"""

    K = 2.1e-0  # dispersion coefficient m^2/sec (STP). Could change for temperature
    # K = .21 #cm^2/s

    # x0 = source_loc[0] # Easting
    # y0 = source_loc[1] # Northing
    # z0 = source_loc[2] # Z

    (
        x0,
        y0,
        z0,
    ) = source_loc

    # y, x, z = np.ogrid[-y_extent:y_extent, x0:x0+x_extent, 0:z_extent]
    x = beam_loc[:, 0]
    y = beam_loc[:, 1]
    z = beam_loc[:, 2]

    # Transform into global coordinates
    x_glob = x - x0
    y_glob = y - y0
    z_glob = z - z0

    # Figure out how much to rotate in the xy plane from GP formulation
    phi = np.arctan2(y_glob, x_glob)  # Angle of laser point from east axis from x0y0

    l_xy = np.sqrt(x_glob**2 + y_glob**2)
    # New coordinates
    x_prime = l_xy * np.cos(u_dir - phi)
    y_prime = l_xy * np.sin(u_dir - phi)

    coeff = Q / (4 * np.pi * K * (x_prime))
    # coeff[coeff <  0 ] = 0

    term1 = -((y_prime) ** 2) / (4 * K * (x_prime) / u_mag)
    term2 = -((z - z0) ** 2) / (4 * K * (x_prime) / u_mag)

    Conc = coeff * np.exp(term1 + term2)

    Conc[Conc < 0] = 0  # don't allow Nan or zero values

    return Conc


def get_synthetic_measurement(
    retros, sources, origin, u_mag, u_dir, stability="A", pts_per_beam=10000, plot=False
):
    """
    Simulates a synthetic measurement of a plume for given retroreflectors, sources, wind speed and direction.

    Parameters
    ----------
    retros : pd.DataFrame
        Dataframe of retroreflector locations. Must have columns x, y, z.
    sources : pd.DataFrame
        Dataframe of source locations. Must have columns x, y, z, strength.
    origin : array-like
        The origin of the beam.
    u_mag : float
        Wind speed in m/s.
    u_dir : float
        Wind direction in radians from the east axis.
    stability : str, optional
        Stability class for the atmosphere. Default is "A" (very unstable).
    pts_per_beam : int, optional
        Number of points to simulate along each beam. Default is 10000.
    plot : bool, optional
        If True, plots the concentration along each beam and the maximum concentration. Default is True.

    Returns
    -------
    pic : np.ndarray
        Array of integrated concentrations along each beam.
    sim_info : dict
        Dictionary containing information about the simulation, including wind speed, wind direction, stability class,
        maximum concentration information, and a list of beam information dictionaries.
    """
    # plume_contribution(beam_loc, source_loc, Q, u_mag, u_dir):
    if plot:
        fig, (ax, ax1) = plt.subplots(2, 1, sharex=True)

    pic = np.zeros((len(retros),))

    max_conc = 0
    max_beam_info = None
    beam_info_list = []

    # for i_retro in tqdm.tqdm(range(len(retros))):
    for i_retro in range(len(retros)):
        beam_pts = line_points(origin, np.array(retros.iloc[i_retro, :]), pts_per_beam)
        segment_length = (
            np.sqrt(np.sum((beam_pts[-1, :] - beam_pts[0, :]) ** 2)) / pts_per_beam
        )

        conc_store = np.zeros((len(beam_pts), len(sources)))

        for i_source in range(len(sources)):
            conc_store[:, i_source] = general_concentration(
                # plume_contribution(
                beam_pts,
                sources[["x", "y", "z"]].iloc[i_source],
                sources["strength"].iloc[i_source],
                u_mag,
                u_dir,
                stability=stability,
            )

        conc_resolved = np.sum(
            conc_store, axis=1
        )  # retain for visulaization / debugging
        pic[i_retro] = np.sum(conc_resolved * segment_length, axis=0)
        # See https://teesing.com/media/library/tools/understanding-units-of-measurement.pdf for units

        max_local_conc = np.max(conc_resolved)
        max_local_conc_location = beam_pts[np.argmax(conc_resolved)]
        end_of_beam_conc = conc_resolved[-1]

        beam_info_list.append(
            {
                "retro_index": i_retro,
                "max_local_concentration": max_local_conc,
                "max_local_concentration_location": max_local_conc_location,
                "end_of_beam_concentration": end_of_beam_conc,
            }
        )

        if np.max(conc_resolved) > max_conc:
            max_conc = np.max(conc_resolved)

            max_beam_info = {
                "retro_index": i_retro,
                "max_concentration": max_conc,
                "max_concentration_location": beam_pts[np.argmax(conc_resolved)],
            }

        if plot:
            ax1.scatter(beam_pts[:, 1], conc_resolved)
            color_path = ax.scatter(
                beam_pts[:, 1],
                beam_pts[:, 2],
                c=conc_resolved,
                zorder=0,
                vmin=0,
                vmax=max_conc,
            )
            ax1.set_ylabel("ppm")
            ax.scatter(
                retros["y"].iloc[i_retro],
                retros["z"].iloc[i_retro],
                marker="x",
                label=str(np.round(pic[i_retro], 2)) + "ppm-m",
            )
            ax.set_title(
                "Wind: %.1f m/s, Direction: %.0f, Class: %s"
                % (u_mag, np.degrees(u_dir), stability)
            )
            ax.plot([0, np.max(retros.y)], [21, 21], "--k")  #
    if plot:
        ax.legend()

    # Collect some info about the simulation into a dictionary:
    sim_info = {
        "u_mag": u_mag,
        "u_dir": u_dir,
        "stability": stability,
        "max_beam_info": max_beam_info,
        "beam_info_list": beam_info_list,
    }

    return pic, sim_info  # len = n_retros


############################################################################################################
# Functions for VRPM Fitting
############################################################################################################
def polar_gaussian2d_simplified(A, mu_y, sigma_y, sigma_z, r, theta):
    """equation 3 from https://pubs.acs.org/doi/pdf/10.1021/es0017108
    with assumptions ground-based emmissions, and no covariance between z&y
    Put r and theta later for similar structure with scipy minimize args
    NOTE: there is an error in the paper. Should be plus in the exponential..."""

    return (
        A
        / (2 * np.pi * sigma_y * sigma_z)
        * np.exp(
            -1
            / 2
            * (
                ((r * np.cos(theta) - mu_y) ** 2 / sigma_y**2)
                + ((r * np.sin(theta)) ** 2 / sigma_z**2)
            )
        )
    )


def new_objective_fun(inputarray, retros, origin, pics, pts_per_beam=50000):
    """

    @param inputarray: A, mu_y, sigma_y, sigma_z
    @param retros: dataframe of retro locations with columns x, y, z (meters)
    @param origin: array-like, 3x1 x, y, z (meters)
    @param pics: measured path-integrated concentration corresonding to retros. (ppm-m)
    @param pts_per_beam: affects resolution.
    @return: SSE - Sum of Squared error between actual measured PICs and measurements from proposed candidate map.
    """
    # TO DO: penalize sharpness (3rd derivitave?)
    # To DO: allow user to input stability class as aspect ratio
    A, mu_y, sigma_y, sigma_z = inputarray

    offset = retros - origin  # need to change back to global coords later
    rs = np.sqrt(((offset) ** 2).sum(axis=1))  # Dataframe of radii

    thetas = np.array(
        np.arctan2(offset.z, offset.y)
    )  # Assume angle in the y-z plane [radians]

    r_beam = np.linspace(0, rs, pts_per_beam)

    pic_candidate = np.zeros((len(rs),))

    for i in range(len(rs)):
        conc_along_beam = polar_gaussian2d_simplified(
            A, mu_y, sigma_y, sigma_z, r_beam[:, i], thetas[i]
        )
        pic_candidate[i] = np.sum(conc_along_beam * rs[i] / pts_per_beam)

    SSE = np.sum((pics - pic_candidate) ** 2)
    print("Candidate PICS: " + str(pic_candidate))
    # print('SSE = ' + str(SSE))
    return SSE


def fit_bivariate_gauss(
    pics, u_mag, u_dir, retros, origin, plot=False, verbose=False, x0=None, bnds=None
):
    if x0 is None:  # Give standard initial guess.
        x0 = [5 * pics[2], 1500, 10, 2.6]
    else:
        pass

    # TODO: Move initial guess outside of this function and into main script
    if bnds is None:
        # bnds = Bounds((0, 14001400, -np.inf, -np.inf), (1e9, 2000, np.inf, np.inf))
        bnds = Bounds(
            (-np.inf, -np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf)
        )
    else:
        pass

    # result = minimize(new_objective_fun, x0=x0, args=(retros, origin, pics), bounds=bnds ,method='L-BFGS-B')
    result = minimize(
        new_objective_fun,
        x0=x0,
        args=(retros, origin, pics),
        bounds=bnds,
        method="Nelder-Mead",
        options={"maxiter": 5000},
    )

    result.x[1] += origin[1]  # add back in origin y component

    # result = basinhopping(new_objective_fun, x0=x0, minimizer_kwargs = {"args": (retros, origin, pics)})

    print(result)
    if not result.success:
        print("Optimization Failed")

    ### most of this is for plotting. Main goal is to "integrate" in order to
    # print(result)
    A_fit, mu_y_fit, sigma_y_fit, sigma_z_fit = result.x

    flux_grid_pts = 500

    y_flux = np.linspace(
        mu_y_fit - 3 * sigma_y_fit, mu_y_fit + 3 * sigma_y_fit, flux_grid_pts
    )
    y_flux_rel = y_flux  # - origin[1]

    z_flux = np.linspace(0, 3 * sigma_z_fit, flux_grid_pts)
    z_flux_rel = z_flux  # - origin[2]

    Y_flux_rel, Z_flux_rel = np.meshgrid(y_flux_rel, z_flux_rel)

    Theta_flux = np.arctan2(Z_flux_rel, Y_flux_rel)
    R_flux = np.sqrt(Y_flux_rel**2 + Z_flux_rel**2)

    dr = R_flux[0, 1] - R_flux[0, 0]
    dtheta = Theta_flux[1, 0] - Theta_flux[0, 0]

    cell_areas = R_flux * dr * dtheta

    fit_conc = polar_gaussian2d_simplified(
        A_fit, mu_y_fit, sigma_y_fit, sigma_z_fit, R_flux, Theta_flux
    )

    area = (y_flux[-1] - y_flux[0]) * (z_flux[-1] - z_flux[0])
    fit_flux = (
        16.04 / (24.45 * 1000) * (fit_conc * cell_areas).sum() * u_mag * np.cos(u_dir)
    )

    if plot:
        plt.pcolormesh(y_flux, z_flux, fit_conc)

    if verbose:
        print("integrated from fit_td")
        print(fit_flux)
        print("max path integrated concentration")
        print(np.max(pics))

    return fit_flux, result.x


def visualize_bivariate(inputs, ax=None):
    A_fit, mu_y_fit, sigma_y_fit, sigma_z_fit = inputs

    flux_grid_pts = 500

    y_flux = np.linspace(
        mu_y_fit - 3 * sigma_y_fit, mu_y_fit + 3 * sigma_y_fit, flux_grid_pts
    )
    y_flux_rel = y_flux  # - origin[1]

    z_flux = np.linspace(0, 3 * sigma_z_fit, flux_grid_pts)
    z_flux_rel = z_flux  # - origin[2]

    Y_flux_rel, Z_flux_rel = np.meshgrid(y_flux_rel, z_flux_rel)

    R_flux = np.sqrt(Y_flux_rel**2 + Z_flux_rel**2)
    Theta_flux = np.arctan2(Z_flux_rel, Y_flux_rel)

    fit_conc = polar_gaussian2d_simplified(
        A_fit, mu_y_fit, sigma_y_fit, sigma_z_fit, R_flux, Theta_flux
    )

    if ax is not None:
        CS = ax.contour(
            y_flux, z_flux, fit_conc, colors="black", label="Reconstruction"
        )
        ax.clabel(CS, inline=True, fontsize=10)

    else:
        plt.contour(y_flux, z_flux, fit_conc)


############################################################################################################
# Function for Simulating DCS Measurements
############################################################################################################
def read_measurement_geometry(file_path):
    """Reads in config file that defines source, retro, and tower locations. Returns dataframes of sources and retros
    in the appropriate coordinate scheme

    Assumes Q_Source_total in # g/m^2/year

    BUG: Right now if you have a single source or reto, it puts it at the bottom of the range, not the middle.
    See if I want to change this after using"""

    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def create_measurement_geometry(config, simulation_type="synthetic"):
    """
    Creates a dataframe of source locations and a dataframe of retro locations based on the configuration file.
    The function can handle two types of simulations: "synthetic" and "real".
    For "synthetic" simulations, it assumes that the sources and retros will be defined on a grid with specified spacing.
    For "real" simulations, it assumes that the configuration file contains specific retro and source locations.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters. The required keys depend on the simulation type.
    simulation_type : str, optional
        The type of simulation. Can be either "synthetic" or "real". Default is "synthetic".

    Returns
    -------
    sources : pd.DataFrame
        Dataframe of source locations. Must have columns x, y, z.
    retros : pd.DataFrame
        Dataframe of retro locations. Must have columns x, y, z.
    """
    # TODO: this is a mess just for back-compatibility.
    # I've introduced a third returned dataframe for point sensors, but I don't want to break old code.
    # I should clean this up at some point.

    if simulation_type == "synthetic":
        return create_measurement_geometry_synthetic(config)
    elif simulation_type == "real":
        return create_measurement_geometry_real(config)
    else:
        raise ValueError(
            f"simulation_type must be either 'synthetic' or 'real'. Received: {simulation_type}"
        )


def create_measurement_geometry_real(config):
    """
    Create a dataframe of source locations and a dataframe of retro locations based on the config file.
    Assumes that the config file contains specific retro and source locations

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters.

    Returns
    -------
    sources : pd.DataFrame
        Dataframe of source locations. Must have columns x, y, z.
    retros : pd.DataFrame
        Dataframe of retro locations. Must have columns x, y, z.
    """

    Q_source_total = config["Q_source_total"]
    source_area = config["source_area"]

    source_locs = config["source_locs"]
    retro_locs = config["retro_locs"]
    point_sensor_locs = config["point_sensor_locs"]

    sources = pd.DataFrame(source_locs, columns=["x", "y", "z"])
    retros = pd.DataFrame(retro_locs, columns=["x", "y", "z"])
    point_sensors = pd.DataFrame(point_sensor_locs, columns=["x", "y", "z"])

    Q_source = (Q_source_total * source_area) / len(sources)
    Q_source /= 1000 * 365 * 24 * 60 * 60  # Convert to kg/s from g/m^2/year

    sources["strength"] = Q_source

    return sources, retros, point_sensors


def create_measurement_geometry_synthetic(config):
    """
    Creates a dataframe of source locations and a dataframe of retro locations based on the config file.
    Assumes that the sources and retros will be defined on a grid with specified spacing.

    Parameters
    ----------
    config : dict
        Dictionary of configuration parameters.

    Returns
    -------
    sources : pd.DataFrame
        Dataframe of source locations. Must have columns x, y, z.
    retros : pd.DataFrame
        Dataframe of retro locations. Must have columns x, y, z.
    """

    Q_source_total = config["Q_source_total"]
    source_area = config["source_area"]

    n_source_x = config["n_source_x"]
    n_source_y = config["n_source_y"]
    span_source_x = config["span_source_x"]
    span_source_y = config["span_source_y"]
    source_loc_y = config["source_loc_y"]
    source_loc_x = config["source_loc_x"]
    source_loc_z = config["source_loc_z"]

    n_retro_x = config["n_retro_x"]
    n_retro_y = config["n_retro_y"]
    n_retro_z = config["n_retro_z"]
    span_retro_x = config["span_retro_x"]
    span_retro_y = config["span_retro_y"]
    span_retro_z = config["span_retro_z"]
    retro_loc_x = config["retro_loc_x"]
    retro_loc_y = config["retro_loc_y"]

    sources_x, sources_y = np.meshgrid(
        source_loc_x + np.linspace(span_source_x[0], span_source_x[1], n_source_x),
        source_loc_y + np.linspace(span_source_y[0], span_source_y[1], n_source_y),
    )

    sources = pd.DataFrame(
        {
            "x": sources_x.ravel(),
            "y": sources_y.ravel(),
            "z": source_loc_z,
        }
    )

    retros_x, retros_y, retros_z = np.meshgrid(
        retro_loc_x + np.linspace(span_retro_x[0], span_retro_x[1], n_retro_x),
        retro_loc_y + np.linspace(span_retro_y[0], span_retro_y[1], n_retro_y),
        np.linspace(span_retro_z[0], span_retro_z[1], n_retro_z),
    )

    retros = pd.DataFrame(
        {"x": retros_x.ravel(), "y": retros_y.ravel(), "z": retros_z.ravel()}
    )

    return sources, retros


def add_point_sensors_to_domain(point_sensors, axs):
    """
    Add the point sensors to the domain plot.

    Parameters
    ----------
    point_sensors : pd.DataFrame
        Dataframe of point sensor locations. Must have columns x, y, z.
    axs : matplotlib.axes.Axes
        Axes to plot on.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure containing the plot.
    axs : matplotlib.axes.Axes
        Axes containing the plot.
    """

    axs.scatter(
        point_sensors.x,
        point_sensors.y,
        point_sensors.z,
        color="C5",
        label="Point Sensors",
    )
    axs.legend()

    return plt.gcf(), axs


def plot_simulation_domain(
    sources, retros, origin, draw_beams=False, plot_3d=True, fig=None, ax=None
):
    """Plots the simulation domain. If fig and ax are not provided, creates a new figure and axes.
    Parameters
    ----------
    sources : pd.DataFrame
        Dataframe of source locations. Must have columns x, y, z.
    retros : pd.DataFrame
        Dataframe of retro locations. Must have columns x, y, z.
    origin : array-like
        3x1 array of x, y, z coordinates of the tower.
    draw_beams : bool, optional
        Whether to draw lines between the tower and each retro. The default is False.
    plot_3d : bool, optional
        Whether to plot in 3D. The default is True.
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If None, creates a new figure. The default is None.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new axes. The default is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Axes containing the plot.

    """
    if fig is None and ax is None:
        fig = plt.figure(figsize=(8, 8))

    if plot_3d:
        if ax is None:
            ax = fig.add_subplot(111, projection="3d")
            ax.set_zlabel("Z")

        ax.scatter(sources.x, sources.y, sources.z, color="C0", label="Sources")
        ax.scatter(retros.x, retros.y, retros.z, color="C1", label="Retros")
        ax.scatter(origin[0], origin[1], origin[2], color="C2", label="Tower")

    else:
        if ax is None:
            ax = fig.add_subplot(111)
            ax.axis("equal")

        ax.scatter(sources.x, sources.y, color="C0", label="Sources")
        ax.scatter(retros.x, retros.y, color="C1", label="Retros")
        ax.scatter(origin[0], origin[1], color="C2", label="Tower")

    if draw_beams:  # Plot lines between the origin and each retro
        for i in range(len(retros)):
            ax.plot(
                [origin[0], retros.x.iloc[i]],
                [origin[1], retros.y.iloc[i]],
                [origin[2], retros.z.iloc[i]],
                color="green",
                alpha=0.5,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    plt.show()
    return fig, ax


def filter_retros(retros, threshold=1e-3):
    return retros[retros.measurement > threshold]


def visualize_enhanced_retros(retros, sources, threshold=1e-3):
    retros_filtered = filter_retros(retros, threshold=threshold)

    plt.figure()
    sns.scatterplot(x="x", y="y", data=retros)
    sns.scatterplot(x="x", y="y", data=retros_filtered, color="red", marker="x")
    sns.scatterplot(x="x", y="y", data=sources, color="orange", marker="o")


def plot_retro_z_profiles(retros):
    # For each unique x,y, pair, plot the concentration vs z on the same plot

    plt.figure()
    for i, (x, y) in enumerate(retros.groupby(["x", "y"])):
        sns.lineplot(x="z", y="measurement", data=y, label="%.0f, %.0f" % x)


def create_beam_info(sim_info):
    beam_info = pd.DataFrame(sim_info["beam_info_list"])
    beam_info["max_at_end"] = (
        beam_info.end_of_beam_concentration == beam_info.max_local_concentration
    )
    beam_info["retro_max_ratio"] = (
        beam_info.max_local_concentration / beam_info.end_of_beam_concentration
    )

    return beam_info


def fiter_beam_info(beam_info):
    beam_info = beam_info[beam_info.end_of_beam_concentration > 1e-4]
    beam_info = beam_info[beam_info.max_local_concentration > 1e-4]

    return beam_info


def get_meas_field(summary, z):
    """returns a field of synthetic DCS measurements at a given retro height from the output of a simulation.
    Parameters
    ----------
    summary : pd.DataFrame

        Dataframe containing the synthetic measurements.
    z : float
        Height at which to plot the field. Must exist within summary dataframe.
    Returns
    -------
    meas_field : np.ndarray
        2D array of synthetic measurements at the given height.
    X, Y : np.ndarray
        2D arrays of x and y coordinates of the measurement field.
    """

    if z not in summary.z.unique():
        raise ValueError(
            f"z = {z} is not a valid height. Must be one of {summary.z.unique()}"
        )

    meas_field = summary.query(f"z == {z}").pivot(
        index="y", columns="x", values="measurement"
    )
    X = summary.query(f"z == {z}").pivot(index="y", columns="x", values="x")
    Y = summary.query(f"z == {z}").pivot(index="y", columns="x", values="y")

    return meas_field, X, Y


def plot_meas_field(
    X,
    Y,
    meas_field,
    ax=None,
    vmin=1e-3,
    vmax=None,
    detection_threshold=3e-3,
    colorbar=True,
):
    """Plots a synthetic DCS measurement field.
    Parameters
    ----------
    X, Y : np.ndarray
        2D array of x coordinates of the measurement field.
    meas_field : np.ndarray
        2D array of synthetic measurements at the given height.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure and axes.
    show_retros : bool, optional
        Whether to plot the retros in the field. The default is True.
    vmin : float, optional
        Minimum value for the colorbar. This should represent DCS detection limit
    vmax : float, optional
        Maximum value for the colorbar. Used to synchromize plotting. Default is None.
    detection_threshold : float, optional
        Threshold for detection. Used to plot the detection threshold as a contour.
    Returns
    -------
    None."""

    if ax is None:
        fig, ax = plt.subplots()

    ax.pcolormesh(
        X, Y, meas_field, cmap=cm.davos, norm=colors.LogNorm(vmin=vmin, vmax=vmax)
    )
    ax.contour(
        X,
        Y,
        meas_field,
        levels=[detection_threshold],
        colors="k",
        linestyles="--",
        label=f"Detection Limit {1000 * detection_threshold:.1e} ppb",
    )
    ax.set_aspect("equal")

    if colorbar:
        cbar = ax.figure.colorbar(
            ax.collections[0], ax=ax, location="bottom", orientation="horizontal"
        )
        cbar.set_label("[ppb]")
        cbar.ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
        cbar.ax.xaxis.set_minor_formatter(ticker.FuncFormatter(format_ticks))

        # cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), rotation=90)

    ax.set_xlim([np.min(X), np.max(X)])
    ax.set_ylim([np.min(Y), np.max(Y)])


def format_ticks(value, tick_number=None):
    """
    Formats the colorbar ticks to be in units of ppb.

    Parameters
    ----------
    value : float
        Value of the tick.
    tick_number : int
        Index of the tick. Not implemented.
    """

    # if tick_number is not None:
    #     raise NotImplementedError("tick_number is not implemented")
    # Used to turn ppm to ppb in colorbar and remove x10^ uncessary characeters
    return f"{1000 * value:.1f}"


def plot_delta_z(summary, z_vals, vmax_plot=0.1):
    """Plots a synthetic DCS measurement field.
    Parameters
    ----------
    summary : pd.DataFrame
        Dataframe containing the synthetic measurements.
    z_vals : list
        List of heights at which to plot the field. Must exist within summary dataframe.
    vmax_plot : float, optional
        Maximum value for the colorbar. Used to synchromize plotting.
    """

    meas_0, X, Y = get_meas_field(summary, z_vals[0])
    meas_z, _, _ = get_meas_field(
        summary, z_vals[1]
    )  # X, y field should be the same for all z

    fig, ax = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
    plot_meas_field(X, Y, meas_0, ax=ax[0], vmax=vmax_plot)
    plot_meas_field(X, Y, meas_z, ax=ax[1], vmax=vmax_plot, colorbar=False)
    plot_meas_field(X, Y, meas_0 - meas_z, ax=ax[2], vmax=vmax_plot, colorbar=False)

    ax[0].set_title(f"z = {z_vals[0]} m")
    ax[1].set_title(f"z = {z_vals[1]} m")
    ax[2].set_title(f"z = {z_vals[0]} - {z_vals[1]} m")
    plt.tight_layout()


def assign_source_strength(sources, total_source_strength, source_area):
    """Assigns source strength to each source based on the total source strength.
    Parameters
    ----------
    sources : pd.DataFrame
        Dataframe of source locations. Must have columns x, y, z.
    total_source_strength : float
        Total source strength in g/m^2/year.
    source_area : float
        Area of the source grid in m^2.
    Returns
    -------
    sources : pd.DataFrame
        Dataframe of source locations. Must have columns x, y, z, strength.
    """

    Q_source = (total_source_strength * source_area) / len(sources)
    Q_source /= (
        1000 * 365 * 24 * 60 * 60
    )  # Convert to kg/s (plume model) from g/m^2/year (Katey et al)

    sources["strength"] = Q_source

    return sources


def simulate_stacked_retros(
    u,
    stability,
    u_dir=0,
    Q=0,
    plot=False,
    save_fig_dir=None,
    save_plot=False,
    config_file="geometry_config.yaml",
):
    """simulate a set of stacked retros for a given stability and wind speed
    Parameters
    ----------
    u : float
        Wind speed [m/s]
    stability : str
        Stability class. One of A, B, C, D, E, F
    u_dir : float, optional
        Wind direction. The default is 0. TODO: Units?
    plot : bool, optional
        Whether to plot the results. The default is False.
    save_fig_dir : str, optional
        Directory to save the figure (Including the filename). The default is None.
    save_plot : bool, optional
        Whether to save the figure. The default is False.
    config_file : str, optional
        Path to the config file. The default is 'geometry_config.yaml'.

    Returns
    -------
    summary : pd.DataFrame
        Dataframe containing the synthetic measurements.
    """

    config = read_measurement_geometry(config_file)
    sources, retros = create_measurement_geometry(config)
    sources = assign_source_strength(sources, Q, config["source_area"])
    origin = config["origin"]

    pics, sim_info = get_synthetic_measurement(
        retros,
        sources,  # This contains the strength information
        origin,
        u_mag=u,
        u_dir=u_dir,
        stability=stability,
        plot=False,
        # this is along beam plotting. Super slow. Keep off.
    )

    pathlengths = calculate_pathlength(retros, origin)

    avg_conc = pics / pathlengths

    beam_info = create_beam_info(sim_info)

    summary = pd.merge(
        retros, beam_info, how="outer", left_index=True, right_index=True
    )

    summary["measurement"] = avg_conc
    summary["x_rel"] = summary.x - summary.x.mean()
    summary["y_rel"] = summary.y - summary.y.mean()

    summary["stability"] = stability
    summary["wind_speed"] = u

    if plot:
        plot_delta_z(summary, z_vals=config["span_retro_z"], vmax_plot=0.02)
        plt.gcf().suptitle("%s stability, %.1f m/s" % (stability, u))

    if save_plot:
        plt.savefig(save_fig_dir, dpi=300)
        plt.close("all")

    return summary


def get_contour_coordinates(X, Y, C, contour_level):
    """
    Get coordinates of a contour at a specified level.

    Parameters:
    - X: 2D array, X-coordinates of the data grid.
    - Y: 2D array, Y-coordinates of the data grid.
    - C: 2D array, data values corresponding to X and Y.
    - contour_level: float, the contour level of interest.

    Returns:
    - contour_coordinates: list of arrays, each array contains (x, y) coordinates of the contour.

    *** CHAT-GPT Written ***

    """

    # Create a contour plot to extract contour coordinates
    fig1 = plt.figure()
    contour = plt.contour(X, Y, C, levels=[contour_level])
    plt.close(fig1)
    # TODO: is there a better way to do this without generating and closing a figure?

    # Extract the coordinates of the specified contour level
    contour_paths = contour.collections[0].get_paths()
    contour_coordinates = [path.vertices for path in contour_paths]

    return np.array(contour_coordinates[0])


def get_max_contour_locations(contours):
    """Get the location of the maximum concentration in the contour plot and their coordinates.

    Parameters
    ----------
    contours : np.ndarray
        The coordinates of the contour plot.

    Returns
    -------
    x_max_coords : tuple
        The x-coordinate of the maximum concentration and the corresponding y-coordinate.
    y_max_coords : tuple
        The y-coordinate of the maximum concentration and the corresponding x-coordinate.
    """

    # TODO: this function may be a good place to make sure the contour is closed
    if type(contours) is not np.ndarray:
        contours = np.array(contours)

    x_max, y_max = contours.max(axis=0)

    y_at_max_x = contours[contours[:, 0] == x_max, 1]
    x_at_max_y = contours[contours[:, 1] == y_max, 0]

    # Try to make y_at_max_x and x_at_max_y into scalars. If they are not a single number, then the contour is likely not closed.
    try:
        y_at_max_x = float(y_at_max_x)
    except:
        raise ValueError(
            f"y_at_max_x should be a single number. got {y_at_max_x}. \n Check that the contour is closed."
        )
    try:
        x_at_max_y = float(x_at_max_y)
    except:
        raise ValueError(
            f"x_at_max_y should be a single number. got {x_at_max_y}. \n Check that the contour is closed."
        )
    x_max_coords = (x_max, y_at_max_x)
    y_max_coords = (x_at_max_y, y_max)

    return x_max_coords, y_max_coords


def convert_wind_direction_from_anemometer(direction):
    """
    This function takes wind readings from an anemometor
    where zero is blowing from the north, increasing clockwise in degrees
    and returns a wind direction for simulation in radians
    where going along the "x" axis is zero degrees,
    increasing counter clockwise, unit circle style

    Parameters
    ----------
    direction : float
        Wind direction in degrees from north (CW).

    Returns
    -------
    direction : float
        Wind direction in radians from east (CCW).
    """

    new_direction_deg = -direction + 270
    # make in range 0-360
    if new_direction_deg > 360:
        new_direction_deg -= 360
    elif new_direction_deg < 0:
        new_direction_deg += 360

    # convert to radians
    new_direction_rad = np.radians(new_direction_deg)

    return new_direction_rad


def convert_wind_direction_to_anemometer(direction):
    """
    This function takes wind readings from a simulation or cartesian slop
    where zero isblowing from the east, increasing counter clockwise in radians
    and returns a wind direction for an anemometer in degrees
    where zero is blowing from the north, increasing clockwise

    Parameters
    ----------
    direction: float
        Wind direction in radians from east (CCW).


    Returns
    -------
    direction : float
        Wind direction in degrees from north (CW).


    """

    new_direction_deg = np.degrees(direction)
    new_direction_deg = -new_direction_deg + 90
    # make in range 0-360
    if new_direction_deg > 360:
        new_direction_deg -= 360
    elif new_direction_deg < 0:
        new_direction_deg += 360

    return new_direction_deg


def calculate_pathlength(retro, origin):
    """Calculate the pathlength from the origin to a retro

    Parameters
    ----------
    retro : pd.DataFrame
        Retro dataframe with columns x, y, z

    origin : list
        [x, y, z] of the origin

    Returns
    -------
    pathlength : float
        Pathlength from origin to retro
    """
    pathlength = np.sqrt(
        (retro.x - origin[0]) ** 2
        + (retro.y - origin[1]) ** 2
        + (retro.z - origin[2]) ** 2
    )

    return pathlength


def plot_contours(df, group_by):
    """Plot the contours from detection limit simulations

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the contours of the simulations.
    group_by : str
        Column to group the data by for plotting
    """

    for name, group in df.groupby(group_by):
        fig, ax = plt.subplots(figsize=(8, 6))
        for _, row in group.iterrows():
            contours = np.array(
                row["contours"]
            )  # Re-make array. This is a workaround that the dataframe doesn't like having 2d arrays as an entry

            ax.plot(
                contours[:, 0],
                contours[:, 1],
                label=f"{row['stability']}, {row['wind_speed']} [m/s], {row['wind_direction']} deg, {row['source_strength']} Q",
                linestyle="solid",
            )
        ax.legend()


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


def process_kml_to_relative_coords(kml_path, ref_lat, ref_lon):
    """
    Extracts coordinates from a KML file, converts them to UTM coordinates,
    and calculates their relative positions to a given reference point.

    Parameters
    ----------
    kml_path : str
        The file path to the KML file.
    ref_lat : float
        The reference latitude for relative position calculation.
    ref_lon : float
        The reference longitude for relative position calculation.

    Returns
    -------
    df_coordinates: pd.DataFrame
        DataFrame containing latitude, longitude, easting, northing,
        relative easting, and relative northing.
    ref_easting : float
        Reference easting coordinate.
    ref_northing : float
        Reference northing coordinate.

    """

    # Load and parse the KML file
    tree = ET.parse(kml_path)
    root = tree.getroot()

    # Define namespaces to find coordinates in KML
    namespaces = {"kml": "http://www.opengis.net/kml/2.2"}

    # Extract coordinates from the KML file
    coordinates = []
    for placemark in root.findall(".//kml:Placemark", namespaces):
        for coord in placemark.findall(".//kml:coordinates", namespaces):
            # Split coordinates (lon, lat, alt) and take the first two (lon, lat)
            coords = coord.text.strip().split()
            for c in coords:
                lon, lat, _ = map(float, c.split(","))
                coordinates.append((lat, lon))

    # Create a DataFrame with the extracted coordinates
    df_coordinates = pd.DataFrame(coordinates, columns=["latitude", "longitude"])

    # Initialize UTM converter
    proj_utm = pyproj.Proj(proj="utm", zone=6, datum="WGS84")

    # Convert reference coordinates to UTM
    ref_easting, ref_northing = proj_utm(ref_lon, ref_lat)

    # Convert all coordinates to UTM
    utm_coords = df_coordinates.apply(
        lambda row: proj_utm(row["longitude"], row["latitude"]), axis=1
    )
    df_coordinates["easting"] = [x[0] for x in utm_coords]
    df_coordinates["northing"] = [x[1] for x in utm_coords]

    # Calculate relative coordinates in meters
    df_coordinates["rel_easting"] = df_coordinates["easting"] - ref_easting
    df_coordinates["rel_northing"] = df_coordinates["northing"] - ref_northing

    return df_coordinates, ref_easting, ref_northing


def process_gimbal_config_to_relative_coords(gimbal_config_path, ref_lat, ref_lon):
    """
    Extracts coordinates from a gimbal configuration file, converts them to UTM coordinates,
    and calculates their relative positions to a given reference point.

    Parameters
    ----------
    gimbal_config_path: str
        The file path to the gimbal configuration file (csv)

    ref_lat: float
        The reference latitude for relative position calculation.

    ref_lon: float


    Returns
    -------
    df_coordinates: pd.DataFrame
        DataFrame containing latitude, longitude, easting, northing,
        relative easting, and relative northing.
    ref_easting: float
        Reference easting coordinate.
    ref_northing: float
        Reference northing coordinate.

    """

    # Load the gimbal configuration file
    df_gimbal = create_gimbal_config_df(gimbal_config_path)

    # Initialize UTM converter
    proj_utm = pyproj.Proj(proj="utm", zone=6, datum="WGS84")

    # Convert reference coordinates to UTM
    ref_easting, ref_northing = proj_utm(ref_lon, ref_lat)

    # Convert all coordinates to UTM
    utm_coords = df_gimbal.apply(lambda row: proj_utm(row["lon"], row["lat"]), axis=1)
    df_gimbal["easting"] = [x[0] for x in utm_coords]
    df_gimbal["northing"] = [x[1] for x in utm_coords]

    # Calculate relative coordinates in meters
    df_gimbal["rel_easting"] = df_gimbal["easting"] - ref_easting
    df_gimbal["rel_northing"] = df_gimbal["northing"] - ref_northing

    return df_gimbal, ref_easting, ref_northing


def create_gimbal_config_df(gimbal_config_path):
    df_gimbal = pd.read_csv(gimbal_config_path, skiprows=1)
    df_gimbal.dropna(subset=["lat", "lon"], inplace=True)
    # drop if the name is "Telescope"
    df_gimbal = df_gimbal[df_gimbal["name"] != "Telescope"]
    return df_gimbal


def create_lake_boundary(coords):
    """
    Creates a polygon representing the lake boundary.

    Parameters
    ----------
    coords : pd.DataFrame
        DataFrame containing the easting and northing of the lake boundary points.

    Returns
    -------
    Polygon
        Polygon object representing the lake boundary.
    """
    points = list(zip(coords["easting"], coords["northing"]))
    lake_boundary = Polygon(points)

    # Calculate the area of the lake using convex hull
    lake_area = MultiPoint(points).convex_hull.area

    return lake_boundary, lake_area


def generate_grid(lake_boundary, spacing):
    """
    Generates a grid of points within the bounding box of the lake boundary.

    Parameters
    ----------
    lake_boundary : Polygon
        Polygon object representing the lake boundary.
    spacing : float
        Spacing between grid points in meters.

    Returns
    -------
    np.ndarray
        Array of grid points.
    """
    min_x, min_y, max_x, max_y = lake_boundary.bounds
    x_coords = np.arange(min_x, max_x, spacing)
    y_coords = np.arange(min_y, max_y, spacing)
    grid_points = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    return grid_points


def filter_points_within_lake(grid_points, lake_boundary):
    """
    Filters grid points to keep only those within the lake boundary.

    Parameters
    ----------
    grid_points : np.ndarray
        Array of grid points.
    lake_boundary : Polygon
        Polygon object representing the lake boundary.

    Returns
    -------
    np.ndarray
        Array of points within the lake.
    """
    lake_points = [
        point for point in grid_points if lake_boundary.contains(Point(point))
    ]
    return np.array(lake_points)


def convert_to_relative(grid_points, ref_easting, ref_northing):
    """
    Converts UTM coordinates to relative coordinates.

    Parameters
    ----------
    grid_points : np.ndarray
        Array of grid points in UTM coordinates.
    ref_easting : float
        Reference easting coordinate.
    ref_northing : float
        Reference northing coordinate.

    Returns
    -------
    np.ndarray
        Array of points in relative coordinates.
    """
    rel_points = np.array(
        [[point[0] - ref_easting, point[1] - ref_northing] for point in grid_points]
    )
    return rel_points
