#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:28:22 2022

@author: elimiller
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
import yaml
from scipy.optimize import minimize, Bounds


def line_points(start, end, num_pts=5000):
    '''start and end 3x1 array-like of xyz coords.'''
    x1, y1, z1 = start
    x2, y2, z2, = end

    t = np.linspace(0, 1, num=num_pts)
    xs = x1 + (x2 - x1) * t
    ys = y1 + (y2 - y1) * t
    zs = z1 + (z2 - z1) * t
    return np.array([xs, ys, zs]).T


def l_to_PG(l):
    '''Turns obukhov length into Pasquill-Gifford stability class based on table 1 from Munoz Esparaza
    Note that our function for dispersion coefficcients below does not have class G, so we will use F instead.
    Also this paper does not have a class E (slightly stable)... TODO: figure this out.
    l: obukhov length in m

    '''
    if 0 < l < 200:
        return 'F'
    elif 200 < l < 1000:
        return 'F'
    elif np.abs(l) > 1000:
        return 'D'
    elif -1000 < l < -200:
        return 'B'
    elif -200 < l < 0:
        return 'A'


def dispersion_coeffs(stability):
    '''
   Returns 
   '''
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
    return a, b, c, d, e, f


# def general_concentration(x,y,z, Q=1, u=1, stability="E"):
def general_concentration(beam_loc, source_loc, Q, u_mag, u_dir, stability="E"):
    '''Q [kg/s] emission rate and returns a concentration [ppm] at that point'''
    ''' With reflection. Page '''
    a, b, c, d, e, f = dispersion_coeffs(stability)
    x0, y0, z0, = source_loc

    x = beam_loc[:, 0]
    y = beam_loc[:, 1]
    z = beam_loc[:, 2]

    # Transform into global coordinates
    x_glob = x - x0
    y_glob = y - y0
    z_glob = z - z0  # Not necessary for ground-based sources

    # Figure out how much to rotate in the xy plane from GP formulation
    phi = np.arctan2(y_glob, x_glob)  # Angle of laser point from east axis from x0y0

    l_xy = np.sqrt(x_glob ** 2 + y_glob ** 2)
    # New coordinates
    x_prime = l_xy * np.cos(u_dir - phi)
    y_prime = l_xy * np.sin(u_dir - phi)

    sigma_y = a * x_prime / (1 + b * x_prime) ** c
    sigma_z = d * x_prime / (1 + e * x_prime) ** f

    # C = Q/(2*pi*sigma_y*sigma_z*u)  * exp(-y_prime**2/(2*sigma_y**2)) * (exp(-(z_glob)**2/(2*sigma_z**2))+exp(-(z_glob)**2/(2*sigma_z**2)))
    h = 0
    C = Q / (2 * np.pi * u_mag * sigma_y * sigma_z) * np.exp(-0.5 * y_prime ** 2 / sigma_y ** 2) * \
        (np.exp(-0.5 * (z - h) ** 2 / sigma_z ** 2) + np.exp(-0.5 * (z + h) ** 2 / sigma_z ** 2))

    C *= 24.45 / 16.04 * 1e6  # Convert from kg/m^3 to ppm

    C[x_prime < 0] = 0

    return C


def plume_contribution(beam_loc, source_loc, Q, u_mag, u_dir):
    ''' Returns: concentration [g/m^3] given plume and POI params.
    
    Inputs:
      beam_loc: x,y,z location of beam point. easting, northing, elev.= m]
      source_loc: x,y,z, location of source. Easting, northing, elevation
      Q: emmission Rate [g/s]
      u_mag: wind speed [m/s]
      u_dir: wind direction [angle from east axis, radians]
      
      Using plume formula from Elder 2021 Supplemental info (Bio. geo. chem....
                                                             
      Want to convert this fuction to more sophisticated dependent on stability '''

    K = 2.1e-0  # dispersion coefficient m^2/sec (STP). Could change for temperature
    # K = .21 #cm^2/s

    # x0 = source_loc[0] # Easting
    # y0 = source_loc[1] # Northing
    # z0 = source_loc[2] # Z

    x0, y0, z0, = source_loc

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

    l_xy = np.sqrt(x_glob ** 2 + y_glob ** 2)
    # New coordinates
    x_prime = l_xy * np.cos(u_dir - phi)
    y_prime = l_xy * np.sin(u_dir - phi)

    coeff = Q / (4 * np.pi * K * (x_prime))
    # coeff[coeff <  0 ] = 0

    term1 = -(y_prime) ** 2 / (4 * K * (x_prime) / u_mag)
    term2 = -(z - z0) ** 2 / (4 * K * (x_prime) / u_mag)

    Conc = coeff * np.exp(term1 + term2)

    Conc[Conc < 0] = 0  # don't allow Nan or zero values

    return Conc


def get_synthetic_measurement(retros, sources, origin, u_mag, u_dir, stability="A", pts_per_beam=10000, plot=True):
    # plume_contribution(beam_loc, source_loc, Q, u_mag, u_dir):
    if plot:
        fig, (ax, ax1) = plt.subplots(2, 1, sharex=True)

    pic = np.zeros((len(retros),))

    max_conc = 0
    max_beam_info = None
    beam_info_list = []

    for i_retro in tqdm.tqdm(range(len(retros))):

        beam_pts = line_points(origin, np.array(retros.iloc[i_retro, :]))
        segment_length = np.sqrt(np.sum((beam_pts[-1, :] - beam_pts[0, :]) ** 2)) / pts_per_beam

        conc_store = np.zeros((len(beam_pts), len(sources)))

        for i_source in range(len(sources)):
            conc_store[:, i_source] = general_concentration(  # plume_contribution(
                beam_pts,
                sources[['x', 'y', 'z']].iloc[i_source],
                sources['strength'].iloc[i_source],
                u_mag,
                u_dir,
                stability=stability)

        conc_resolved = np.sum(conc_store, axis=1)  # retain for visulaization / debugging
        pic[i_retro] = np.sum(conc_resolved * segment_length, axis=0)
        # See https://teesing.com/media/library/tools/understanding-units-of-measurement.pdf for units

        max_local_conc = np.max(conc_resolved)
        max_local_conc_location = beam_pts[np.argmax(conc_resolved)]
        end_of_beam_conc = conc_resolved[-1]

        beam_info_list.append({
            'retro_index': i_retro,
            'max_local_concentration': max_local_conc,
            'max_local_concentration_location': max_local_conc_location,
            'end_of_beam_concentration': end_of_beam_conc,
        })

        if np.max(conc_resolved) > max_conc:
            max_conc = np.max(conc_resolved)

            max_beam_info = {
                'retro_index': i_retro,
                'max_concentration': max_conc,
                'max_concentration_location': beam_pts[np.argmax(conc_resolved)],
            }

        if plot:
            ax1.scatter(beam_pts[:, 1], conc_resolved)
            color_path = ax.scatter(beam_pts[:, 1],
                                    beam_pts[:, 2],
                                    c=conc_resolved,
                                    zorder=0,
                                    vmin=0,
                                    vmax=max_conc
                                    )
            ax1.set_ylabel('ppm')
            ax.scatter(retros['y'].iloc[i_retro], retros['z'].iloc[i_retro], marker='x',
                       label=str(np.round(pic[i_retro], 2)) + 'ppm-m')
            ax.set_title('Wind: %.1f m/s, Direction: %.0f, Class: %s' % (u_mag, np.degrees(u_dir), stability))
            ax.plot([0, np.max(retros.y)], [21, 21], '--k')  #
    if plot:
        ax.legend()

    # Collect some info about the simulation into a dictionary:
    sim_info = {'u_mag': u_mag,
                'u_dir': u_dir,
                'stability': stability,
                'max_beam_info': max_beam_info,
                'beam_info_list': beam_info_list, }

    return pic, sim_info  # len = n_retros


def polar_gaussian2d_simplified(A, mu_y, sigma_y, sigma_z, r, theta):
    ''' equation 3 from https://pubs.acs.org/doi/pdf/10.1021/es0017108
        with assumptions ground-based emmissions, and no covariance between z&y
        Put r and theta later for similar structure with scipy minimize args
        NOTE: there is an error in the paper. Should be plus in the exponential...'''

    return A / (2 * np.pi * sigma_y * sigma_z) * np.exp(
        -1 / 2 * (((r * np.cos(theta) - mu_y) ** 2 / sigma_y ** 2) + ((r * np.sin(theta)) ** 2 / sigma_z ** 2)))


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

    thetas = np.array(np.arctan2(offset.z, offset.y))  # Assume angle in the y-z plane [radians]

    r_beam = np.linspace(0, rs, pts_per_beam)

    pic_candidate = np.zeros((len(rs),))

    for i in range(len(rs)):
        conc_along_beam = polar_gaussian2d_simplified(A, mu_y, sigma_y, sigma_z, r_beam[:, i], thetas[i])
        pic_candidate[i] = np.sum(conc_along_beam * rs[i] / pts_per_beam)

    SSE = np.sum((pics - pic_candidate) ** 2)
    print('Candidate PICS: ' + str(pic_candidate))
    # print('SSE = ' + str(SSE))
    return SSE


def fit_bivariate_gauss(pics, u_mag, u_dir, retros, origin, plot=False, verbose=False, x0=None, bnds=None):
    if x0 is None:  # Give standard initial guess.
        x0 = [5 * pics[2], 1500, 10, 2.6]
    else:
        pass

    # TODO: Move initial guess outside of this function and into main script
    if bnds is None:
        # bnds = Bounds((0, 14001400, -np.inf, -np.inf), (1e9, 2000, np.inf, np.inf))
        bnds = Bounds((-np.inf, -np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf))
    else:
        pass

    # result = minimize(new_objective_fun, x0=x0, args=(retros, origin, pics), bounds=bnds ,method='L-BFGS-B')
    result = minimize(new_objective_fun,
                      x0=x0,
                      args=(retros, origin, pics),
                      bounds=bnds,
                      method='Nelder-Mead',
                      options={'maxiter': 5000},
                      )

    result.x[1] += origin[1]  # add back in origin y component

    # result = basinhopping(new_objective_fun, x0=x0, minimizer_kwargs = {"args": (retros, origin, pics)})

    print(result)
    if not result.success:
        print('Optimization Failed')

    ### most of this is for plotting. Main goal is to "integrate" in order to                                  
    # print(result)                                 
    A_fit, mu_y_fit, sigma_y_fit, sigma_z_fit = result.x

    flux_grid_pts = 500

    y_flux = np.linspace(mu_y_fit - 3 * sigma_y_fit, mu_y_fit + 3 * sigma_y_fit, flux_grid_pts)
    y_flux_rel = y_flux  # - origin[1]

    z_flux = np.linspace(0, 3 * sigma_z_fit, flux_grid_pts)
    z_flux_rel = z_flux  # - origin[2]

    Y_flux_rel, Z_flux_rel = np.meshgrid(y_flux_rel, z_flux_rel)

    Theta_flux = np.arctan2(Z_flux_rel, Y_flux_rel)
    R_flux = np.sqrt(Y_flux_rel ** 2 + Z_flux_rel ** 2)

    dr = R_flux[0, 1] - R_flux[0, 0]
    dtheta = Theta_flux[1, 0] - Theta_flux[0, 0]

    cell_areas = R_flux * dr * dtheta

    fit_conc = polar_gaussian2d_simplified(A_fit, mu_y_fit, sigma_y_fit, sigma_z_fit, R_flux, Theta_flux)

    area = (y_flux[-1] - y_flux[0]) * (z_flux[-1] - z_flux[0])
    fit_flux = 16.04 / (24.45 * 1000) * (fit_conc * cell_areas).sum() * u_mag * np.cos(u_dir)

    if plot:
        plt.pcolormesh(y_flux, z_flux, fit_conc)

    if verbose:
        print('integrated from fit_td')
        print(fit_flux)
        print('max path integrated concentration')
        print(np.max(pics))

    return fit_flux, result.x


def visualize_bivariate(inputs, ax=None):
    A_fit, mu_y_fit, sigma_y_fit, sigma_z_fit = inputs

    flux_grid_pts = 500

    y_flux = np.linspace(mu_y_fit - 3 * sigma_y_fit, mu_y_fit + 3 * sigma_y_fit, flux_grid_pts)
    y_flux_rel = y_flux  # - origin[1]

    z_flux = np.linspace(0, 3 * sigma_z_fit, flux_grid_pts)
    z_flux_rel = z_flux  # - origin[2]

    Y_flux_rel, Z_flux_rel = np.meshgrid(y_flux_rel, z_flux_rel)

    R_flux = np.sqrt(Y_flux_rel ** 2 + Z_flux_rel ** 2)
    Theta_flux = np.arctan2(Z_flux_rel, Y_flux_rel)

    fit_conc = polar_gaussian2d_simplified(A_fit, mu_y_fit, sigma_y_fit, sigma_z_fit, R_flux, Theta_flux)

    if ax is not None:
        CS = ax.contour(y_flux,
                        z_flux,
                        fit_conc,
                        colors='black',
                        label='Reconstruction'
                        )
        ax.clabel(CS, inline=True, fontsize=10)

    else:
        plt.contour(y_flux, z_flux, fit_conc)


def read_measurement_geometry(file_path):
    '''Reads in config file that defines source, retro, and tower locations. Returns dataframes of sources and retros
    in the appropriate coordinate scheme

    Assumes Q_Source_total in # g/m^2/year

    BUG: Right now if you have a single source or reto, it puts it at the bottom of the range, not the middle.
    See if I want to change this after using'''

    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    Q_source_total = config['Q_source_total']
    source_area = config['source_area']

    n_source_x = config['n_source_x']
    n_source_y = config['n_source_y']
    span_source_x = config['span_source_x']
    span_source_y = config['span_source_y']
    source_loc_y = config['source_loc_y']
    source_loc_x = config['source_loc_x']
    source_loc_z = config['source_loc_z']

    n_retro_x = config['n_retro_x']
    n_retro_y = config['n_retro_y']
    n_retro_z = config['n_retro_z']
    span_retro_x = config['span_retro_x']
    span_retro_y = config['span_retro_y']
    span_retro_z = config['span_retro_z']
    retro_loc_x = config['retro_loc_x']
    retro_loc_y = config['retro_loc_y']

    sources_x, sources_y = np.meshgrid(
        source_loc_x + np.linspace(span_source_x[0], span_source_x[1], n_source_x),
        source_loc_y + np.linspace(span_source_y[0], span_source_y[1], n_source_y)
    )

    Q_source = (Q_source_total * source_area) / len(sources_x.ravel())
    Q_source /= (1000 * 365 * 24 * 60 * 60)  # Convert to kg/s

    sources = pd.DataFrame({'x': sources_x.ravel(),
                            'y': sources_y.ravel(),
                            'z': source_loc_z,
                            'strength': Q_source})

    retros_x, retros_y, retros_z = np.meshgrid(retro_loc_x + np.linspace(span_retro_x[0], span_retro_x[1], n_retro_x),
                                               retro_loc_y + np.linspace(span_retro_y[0], span_retro_y[1], n_retro_y),
                                               np.linspace(span_retro_z[0], span_retro_z[1], n_retro_z)
                                               )

    retros = pd.DataFrame({'x': retros_x.ravel(),
                           'y': retros_y.ravel(),
                           'z': retros_z.ravel()})

    return sources, retros


def plot_simulation_domain(sources, retros, origin, draw_beams=False, plot_3d=True):
    fig = plt.figure(figsize=(8, 8))

    if plot_3d:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel('Z')
    else:
        ax = fig.add_subplot(111)
        ax.axis('equal')

    ax.scatter(sources.x, sources.y, sources.z, color='C0', label='Sources')
    ax.scatter(retros.x, retros.y, retros.z, color='C1', label='Retros')
    ax.scatter(origin[0], origin[1], origin[2], color='C2', label='Tower')

    if draw_beams:  # Plot lines between the origin and each retro
        for i in range(len(retros)):
            ax.plot([origin[0], retros.x.iloc[i]], [origin[1], retros.y.iloc[i]], [origin[2], retros.z.iloc[i]],
                    color='green', alpha=.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    # Make axes equal, repliating plt.axes('equal') for 3d projection: BUG: Not sure if this is working
    plt.show()


def filter_retros(retros, threshold=1e-3):
    return retros[retros.measurement > threshold]


def visualize_enhanced_retros(retros, sources, threshold=1e-3):
    retros_filtered = filter_retros(retros, threshold=threshold)

    plt.figure()
    sns.scatterplot(x='x', y='y', data=retros)
    sns.scatterplot(x='x', y='y', data=retros_filtered, color='red', marker='x')
    sns.scatterplot(x='x', y='y', data=sources, color='orange', marker='o')


def plot_retro_z_profiles(retros):
    # For each unique x,y, pair, plot the concentration vs z on the same plot

    plt.figure()
    for i, (x, y) in enumerate(retros.groupby(['x', 'y'])):
        sns.lineplot(x='z', y='measurement', data=y, label='%.0f, %.0f' % x)
