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
from VRPM_functions import *  # TODO: Fix this import

# %%
# Set up options
plt.close('all')
matplotlib.use('macosx')
matplotlib.style.use('eli_default')
matplotlib.rcParams.update({'font.size': 16})
warnings.filterwarnings("ignore")

sources, retros = read_measurement_geometry('geometry_config.yaml')

# TODO: move this into a config file
origin = [0, 0, 230]

u = 1
u_dir = 0
stability = 'A'

lg = np.array(np.sqrt(retros.x ** 2 + retros.y ** 2))  # approx just for testing

# plot_simulation_domain(sources, retros, origin, draw_beams=True, plot_3d=True)

# %%

store_df = pd.DataFrame(columns=['u', 'dir', 'stability', 'estimates', 'C', 'C_max'])

pics, sim_info = get_synthetic_measurement(retros,
                                           sources,
                                           origin,
                                           u_mag=u,
                                           u_dir=u_dir,
                                           stability=stability,
                                           plot=False,
                                           )

# pic_store[i, :] = pics
avg_conc = pics / lg

def create_beam_info(sim_info):
    beam_info = pd.DataFrame(sim_info['beam_info_list'])
    beam_info['max_at_end'] = beam_info.end_of_beam_concentration == beam_info.max_local_concentration
    beam_info['retro_max_ratio'] = beam_info.max_local_concentration / beam_info.end_of_beam_concentration

    return beam_info
def fiter_beam_info(beam_info):
    beam_info = beam_info[beam_info.end_of_beam_concentration > 1e-4]
    beam_info = beam_info[beam_info.max_local_concentration > 1e-4]

    return beam_info

beam_info = create_beam_info(sim_info)

retros['measurement'] = avg_conc
retros['x_rel'] = retros.x - sources.x.mean()
retros['y_rel'] = retros.y - sources.y.mean()

summary = pd.merge(retros, beam_info, how='outer', left_index=True, right_index=True)

sns.boxplot(data=summary, x='z', hue='y', y='measurement')
plt.figure()
sns.boxplot(data=summary, x='z', hue='y', y='max_local_concentration')

#%%
# lot curves of concentration vs z for each retro. Use groupby to plot each x,y, pair as a line
plt.figure()
sns.boxplot(x='z', y='measurement', data=retros)



# retros_filtered = filter_retros(retros)
# visualize_enhanced_retros(retros, sources)
# plot_retro_z_profiles(retros)



#%%

store_df = store_df.append({'u': u, 'dir': u_dir, 'stability': stability, 'estimates': avg_conc},
                           ignore_index=True)

if False:
    a = plt.gcf().get_axes()[0]
    visualize_bivarpiiate(results[i, :], ax=a)

store_df['maximum_concentration'] = store_df.estimates.apply(lambda x: np.nanmax(x))
store_df['average_concentration'] = store_df.estimates.apply(lambda x: np.nanmean(x))

# avg_conc = pic_store / lg
# avg_conc = avg_conc[0]  # TODO fix this indexing issue
# %%
# Plot everything.
# TODO: Wrap plotting code into function

plt.figure()

plt.axis('equal')

# Create grid for concentration visualization
side_length = 500
slice_height = .50
X, Y = np.meshgrid(
    np.linspace(sources.x.min() - 1000, retros.x.max(), num=side_length),
    np.linspace(origin[1], retros.y.max() + 100, num=side_length)
)
Z = slice_height + np.zeros_like(X)
C = np.zeros_like(X.ravel())

test_pts = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T

for i in range(len(sources)):
    C += general_concentration(test_pts,
                               sources.iloc[i, 0:3],
                               sources.strength.iloc[i],
                               u,
                               u_dir,
                               stability
                               )

plt.pcolormesh(X.reshape(side_length, side_length), Y.reshape(side_length, side_length),
               C.reshape(side_length, side_length),
               alpha=.9,
               cmap=cm.lajolla,
               norm=colors.LogNorm(vmin=1e-4, vmax=10),
               )

plt.colorbar(cmap=cm.lajolla, label='PPM enhancement at %.1f meters AGL' % slice_height)

plt.scatter(origin[0], origin[1], c='orange', label='Laser')

# plt.scatter(retros.x, retros.y, c='red', label='Retros', s=pic_store[0,:] / np.max(pic_store) * 50 )

plt.scatter(sources.x, sources.y, c='blue', label='Big Trail Lake')
plt.plot(sources.x.mean() + (np.cos(u_dir) * np.linspace(0, 1000)),
         sources.y.mean() + (np.sin(u_dir) * np.linspace(0, 1000)),
         ':k',
         label='Wind Direction')
plt.legend()

sns.scatterplot(x='x',
                y='y',
                data=retros,
                hue=pics,
                # size=pic_store[0, :],
                legend=False)

ppm_threshold = 1e-3

for i in range((avg_conc > ppm_threshold).sum()):
    x = retros.x[avg_conc > ppm_threshold].iloc[i]
    y = retros.y[avg_conc > ppm_threshold].iloc[i]
    s = str(np.round(1e3 * avg_conc[avg_conc > ppm_threshold][i], 1)) + ' ppb'
    # plt.text(x=x-10, y=y+10, s=s, fontdict={'fontsize': 'xx-small'})
    plt.plot([origin[0], x], [origin[1], y], color='darkgreen', alpha=.95)

plt.ylabel('Northing (m)')
plt.xlabel('Easting (m)')
# plt.ylim(bottom=0)
# plt.xlim(right=0)
plt.title('Wind: %.2f m/s, Stability: \"%s\"' % (u, stability))

plt.figure()
sns.scatterplot(data=store_df, x='u', y='maximum_concentration', s=100, hue='stability')
plt.ylabel('Maximum Path-integrated concentration [ppm]')
plt.xlabel('U [m/s]')
plt.title('10x BTL Mean Emission Scenario')
# %%

plt.figure()
plt.plot(x_synthetic, avg_conc.reshape((10, 4))[:, 0])
plt.plot(x_synthetic, avg_conc.reshape((10, 4))[:, 1])
plt.ylabel('Concentretaion measurement [ppm]')
plt.xlabel('distance from emissions center [m]')
plt.xlabel('distance from emissions center [m]')
plt.title('Near-Feature Measurements ')
C.max()
plt.text(30, .0025, 'Maximum .5m concentration: %.2f ppm' % C.max())