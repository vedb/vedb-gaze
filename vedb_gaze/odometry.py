# Odometry visualization
import vedb_gaze
import plot_utils
import file_io

import matplotlib.pyplot as plt
import numpy as np
import pathlib
import time

from scipy.signal import spectrogram, savgol_filter, find_peaks
from scipy import interpolate
from matplotlib.gridspec import GridSpec
import plot_utils

from .utils import dictlist_to_arraydict, get_frame_indices
from .visualization import angle_hist


def resample_data(in_time, in_data, out_time, method='scipy', **kwargs):
    # TODO handle confidence, nans
    if method=='scipy':
        out = interpolate.interp1d(in_time, in_data, axis=0)(out_time)
    return out

def load_odometry(folder,
                  resample=True,
                  smooth_filter_function=savgol_filter,
                  smooth_kwargs=None,
                  fps = 200,
                  verbose=False):
    if smooth_kwargs is None:
        if smooth_filter_function is savgol_filter:
            smooth_kwargs = dict(window_length=51,
                                 polyorder=2)
        else:
            smooth_kwargs = {}
    if 'axis' in smooth_kwargs:
        # Dont' allow, must be 0th (first) axis
        _ = smooth_kwargs.pop('axis')
    if verbose:
        print('Loading...')
    odo_timestamp_file = folder / 'odometry_timestamps_0start.npy'
    odo_file = folder / 'odometry.pldata'
    # Load
    t0 = time.time()
    odo_t = np.load(odo_timestamp_file)
    t1 = time.time()
    data = file_io.load_msgpack(odo_file)
    data_dict = dictlist_to_arraydict(data)
    # Remove extraneous fields
    _ = data_dict.pop('topic')
    _ = data_dict.pop('timestamp')
    _ = data_dict.pop('source_timestamp')
    t2 = time.time()
    elapsed1 = t1-t0
    elapsed2 = t2-t1
    data_fields = ['tracker_confidence', 'position', 'orientation', 'linear_velocity', 'angular_velocity', 'linear_acceleration', 'angular_acceleration']
    if verbose:
        print(f"Seconds elapsed loading:\n> Timestamps: {elapsed1:0.2f}\n> Data: {elapsed2:0.2f}")
    if resample:
        if verbose: 
            print('Resampling...')
        # Resample
        odo_t_rs = np.arange(odo_t[0], odo_t[-1], 1/fps)
        for field in data_fields:
            # TODO: handle nans
            #if np.ndim(v) == 1:
            #    to_resample = np.atleast_2d(v).T
            #else:
            #    to_resample = v            
            #keep_idx = ~np.all(isnan(to_resample), axis=1)
            #to_resample = to_resample[keep_idx]
            #data_dict[field] = interpolate.interp1d(odo_t, to_resample, kind='cubic', axis=0)(odo_t_rs)
            data_dict[field] = interpolate.interp1d(odo_t, data_dict[field], kind='cubic', axis=0)(odo_t_rs)
        odo_t = odo_t_rs
    # Find nans in data
    #nan_data = [np.isnan(v) for v in data_dict.values()]
    #nan_idx = np.hstack([np.atleast_2d(x).T if np.ndim(x) == 1 else x for x in nan_data] ).sum(1) > 0
    #for k in data_dict.keys():
    #    data_dict[k][nan_idx] = 0
    if smooth_filter_function is not None:
        if verbose:
            print('Smoothing...')
        # Smooth
        for field in data_fields:
            #1/0
            nan_idx = np.any(np.isnan(np.atleast_2d(data_dict[field])), axis=1)
            
            data_dict[field] = smooth_filter_function(data_dict[field], axis=0, **smooth_kwargs)
            if np.any(nan_idx):
                data_dict[field][nan_idx] = np.nan
    #for k in data_dict.keys():
    #    data_dict[k][nan_idx] = np.nan
    # Compute useful quantities
    data_dict['absolute_linear_velocity'] = np.linalg.norm(data_dict['linear_velocity'], axis=1)
    ori_ang = euler_from_quaternion(*data_dict['orientation'].T)
    data_dict['roll'], data_dict['pitch'], data_dict['yaw'] = ori_ang.T

    # Put time back
    data_dict['timestamp'] = odo_t
    return data_dict

def plot_odo_position(odo_pos, 
                      odo_ori=None,
                      do_3d=False,
                      quiver_sampling=200, 
                      quiver_kw=None,
                      ax=None):
    """Convention here is x is horizontal, y is front-to-back, z is up-and-down"""
    x, z, y = odo_pos.T
    do_quiver = odo_ori is not None
    if do_quiver:
        if len(odo_ori.shape) == 2:
            ori_ang = euler_from_quaternion(*odo_ori.T)
            pitch, roll, yaw = ori_ang.T
        else:
            yaw = odo_ori
        u = np.sin(np.radians(yaw)) 
        v = np.cos(np.radians(yaw)) 

    if ax is None:
        if do_3d:
            subplot_kw = dict(projection='3d')
        else:
            subplot_kw = None
        fig, ax = plt.subplots(
            subplot_kw=subplot_kw)
    else:
        fig = ax.figure
    xr, zr, yr = np.ptp(odo_pos, axis=0)
    max_range = np.max([xr, zr, yr])
    # Plot
    if do_3d:
        ax.plot(x, -y, z, color='gray')
        ax.scatter(x[0], y[0], z[0], color='g', marker='o')
        # optional time coloring
        ax.scatter(x[1], y[1], z[1], color='r', marker='s')
        if do_quiver:
            pass 
            # 3D version?
            # plt.quiver(x[::quiver_sampling],
            #            -y[::quiver_sampling],
            #            u[::quiver_sampling],
            #            v[::quiver_sampling],
            #            color=quiver_color)  # , scale_units='inches', scale=2)
    else:
        ax.plot(x, -y, color='gray')
        # optional time coloring
        ax.scatter(x[0], -y[0], color='g', marker='o')
        ax.scatter(x[-1], -y[-1], color='r', marker='s')
        if do_quiver:
            if quiver_kw is None:
                quiver_kw = {}
            ax.quiver(x[::quiver_sampling],
                       -y[::quiver_sampling],
                       u[::quiver_sampling],
                       v[::quiver_sampling],
                       **quiver_kw)

    ax.set_xlabel('X')
    adj = (max_range - xr) / 2
    ax.set_xlim([x.min() - adj, x.max() + adj])
    ax.set_ylabel('Y')
    adj = (max_range - yr) / 2
    ax.set_ylim([y.min() - adj, y.max() + adj])
    if do_3d:
        ax.set_zlabel('Z')
        adj = (max_range - zr) / 2
        ax.set_zlim([z.min() - adj, z.max() + adj])
    ax.axis('equal')
    return fig


###
def euler_from_quaternion(w, x, y, z):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        pitch is rotation around x (ear to ear) in radians (counterclockwise)
        roll is rotation around y (nose to occiput) in radians (counterclockwise)
        yaw is rotation around z (spine to crown) in radians (counterclockwise)
        """
        # Dont' ask. Realsense is idiotic.
        x_ = -z
        y_ = x
        z_ = -y
        x = x_
        y = y_
        z = z_
        pitch =  -np.arcsin(2.0 * (x*z - w*y)) * 180.0 / np.pi;
        roll  =  np.arctan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z) * 180.0 / np.pi;
        yaw   =  np.arctan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z) * 180.0 / np.pi;
        return np.vstack([pitch, roll, yaw]).T # in degrees



def convert_speed(mps, to='mins_per_mile'):
    """Convert meters per second to more intuitive units
    
    Parameters
    ----------
    mps : scalar or array
        velocity in meters per second
    to : str
        identifier for output, may be:
        'min / mile', 'mins per mile'
        or 
        'mph', 'miles per hour'
    """
    mps_per_mph = 2.237
    m_per_mi = 1609.34
    mi_per_m = m_per_mi ** -1
    sec_per_hr = 60 * 60
    mins_per_hr = 60
    sec_per_min = 60

    if to in ('min / mile', 'mins_per_mile', 'min / mile'):
        sec_per_m = mps **-1
        mins_per_mile = sec_per_m * m_per_mi * (sec_per_min**-1)
        out = mins_per_mile
    elif to in ('miles per hour', 'mph'):
        miles_per_hour = mps * sec_per_hr / m_per_mi
        out = miles_per_hour
    return out


def plot_at_times(tt, y, time_start, time_end, 
                  time_units='seconds', ax=None, **kwargs):
    """Simple timecourse plotting
    
    Plot a quantity `y` (that extends over a longer time) only from 
    `time_start` to `time_end`

    Parameters
    ----------

    """
    if ax is None:
        _, ax = plt.subplots()
    if time_units in ('seconds', 's'):
        multiplier = 1
    elif time_units in ('minutes', 'm'):
        multiplier = 60
    st, fin = get_frame_indices(time_start * multiplier, time_end * multiplier, tt)
    ax.plot(tt[st:fin] / multiplier, y[st:fin], **kwargs)



def plot_odometry_values(odometry_dict, 
    time_start=None,
    time_end=None,
    cmap_time='Oranges_r', 
    legend_kw=None,
    quiver_sampling=400,
    figsize=(12, 12),
    ):
    """Plot an overview of odometry for a session
    
    to save load time, odometry_session can be a list of loaded session odometry values:
    session, odo_t, odo_pos, roll, pitch, yaw, abs_velocity, odo_linv, odo_lina, odo_angv, odo_anga = session

    """
    # if isinstance(odometry_session, list):
    #     session, odo_t, odo_pos, roll, pitch, yaw, abs_velocity, odo_linv, odo_lina, odo_angv, odo_anga = odometry_session
    # else:
    #     odo_t, odo_pos, roll, pitch, yaw, abs_velocity, odo_linv, odo_lina, odo_angv, odo_anga =\
    #         load_odometry(odometry_session, resample=True, smooth=True, fps = 200)
    odo_t = odometry_dict['timestamp'].copy()
    odo_t = odo_t - odo_t[0]
    data_fields = ['position', 'roll', 'pitch', 'yaw', 'absolute_linear_velocity',\
                   'linear_velocity', 'linear_acceleration', 'angular_velocity', 'angular_acceleration']
    odo_pos, roll, pitch, yaw, abs_velocity, odo_linv, odo_lina, odo_angv, odo_anga = \
        tuple([odometry_dict[field] for field in data_fields])
    
    if legend_kw is None:
        legend_kw = dict(frameon=False, ncol=2, loc='upper left')
    if time_start is None:
        time_start = odo_t.min() / 60 # minutes #  2.9 #
    if time_end is None:
        time_end = odo_t.max() / 60 # minutes  # 4.1 # 
    time_ticks = np.arange(0, np.max(np.floor(odo_t / 60)) + 1)
    
    quiver_kw = dict(color='orange', scale=30)

    # layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(6, 3, figure=fig, )
    hist_axs = []
    for col in range(3):
        hist_axs.append(fig.add_subplot(gs[0,col], projection='polar'))
    timeline_ax0 = fig.add_subplot(gs[1, :])
    timeline_ax1 = fig.add_subplot(gs[2, :])
    timeline_ax2 = fig.add_subplot(gs[3, :])
    position_ax = fig.add_subplot(gs[4:,:])

    # Plot
    # histograms
    for ax, o, label in zip(hist_axs, (pitch, yaw, roll), ('pitch', 'yaw', 'roll')):
        # Hist plots
        angle_hist(o, ax=ax)
        if label in ('yaw', 'roll'):
            angle_hist(o, ax=hist_axs[0])
        ax.set_yticklabels([])
        ax.set_title(label)
        # Timeline plots
        #timeline_ax0.plot(odo_t / 60, o, label=label)
        plot_at_times(odo_t, o, time_start=time_start, time_end=time_end, time_units='m', 
                ax=timeline_ax0, alpha=0.7, label=label)

    ## Timelines
    # Orientation
    timeline_ax0.legend(**legend_kw)
    timeline_ax0.set_ylabel('Orientation')
    xl = timeline_ax0.get_xlim()
    timeline_ax0.set_xticks(time_ticks)
    timeline_ax0.grid(axis='x', ls=':')
    yl = timeline_ax0.get_ylim()
    ymin = yl[0] * np.ones_like(time_ticks)

    timeline_ax0.scatter(time_ticks, ymin, c=time_ticks, cmap=cmap_time)
    timeline_ax0.set_xlim(xl)


    # Angular velocity
    plot_at_times(odo_t, odo_angv, time_start=time_start, time_end=time_end, time_units='m', 
                ax=timeline_ax1, alpha=0.7)
    timeline_ax1.set_ylabel('Angular velocity')
    timeline_ax1.legend(['pitch','yaw','roll'], **legend_kw)
    yl = timeline_ax1.get_ylim()
    ymin = yl[0] * np.ones_like(time_ticks)
    xl = timeline_ax1.get_xlim()
    timeline_ax1.set_xticks(time_ticks)
    timeline_ax1.grid(axis='x', ls=':')
    timeline_ax1.scatter(time_ticks, ymin, c=time_ticks, cmap=cmap_time)
    timeline_ax1.set_xlim(xl)


    plot_at_times(odo_t, odo_linv, time_start=time_start, time_end=time_end, time_units='m', 
                ax=timeline_ax2, alpha=0.7)
    timeline_ax2.set_ylabel('Linear velocity')
    timeline_ax2.legend(['x','z','y'], **legend_kw)
    yl = timeline_ax2.get_ylim()
    ymin = yl[0] * np.ones_like(time_ticks)
    xl = timeline_ax2.get_xlim()
    timeline_ax2.set_xticks(time_ticks)
    timeline_ax2.grid(axis='x', ls=':')
    timeline_ax2.scatter(time_ticks, ymin, c=time_ticks, cmap=cmap_time)
    timeline_ax2.set_xlim(xl)
    _ = plot_odo_position(odo_pos, odo_ori=yaw, 
                        quiver_sampling=quiver_sampling, 
                        quiver_kw=quiver_kw,
                        ax=position_ax)
    time_idx = np.array([np.argmin(np.abs(ti - odo_t / 60)) for ti in time_ticks])
    position_ax.scatter(odo_pos[time_idx,0], -odo_pos[time_idx,2], c=time_ticks, cmap=cmap_time, zorder=7, edgecolor='k', lw=0.3)
    _ = position_ax.set_ylabel("Position (x, y)\n(meters)")
