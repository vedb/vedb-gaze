# Utilities supporting gaze analysis

import numpy as np
import pandas as pd
import thinplate as tps  # library here: http://github.com/cheind/py-thin-plate-spline
from scipy import interpolate
import yaml
import os



def read_pl_gaze_csv(session_folder, output_id):
    sub_directory = str(output_id) * 3
    csv_file_name = os.path.join(
        session_folder, 'exports', sub_directory, "gaze_positions.csv")
    print("CSV File Name: ", csv_file_name)
    return pd.read_csv(csv_file_name)


def read_yaml(parameters_fpath):
    """Thin wrapper to safely read a yaml file into a dictionary"""
    param_dict = dict()
    with open(parameters_fpath,"r") as fid:
        param_dict = yaml.safe_load(fid)
    return param_dict

def write_yaml(parameters, fpath):
    """Thin wrapper to write a dictionary to a yaml file"""
    with open(fpath, mode='w') as fid:
        yaml.dump(pd.params, fid)


def filter_list(lst, idx):
    """Convenience function to select items from a list with a binary index"""
    return [x for x, i in zip(lst, idx) if i]


def stack_arraydicts(*inputs, sort_key=None):
    output = arraydict_to_dictlist(inputs[0])
    for arrdict in inputs[1:]:
        arrlist = arraydict_to_dictlist(arrdict)
        output.extend(arrlist)
    if sort_key is not None:
        output = sorted(output, key=lambda x: x[sort_key])
    output = dictlist_to_arraydict(output)
    return output


def unique(seq, idfun=None):
    """Returns only unique values in a list (with order preserved).
    (idfun can be defined to select particular values??)
    
    Stolen from the internets 11.29.11
    
    Parameters
    ----------
    seq : TYPE
        Description
    idfun : None, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen:
            seen[marker] += 1
            continue
        else:
            seen[marker] = 1
            result.append(item)
    return result, seen


def match_time_points(*data, fn=np.median, window=None):
    """Compute gaze position across matched time points
    
    Currently selects all gaze points within half a video frame of 
    the target time (first data timestamp field) and takes median
    of those values. 
    
    NOTE: This is messy. computing median doesn't work for fields of 
    data that are e.g. dictionaries. These must be removed before 
    calling this function for now. 
    """
    if window is None:
        # Overwite any function argument if window is set to none;
        # this will do nearest-frame resampling
        def fn(x, axis=None):
            return x
    # Timestamps for first input are used as a reference
    reference_time = data[0]['timestamp']
    # Preallocate output list
    output = []
    # Loop over all subsequent fields of data
    for d in data[1:]:
        t = d['timestamp'].copy()
        new_dict = dict(timestamp=reference_time)
        # Loop over all timestamps in time reference
        for i, frame_time in enumerate(reference_time):
            # Preallocate lists
            if i == 0:
                for k, v in d.items():
                    if k in new_dict:
                        continue
                    shape = v.shape
                    new_dict[k] = np.zeros(
                        (len(reference_time),) + shape[1:], dtype=v.dtype)
            if window is None:
                # Nearest frame selection
                fr = np.argmin(np.abs(t - frame_time))
                time_index = np.zeros_like(t) > 0
                time_index[fr] = True
            else:
                # Selection of all frames within window
                time_index = np.abs(t - frame_time) < window
            # Loop over fields of inputs
            for k, v in d.items():
                if k == 'timestamp':
                    continue
                try:
                    frame = fn(v[time_index], axis=0)
                    new_dict[k][i] = frame
                except:
                    # Field does not support indexing of this kind;
                    # This should probably raise a warning at least...
                    pass
        # Remove any keys with all fields deleted
        keys = list(d.keys())
        for k in keys:
            if len(new_dict[k]) == 0:
                _ = new_dict.pop(k)
            else:
                new_dict[k] = np.asarray(new_dict[k])
        output.append(new_dict)
    # Flexible output, depending on number of inputs
    if len(output) == 1:
        return output[0]
    else:
        return tuple(output)


def onoff_from_binary(data, return_duration=True):
    """Converts a binary variable data into onsets, offsets, and optionally durations
    
    This may yield unexpected behavior if the first value of `data` is true.
    
    Parameters
    ----------
    data : array-like, 1D
        binary array from which onsets and offsets should be extracted
    return_duration : bool, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    
    """
    if data[0]:
        start_value = 1
    else:
        start_value = 0
    data = data.astype(np.float).copy()

    ddata = np.hstack([[start_value], np.diff(data)])
    (onsets,) = np.nonzero(ddata > 0)
    # print(onsets)
    (offsets,) = np.nonzero(ddata < 0)
    # print(offsets)
    onset_first = onsets[0] < offsets[0]
    len(onsets) == len(offsets)

    on_at_end = False
    on_at_start = False
    if onset_first:
        if len(onsets) > len(offsets):
            offsets = np.hstack([offsets, [-1]])
            on_at_end = True
    else:
        if len(offsets) > len(onsets):
            onsets = np.hstack([-1, offsets])
            on_at_start = True
    onoff = np.vstack([onsets, offsets])
    if return_duration:
        duration = offsets - onsets
        if on_at_end:
            duration[-1] = len(data) - onsets[-1]
        if on_at_start:
            duration[0] = offsets[0] - 0
        onoff = np.vstack([onoff, duration])

    onoff = onoff.T.astype(np.int)
    return onoff


def onoff_to_binary(onoff, length):
    """Convert (onset, offset) tuples to binary index
    
    Parameters
    ----------
    onoff : list of tuples
        Each tuple is (onset_index, offset_index, [duration_in_frames]) for some event
    length : total length of output vector
        Scalar value for length of output binary index
    
    Returns
    -------
    index
        boolean index vector
    """
    index = np.zeros(length,)
    for on, off in onoff[:, :2]:
        index[on:off] = 1
    return index > 0


def time_to_index(onsets_offsets, timeline):
    """find indices between onsets & offsets in timeline

    Parameters
    ----------
    """
    out = np.zeros_like(onsets_offsets)
    for ct, (on, off) in enumerate(onsets_offsets):
        i = np.flatnonzero(timeline > on)[0]
        j = np.flatnonzero(timeline < off)[-1]
        out[ct] = [i, j]
    return out


def dictlist_to_arraydict(dictlist):
    """Convert from pupil format list of dicts to dict of arrays"""
    dict_fields = list(dictlist[0].keys())
    out = {}
    for df in dict_fields:
        out[df] = np.array([d[df] for d in dictlist])
    return out


def arraydict_to_dictlist(arraydict):
    """Convert from dict of arrays to pupil format list of dicts"""
    dict_fields = list(arraydict.keys())
    first_key = dict_fields[0]
    n = len(arraydict[first_key])
    out = []
    for j in range(n):
        frame_dict = {}
        for k in dict_fields:
            value = arraydict[k][j]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            frame_dict[k] = value
        out.append(frame_dict)
    return out


def compute_error(marker_pos,
                  gaze_left,
                  gaze_right,
                  method='tps',
                  error_smoothing_kernels=None,
                  vhres=None,
                  lambd=0.001,
                  extrapolate=False,
                  confidence_threshold=0,
                  image_resolution=(2048, 1536),
                  degrees_horiz=125,
                  degrees_vert=111,):
    """Compute error at set points and interpolation between those points

    Parameters
    ----------
    marker : array
        estimated marker position and confidence; needs field 'norm_pos' only ()

    """
    # Pixels per degree, coarse estimate for error computation
    # Default degrees are 125 x 111, this assumes all data is collected
    # w/ standard size, which is not true given new lenses (defaults must be
    # updated for new lenses)
    hppd = image_resolution[0] / degrees_horiz
    vppd = image_resolution[1] / degrees_vert
    # Coarse, so it goes
    ppd = np.mean([vppd, hppd])

    # < This section will vary with input format >
    # Estimated gaze position, in normalized (0-1) coordinates
    gl = gaze_left['norm_pos']
    # Gaze left - confidence index (gl_ci)
    gl_ci = gaze_left['confidence'] > confidence_threshold
    gr = gaze_right['norm_pos']
    # Gaze right - confidence index (gl_ci)
    gr_ci = gaze_right['confidence'] > confidence_threshold
    # < To here>

    vp_image = marker_pos * image_resolution
    gl_image = gl * image_resolution
    gr_image = gr * image_resolution

    err_left = np.linalg.norm(gl_image[gl_ci] - vp_image[gl_ci], axis=1) / ppd
    err_right = np.linalg.norm(gr_image[gr_ci] - vp_image[gr_ci], axis=1) / ppd
    if vhres is None:
        hres, vres = (np.array(image_resolution) * 0.25).astype(np.int)
    else:
        vres, hres = vhres
    # Interpolate to get error over whole image
    vpix = np.linspace(0, 1, vres)
    hpix = np.linspace(0, 1, hres)
    xg, yg = np.meshgrid(hpix, vpix)
    # Grid interpolation, basic
    if gl_ci.sum() == 0:
        tmp_l = np.ones_like(xg) * np.nan
    else:
        tmp_l = interpolate.griddata(vp[gl_ci], np.nan_to_num(
            err_left, nan=np.nanmean(err_left)), (xg, yg), method='cubic', fill_value=np.nan)
    if gr_ci.sum() == 0:
        tmp_r = np.ones_like(xg) * np.nan
    else:
        tmp_r = interpolate.griddata(vp[gr_ci], np.nan_to_num(err_right, nan=np.nanmean(
            err_right)), (xg, yg), method='cubic', fill_value=np.nan)

    if method == 'griddata':
        err_left_image = tmp_l
        err_right_image = tmp_r
        if error_smoothing_kernels is not None:
            tmp_l = np.nan_to_num(err_left_image, nan=np.nanmax(err_left))
            tmp_r = np.nan_to_num(err_right_image, nan=np.nanmax(err_right))
            tmp_l = cv2.blur(tmp_l, error_smoothing_kernels)
            tmp_r = cv2.blur(tmp_r, error_smoothing_kernels)
            tmp_l[np.isnan(err_left_image)] = np.nan
            tmp_r[np.isnan(err_right_image)] = np.nan
            err_left_image = tmp_l
            err_right_image = tmp_r

    elif method == 'tps':
        x, y = marker_pos.T
        # Left
        to_fit_l = np.vstack([x[gl_ci], y[gl_ci], err_left]).T
        theta_l = tps.TPS.fit(to_fit_l, lambd=lambd)
        err_left_image = tps.TPS.z(np.vstack(
            [xg.flatten(), yg.flatten()]).T, to_fit_l, theta_l).reshape(*xg.shape)
        # Right
        to_fit_r = np.vstack([x[gr_ci], y[gr_ci], err_right]).T
        theta_r = tps.TPS.fit(to_fit_r, lambd=lambd)
        err_right_image = tps.TPS.z(np.vstack(
            [xg.flatten(), yg.flatten()]).T, to_fit_r, theta_r).reshape(*xg.shape)
        if not extrapolate:
            err_left_image[np.isnan(tmp_l)] = np.nan
            err_right_image[np.isnan(tmp_r)] = np.nan

    return dict(left=err_left,
                right=err_right,
                left_image=err_left_image,
                right_image=err_right_image,
                left_marker_pos=marker_pos[gl_ci],
                right_marker_pos=marker_pos[gr_ci],
                xgrid=xg,
                ygrid=yg)
