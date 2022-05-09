# Utilities supporting gaze analysis

import numpy as np
import pandas as pd
import yaml
import copy
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


def filter_list(lst, idx):
    """Convenience function to select items from a list with a binary index"""
    return [x for x, i in zip(lst, idx) if i]


def filter_arraydict(arraydict, idx):
    """Apply the same index to all fields in a dict of arrays"""
    dictlist = arraydict_to_dictlist(arraydict)
    dictlist = filter_list(dictlist, idx)
    out = dictlist_to_arraydict(dictlist)
    return out


def stack_arraydicts(*inputs, sort_key=None):
    output = arraydict_to_dictlist(inputs[0])
    for arrdict in inputs[1:]:
        arrlist = arraydict_to_dictlist(arrdict)
        output.extend(arrlist)
    if sort_key is not None:
        output = sorted(output, key=lambda x: x[sort_key])
    output = dictlist_to_arraydict(output)
    return output


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


def get_function(function_name):
    """Load a function to a variable by name

    Parameters
    ----------
    function_name : str
        string name for function (including module)
    """
    if callable(function_name):
        return function_name
    import importlib
    fn_path = function_name.split('.')
    module_name = '.'.join(fn_path[:-1])
    fn_name = fn_path[-1]
    module = importlib.import_module(module_name)
    func = getattr(module, fn_name)
    return func


def _check_dict_list(dict_list, n=1, **kwargs):
    tmp = dict_list
    for k, v in kwargs.items():
        tmp = [x for x in tmp if (hasattr(x, k)) and (getattr(x, k) == v)]
    if n is None:
        return tmp
    if len(tmp) == n:
        if n == 1:
            return tmp[0]
        else:
            return tmp
    else:
        raise ValueError('Requested number of items not found')

def load_pipeline_elements(session,
                           pupil_param_tag='plab_default',
                           pupil_drift_param_tag=None,
                           cal_marker_param_tag='circles_halfres',
                           cal_marker_filter_param_tag='cluster_default',
                           calib_param_tag='monocular_tps_default',
                           calibration_epoch=0,
                           val_marker_param_tag='checkerboard_halfres',
                           val_marker_filter_param_tag='basic_split',
                           mapping_param_tag='default_mapper',
                           error_param_tag='smooth_tps_default',
                           dbi=None,
                           is_verbose=True,
                           ):
    if dbi is None:
        dbi = session.dbi
    verbosity = copy.copy(dbi.is_verbose)
    dbi.is_verbose = is_verbose >= 1
    
    # Get all documents associated with session
    session_docs = dbi.query(session=session._id)
    # Create outputs dict
    outputs = dict(session=session)
    
    if pupil_param_tag is not None:
        outputs['pupil'] = {}
        for eye in ['left', 'right']:
            try:
                print("> Searching for %s pupil (%s)" % (eye, pupil_param_tag))
                outputs['pupil'][eye] = _check_dict_list(session_docs, 
                                                         n=1,
                                                         type='PupilDetection', 
                                                         tag=pupil_param_tag, 
                                                         eye=eye)
                print(">> FOUND %s pupil" % (eye))
            except:
                print('>> NOT found')

    if cal_marker_param_tag is not None:
        try:
            print("> Searching for calibration markers...")
            outputs['calibration_marker_all'] = _check_dict_list(
                session_docs, n=1, tag=cal_marker_param_tag, epoch='all')
            print(">> FOUND it")
        except:
            print('>> NOT found')

    if cal_marker_filter_param_tag is not None:
        try:
            print("> Searching for filtered calibration markers...")
            cfiltered_tag = '-'.join([cal_marker_param_tag,
                                      cal_marker_filter_param_tag])
            outputs['calibration_marker_filtered'] = _check_dict_list(
                session_docs, n=1, tag=cfiltered_tag, epoch=calibration_epoch)
            print(">> FOUND it")
        except:
            print('>> NOT found')

    if val_marker_param_tag is not None:
        try:
            print("> Searching for validation markers...")
            outputs['validation_marker_all'] = _check_dict_list(
                session_docs, n=1, tag=val_marker_param_tag, epoch='all')
            print(">> FOUND it")
        except:
            print('>> NOT found')

    if val_marker_filter_param_tag is not None:
        try:
            print("> Searching for filtered validation markers...")
            vfiltered_tag = '-'.join([val_marker_param_tag,
                                    val_marker_filter_param_tag])
            tmp = _check_dict_list(session_docs, n=None, tag=vfiltered_tag)
            if len(tmp) == 0:
                1/0  # error out, nothing found
            tmp = sorted(tmp, key=lambda x: x.epoch)
            outputs['validation_marker_filtered'] = tmp
            print(">> FOUND %d" % (len(tmp)))
        except:
            print(">> NOT found")

    if calib_param_tag is not None:
        if 'monocular' in calib_param_tag:
            eyes = ['left', 'right']
        else:
            eyes = ['both']

        for ie, eye in enumerate(eyes):
            if ie == 0:
                outputs['calibration'] = {}
                outputs['gaze'] = {}
                outputs['error'] = {}
            try:
                print("> Searching for %s calibration" % eye)
                calib_tag_full = '-'.join([pupil_param_tag, 
                                           cal_marker_param_tag, 
                                           cal_marker_filter_param_tag,
                                           calib_param_tag])
                outputs['calibration'][eye] = _check_dict_list(session_docs, 
                    n=1,
                    type='Calibration',
                    tag=calib_tag_full,
                    eye=eye,
                    epoch=calibration_epoch)
                print(">> FOUND %s calibration" % eye)
            except:
                print('>> NOT found')
            try:
                print("> Searching for %s gaze" % eye)
                gaze_tag_full = '-'.join([pupil_param_tag,
                                          cal_marker_param_tag,
                                          cal_marker_filter_param_tag,
                                          calib_param_tag,
                                          mapping_param_tag,
                                          ])
                outputs['gaze'][eye] = _check_dict_list(session_docs, n=1, 
                    type='Gaze', tag=gaze_tag_full, eye=eye)
                print(">> FOUND %s gaze" % eye)
            except:
                print('>> NOT found')

            try:
                print("> Searching for error")
                err_tags = [pupil_param_tag, cal_marker_param_tag, cal_marker_filter_param_tag,
                            calib_param_tag, mapping_param_tag, val_marker_param_tag, val_marker_filter_param_tag, error_param_tag]
                # Skip any steps not provided? Likely to cause bugs below
                err_tag = '-'.join(err_tags)
                err = _check_dict_list(session_docs, n=None,
                                        tag=err_tag, eye=eye)
                err = sorted(err, key=lambda x: x.epoch)
                outputs['error'][eye] = err
            except:
                print(">> NO error found for %s"%eye)

    for field in ['pupil', 'calibration', 'gaze']:
        if len(outputs[field]) == 0:
            _ = outputs.pop(field)

    if 'error' in outputs:
        if 'left' in outputs['error']:
            n_err_left = len(outputs['error']['left'])
        else:
            n_err_left = 0
        if 'right' in outputs['error']:
            n_err_right = len(outputs['error']['right'])
        else:
            n_err_right = 0
        if (n_err_left != n_err_right):
            print("Error mismatch: %d on left, %d on right" %
              (n_err_left, n_err_right))
            _ = outputs.pop("error")

    dbi.is_verbose = verbosity

    return outputs
