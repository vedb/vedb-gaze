# Utilities supporting gaze analysis
"""General utilities supporting the vedb_gaze pipeline.

Provides:

- yaml reading / writing and small list helpers
- conversion between "arraydicts" (dict of arrays, e.g. 'timestamp',
  'norm_pos', 'confidence') and "dictlists" (list of per-sample dicts, the
  Pupil Labs format), plus filtering / stacking of arraydicts
- time handling: matching data across clocks (`match_time_points`),
  converting between binary labels, (onset, offset, duration) "onoff"
  arrays, and time ranges (`onoff_from_binary`, `onoff_to_binary`,
  `time_to_index`, `get_frame_indices`), and resampling (`resample_data`)
- construction of pipeline output filenames from processing tags
  (`make_file_strings`) and loading of pipeline outputs
  (`load_pipeline_elements`). Default tags come from the package config
  (`vedb_gaze.options.config`, 'defaults' section).
"""

import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.stats import zscore
import pathlib
import hashlib
import yaml
import copy
import os

from .options import config

defaults = {}
for k in ['pupil', 'eyelid', 'pupil_detrend', 
          'calibration_marker', 'calibration_split', 
          'calibration_cluster', 'validation_marker', 
          'validation_split', 'validation_cluster', 
          'calibration', 'gaze', 'error', ]:
    tmp = config.get('defaults', k)
    if tmp == 'None':
        defaults[k] = None
    else:
        defaults[k] = tmp

def read_pl_gaze_csv(session_folder, output_id):
    """Read a Pupil Player gaze export csv file

    Reads <session_folder>/exports/<sub_directory>/gaze_positions.csv,
    where sub_directory is str(`output_id`) repeated 3 times (e.g. 0 ->
    '000'; note 1 -> '111', not '001'). Prints the file name.

    Parameters
    ----------
    session_folder : str
        path to session folder
    output_id : int or str
        export identifier

    Returns
    -------
    pandas.DataFrame
        contents of gaze_positions.csv
    """
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
        yaml.dump(parameters, fid)


def force_list(x, convert_tuple=False):
    """Assure that a variable is a list. 

    Any variable that is not a list (including a tuple) is enclosed in a list."""
    if x is None:
        return []
    if convert_tuple and isinstance(x, tuple):
        x = list(x)
    if not isinstance(x, list):
        x = [x]
    return x


def unique(seq, idfun=None):
    """Returns only unique values in a list (with order preserved).

    The first occurrence of each item is kept.

    Stolen from the internets 11.29.11

    Parameters
    ----------
    seq : iterable
        items from which to select unique values
    idfun : callable or None, optional
        function mapping each item to a hashable key; items with the same
        key are considered duplicates. By default None (the item itself)

    Returns
    -------
    result : list
        unique items of `seq`, in order of first occurrence
    seen : dict
        count of occurrences of each key
    """
    # order preserving
    if idfun is None:
        def idfun(x):
            """Identity key function (default)."""
            return x
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

    Resamples each data arraydict onto the timestamps of the first one.
    If `window` is None (default), the nearest sample in time is taken
    (and `fn` is ignored). Otherwise, all samples within `window` of each
    reference time are selected and aggregated with `fn`.

    NOTE: This is messy. computing median doesn't work for fields of
    data that are e.g. dictionaries. These must be removed before
    calling this function for now. Fields for which indexing / `fn`
    fails are silently left as zeros.

    Parameters
    ----------
    *data : dict
        two or more arraydicts, each with a 'timestamp' field (on the same
        clock) and numpy array fields with time as the first dimension.
        The first is the time reference.
    fn : callable, optional
        aggregation function, called as fn(values, axis=0), by default
        np.median
    window : scalar or None, optional
        half-width (same units as timestamps, usually seconds) of window
        around each reference time, by default None (nearest sample)

    Returns
    -------
    dict or tuple of dicts
        one arraydict per input after the first, with 'timestamp' equal to
        the reference timestamps; a single dict if only two inputs are
        given
    """
    if window is None:
        # Overwite any function argument if window is set to none;
        # this will do nearest-frame resampling
        def fn(x, axis=None):
            """Identity aggregation, for nearest-sample matching."""
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
    data : np.ndarray, 1D
        binary (bool or 0/1) array from which onsets and offsets should be
        extracted. Must contain at least one True value (otherwise an
        IndexError is raised).
    return_duration : bool, optional
        whether to include a duration column, by default True

    Returns
    -------
    np.ndarray
        (n_events, 3) int array of (onset, offset, duration) in samples
        (or (n_events, 2) without duration). Onset is the index of the
        first True sample and offset the index of the first False sample
        after it (exclusive end), so duration = offset - onset. For an
        event still on at the end of `data`, offset is len(data) if it is
        the only event, otherwise -1 (duration is computed correctly as
        len(data) - onset in both cases).
    """
    if data[0]:
        start_value = 1
    else:
        start_value = 0
    data = data.astype(float).copy()

    ddata = np.hstack([[start_value], np.diff(data)])
    (onsets,) = np.nonzero(ddata > 0)
    # print(onsets)
    (offsets,) = np.nonzero(ddata < 0)
    # print(offsets)
    if (len(offsets) == 0) & (len(onsets) == 1):
        offsets = [len(data)]
        on_at_end = True
    else:
        on_at_end = False
    onset_first = onsets[0] < offsets[0]
    len(onsets) == len(offsets)

    #on_at_end = False
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

    onoff = onoff.T.astype(int)
    return onoff


def onoff_to_binary(onoff, length):
    """Convert (onset, offset) tuples to binary index
    
    Parameters
    ----------
    onoff : np.ndarray
        (n, 2) or (n, 3) array; each row is (onset_index, offset_index,
        [duration_in_frames]) for some event. Offset is exclusive.
    length : int
        total length of output binary index

    Returns
    -------
    index : np.ndarray
        (length,) boolean index vector, True within events
    """
    index = np.zeros(length,)
    for on, off in onoff[:, :2]:
        index[on:off] = 1
    return index > 0


def time_to_index(onsets_offsets, timeline, index_type='integer'):
    """find indices between onsets & offsets in timeline

    Parameters
    ----------
    onsets_offsets : array-like
        (n, 2) array of onsets and offsets in TIME (exactly 2 columns;
        (n, 3) onoff arrays must be sliced first)
    timeline : np.ndarray
        1d array of timestamps; this is the timeline into which to translate the time indices
    index_type : str, optional
        'integer' or 'boolean', by default 'integer'.
        'integer' returns integer indices for onsets and offsets in the specified timeline
        'boolean' returns a boolean index into `timeline` that is True
        within any of the segments

    Returns
    -------
    np.ndarray
        if 'integer', (n, 2) int array of (first index with timeline >= onset,
        1 + last index with timeline < offset), i.e. an exclusive end
        index; if 'boolean', (len(timeline),) boolean array. Raises an
        IndexError if a segment contains no timeline points.
    """
    if not isinstance(onsets_offsets, np.ndarray):
        onsets_offsets = np.asarray(onsets_offsets)
    out = np.zeros(onsets_offsets.shape, dtype=int)
    for ct, (on, off) in enumerate(onsets_offsets):
        i = np.flatnonzero(timeline >= on)[0]
        j = np.flatnonzero(timeline < off)[-1]
        out[ct] = [int(i), int(j)+1]
    if index_type=='boolean':
        out = onoff_to_binary(out, len(timeline))
    return out


def filter_list(lst, idx):
    """Convenience function to select items from a list with a binary index"""
    return [x for x, i in zip(lst, idx) if i]


def filter_arraydict(arraydict, idx):
    """Apply the same index to all fields in a dict of arrays

    Parameters
    ----------
    arraydict : dict
        dict of arrays, all with the same length
    idx : array-like
        boolean index (same length as fields of `arraydict`)

    Returns
    -------
    dict
        arraydict with only the selected items (rebuilt via a dictlist,
        so fails if no items are selected)
    """
    dictlist = arraydict_to_dictlist(arraydict)
    dictlist = filter_list(dictlist, idx)
    out = dictlist_to_arraydict(dictlist)
    return out


def stack_arraydicts(*inputs, sort_key=None):
    """Concatenate arraydicts with the same fields

    Parameters
    ----------
    *inputs : dict
        arraydicts to concatenate (fields of the first define the output)
    sort_key : str or None, optional
        field by which to sort the stacked items (e.g. 'timestamp'), by
        default None (keep input order)

    Returns
    -------
    dict
        stacked arraydict. If all inputs are empty, the first input
        itself (not a copy) is returned.
    """
    output =arraydict_to_dictlist(inputs[0])
    for arrdict in inputs[1:]:
        arrlist = arraydict_to_dictlist(arrdict)
        output.extend(arrlist)
    if sort_key is not None:
        output = sorted(output, key=lambda x: x[sort_key])
    # Handle case in which all fields are empty. 
    if len(output) > 0: 
        output = dictlist_to_arraydict(output)
    else:
        # In degenerate case, return first dict of empty arrays
        # This might be a one-off fix, unclear
        output = inputs[0]
    return output


def dictlist_to_arraydict(dictlist):
    """Convert from pupil format list of dicts to dict of arrays

    Parameters
    ----------
    dictlist : list of dict
        non-empty list of per-sample dicts; keys of the first dict define
        the output fields

    Returns
    -------
    dict
        one np.array per field, with samples along the first dimension
    """
    dict_fields = list(dictlist[0].keys())
    out = {}
    for df in dict_fields:
        out[df] = np.array([d[df] for d in dictlist])
    return out


def arraydict_to_dictlist(arraydict):
    """Convert from dict of arrays to pupil format list of dicts

    Parameters
    ----------
    arraydict : dict
        dict of arrays; the length of the first field sets the number of
        items

    Returns
    -------
    list of dict
        one dict per sample; array-valued entries are converted to lists
    """
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


def get_frame_indices(start_time, end_time, all_time):
	"""Finds start and end indices for frames that are between `start_time` and `end_time`
	
	Note that `end_frame` returned will be the first frame that occurs after
	end_time, such that some data[start_frame:end_frame] will span the range 
	between `start_time` and `end_time`. 

	Parameters
	----------
	start_time: scalar
		time after which to select frames
	end_time: scalar
		time before which to select frames
	all_time: np.ndarray
		full array of timestamps for data into which to index.

	Returns
	-------
	start_frame : int
		index of first frame with time strictly > `start_time`
	end_frame : int
		1 + index of last frame with time strictly < `end_time`
		(raises an IndexError if no frames fall in the range)
	"""
	ti = (all_time > start_time) & (all_time < end_time)
	time_clipped = all_time[ti]
	indices, = np.nonzero(ti)
	start_frame, end_frame = indices[0], indices[-1] + 1
	return start_frame, end_frame 


def get_function(function_name):
    """Load a function to a variable by name

    Parameters
    ----------
    function_name : str or callable
        string name for function (including module), e.g.
        'vedb_gaze.marker_detection.find_concentric_circles'. If a
        callable is given, it is returned as is.

    Returns
    -------
    callable
        the named function
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
    """Select items from a list by attribute values and check their number

    Parameters
    ----------
    dict_list : list
        list of objects to filter. Note: matching uses hasattr / getattr,
        so items must be objects with attributes (plain dicts never match
        any keyword).
    n : int or None, optional
        number of matching items expected, by default 1. If None, all
        matches are returned without checking.
    **kwargs
        attribute=value pairs that matching items must have

    Returns
    -------
    object or list
        the single match if `n` is 1, otherwise the list of matches

    Raises
    ------
    ValueError
        if the number of matches is not `n`
    """
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

def make_file_strings(
        pupil=defaults['pupil'],
        eyelid=defaults['eyelid'],
        pupil_detrend=defaults['pupil_detrend'],
        calibration_marker=defaults['calibration_marker'],
        calibration_split=defaults['calibration_split'],
        calibration_cluster=defaults['calibration_cluster'],
        validation_marker=defaults['validation_marker'],
        validation_split=defaults['validation_split'],
        validation_cluster=defaults['validation_cluster'],
        calibration=defaults['calibration'],
        gaze=defaults['gaze'],
        error=defaults['error'],
        calibration_epoch=0,
        # Extra
        eye=None,
        fov_str=None,
        validation_checkerboard_size = '4x7',
        validation_epoch=None,
        #output_dir=None # Maybe include to only keep paths that are verified as there?
        # Tho this function does not load, only returns strings, sometimes
        # with formatting characters for use by other functions, so maybe not.
        ):
    """
    Construct filename templates for gaze-pipeline outputs.

    Each argument is a processing tag (parameter set name) for one
    pipeline step; defaults come from the 'defaults' section of the
    package config. Calibration, gaze, and error filenames include a hash
    of the tags of all upstream steps (calibration hash: calibration
    marker, split, cluster, epoch, pupil, pupil_detrend; error hash: those
    plus eyelid, calibration, gaze, and validation marker, split, cluster;
    None tags are skipped). This function only builds strings; it does not
    check that files exist.

    Parameters
    ----------
    pupil : str or None
        Tag for the pupil detection algorithm (used in pupil filename and
        hashes).
    eyelid : str or None
        Eyelid detection tag; only used in the error hash.
    pupil_detrend : str or None
        Pupil detrending tag; only used in the calibration and error hashes.
    calibration_marker : str or None
        Calibration marker detection tag.
    calibration_split : str or None
        Calibration marker split tag; only used in hashes.
    calibration_cluster : str or None
        Calibration marker clustering tag.
    validation_marker, validation_cluster : str or None
        Tags for validation marker detection and clustering.
    validation_split : str or None
        Validation marker split tag; only used in the error hash.
    calibration : str or None
        Calibration algorithm tag (used in calibration and gaze filenames).
    gaze : str or None
        Gaze mapping algorithm tag.
    error : str or None
        Error computation tag.
    calibration_epoch : int, optional
        Calibration epoch index, by default 0.
    eye : str or None, optional
        Eye ('left' or 'right') inserted in filenames, by default None
        (leaves a '%s' placeholder).
    fov_str : str or None, optional
        Field-of-view string inserted in error filenames, by default None
        (leaves a '%s' placeholder).
    validation_checkerboard_size : str, optional
        Checkerboard size; if not '4x7', replaces '4x7' in the
        `validation_marker` tag, by default '4x7'.
    validation_epoch : int, list, tuple, or None, optional
        Validation epoch(s) for validation and error filenames, by default
        None (epochs 0-5).

    Returns
    -------
    out : dict
        Filenames (templates) with keys 'pupil', 'calibration_marker',
        'calibration_cluster', 'calibration', 'gaze' (str), and
        'validation_marker', 'validation_cluster', 'error' (lists, one per
        validation epoch). A value is None if its tag is None. Templates
        contain '%s' placeholders for `eye` and / or `fov_str` if those
        were not given.
    """
        # Hashes of inputs for steps with too many inputs for a_b_c type filename construction
    if fov_str is None:
        fov_str = '%s'
    if eye is None:
        eye = '%s'
    if validation_checkerboard_size != '4x7':
        validation_marker = validation_marker.replace('4x7', validation_checkerboard_size)
    #print(validation_marker)
        
    calibration_args = [x for x in [calibration_marker, calibration_split, \
                                       calibration_cluster, f'epoch{calibration_epoch:02d}', \
                                       pupil, pupil_detrend] if x is not None]
    # '-' will mess up later parsing of file names, so replace; this *might* make hashes non-unique, but is most likely to be fine.
    calibration_input_hash = hashlib.blake2b(('-'.join(calibration_args)).replace('-','0').encode(), digest_size=10).hexdigest()
    error_args = [x for x in [calibration_marker, calibration_split, \
                                       calibration_cluster, f'epoch{calibration_epoch:02d}', \
                                       pupil, eyelid, pupil_detrend, \
                                       calibration, gaze,
                                       validation_marker, validation_split, validation_cluster, \
                                       ] if x is not None]
    error_input_hash = hashlib.blake2b(('-'.join(error_args)).replace('-','0').encode(), digest_size=10).hexdigest()
    if validation_epoch is None: 
        validation_epoch = [0, 1, 2, 3, 4, 5]
    else:
        validation_epoch = force_list(validation_epoch, convert_tuple=True)
    out = dict(
        pupil = f'pupil_detection-{eye}-{pupil}.npz' if pupil is not None else None,
        calibration_marker = f'markers-{calibration_marker}-epoch{calibration_epoch:02d}.npz' if calibration_marker is not None else None,
        calibration_cluster = f'markers-{calibration_marker}-{calibration_cluster}-epoch{calibration_epoch:02d}.npz' if calibration_cluster is not None else None,
        calibration = f'calibration-{eye}-{calibration}-{calibration_input_hash}.npz' if calibration is not None else None,
        gaze = f'gaze-{eye}-{gaze}-{calibration}-{calibration_input_hash}.npz' if gaze is not None else None,
        validation_marker = [f'markers-{validation_marker}-epoch{ve:02d}.npz' for ve in validation_epoch] if validation_marker is not None else None,
        validation_cluster =  [f'markers-{validation_marker}-{validation_cluster}-epoch{ve:02d}.npz' for ve in validation_epoch] if validation_cluster is not None else None,
        error = [f'error-{eye}-{error}_{fov_str}-{error_input_hash}-epoch{ve:02d}.npz' for ve in validation_epoch] if error is not None else None,
        )
    return out

def _load_files(fstr, folder, eye):
    """Load .npz pipeline output file(s) into dicts

    Parameters
    ----------
    fstr : str or list of str
        filename or filename template(s) (with a '%s' for eye if `eye` is
        not None), as from `make_file_strings`
    folder : pathlib.Path
        directory containing the files
    eye : str, list, tuple, or None
        eye(s) substituted into `fstr` via `fstr % eye`; 'both' means
        ['left', 'right']; None for no substitution

    Returns
    -------
    dict, list, or None
        If `eye` is None: a dict of the file contents (str `fstr`; the
        file must exist) or a list of dicts, with None for missing files
        (list `fstr`). Otherwise a dict keyed by eye of the same; for str
        `fstr`, missing files are omitted. None if nothing was loaded.
    """
    if eye == 'both':
        eye = ['left','right']
    elif eye is None:
        eye = [None]
    out = {}
    for e in force_list(eye, convert_tuple=True):
        if isinstance(fstr, (list, tuple)):
            if e is None:
                out = []
            else:
                out[e] = []
            
            for fs in fstr:
                if e is None:
                    fnm = folder / fs
                    if fnm.exists():
                        out.append(dict(np.load(fnm, allow_pickle=True)))
                    else:
                        out.append(None)
                else:
                    fnm = folder / (fs%e)    
                    if fnm.exists():
                        out[e].append(dict(np.load(fnm, allow_pickle=True)))
                    else:
                        out[e].append(None)
        else:
            if e is None:
                fnm = folder / fstr
                out = dict(np.load(fnm, allow_pickle=True))
            else:
                fnm = folder / (fstr%e)
                if fnm.exists():
                    out[e] = dict(np.load(fnm, allow_pickle=True))
    if len(out) == 0:
        out = None
    return out


def load_pipeline_elements(folder,
        pupil=defaults['pupil'],
        eyelid=defaults['eyelid'],
        pupil_detrend=defaults['pupil_detrend'],
        calibration_marker=defaults['calibration_marker'],
        calibration_split=defaults['calibration_split'],
        calibration_cluster=defaults['calibration_cluster'],
        validation_marker=defaults['validation_marker'],
        validation_split=defaults['validation_split'],
        validation_cluster=defaults['validation_cluster'],
        calibration=defaults['calibration'],
        gaze=defaults['gaze'],
        error=defaults['error'],
        calibration_epoch=0,
        is_verbose=1,
        eye=('left','right'),
        **kwargs,
        ):
    """Load all elements of gaze pipeline into a dict given processing spec and folder

    Filenames are built with `make_file_strings`; steps whose tag is None
    are skipped.

    Parameters
    ----------
    folder : str or pathlib.Path
        full path to directory with files in it
    pupil, eyelid, pupil_detrend, calibration_marker, calibration_split,
    calibration_cluster, validation_marker, validation_split,
    validation_cluster, calibration, gaze, error : str or None
        processing tags for each step (see `make_file_strings`); defaults
        from the package config
    calibration_epoch : int, optional
        calibration epoch, by default 0
    is_verbose : int, optional
        unused, by default 1
    eye : str or tuple, optional
        eye(s) for which to load per-eye files, by default ('left', 'right')
    **kwargs
        passed to `make_file_strings` (e.g. `fov_str`, `validation_epoch`,
        `validation_checkerboard_size`). Note that error filenames contain
        an fov placeholder in addition to the eye placeholder, so loading
        error files likely requires `fov_str` to be given.

    Returns
    -------
    dict
        'folder' (name of `folder`), plus one entry per non-None step with
        keys as returned by `make_file_strings`: marker and cluster files
        as dicts / lists of dicts (None for missing epochs), 'calibration'
        as a dict of eye -> `Calibration` object, and pupil, gaze, and
        error as dicts keyed by eye (see `_load_files`).

    Raises
    ------
    Exception
        if no pupil, gaze, or error files are found for a requested step
    """
    from .calibration import Calibration
    folder = pathlib.Path(folder)
    # Create outputs dict
    outputs = dict(folder=folder.name)
    file_paths = make_file_strings(
        pupil=pupil,
        eyelid=eyelid,
        pupil_detrend=pupil_detrend,
        calibration_marker=calibration_marker,
        calibration_split=calibration_split,
        calibration_cluster=calibration_cluster,
        validation_marker=validation_marker,
        validation_split=validation_split,
        validation_cluster=validation_cluster,
        calibration=calibration,
        gaze=gaze,
        error=error,
        calibration_epoch=calibration_epoch,
        **kwargs,
    )
    for k, fpath in file_paths.items():
        if fpath is not None:
            if k in ['calibration_marker', 'calibration_cluster',
                     'validation_marker', 'validation_cluster',]:
                outputs[k] = _load_files(fpath, folder, None)
            elif k in ['calibration']:
                if eye == 'both':
                    outputs[k] = Calibration.load(folder / fpath%e)
                else:
                    outputs[k] = {}
                    for e in force_list(eye, convert_tuple=True):
                        outputs[k][e] = Calibration.load(folder / (fpath%e))
            else:
                outputs[k] = _load_files(fpath, folder, eye)
                if outputs[k] is None:
                    raise Exception("WTF YO")

    return outputs

def remove_outliers(timestamps, data, 
                    z_threshold=4,
                    absolute_min=None,
                    absolute_max=None
                    ):
    """remove outliers from dataset

    Parameters
    ----------
    timestamps : array-like
        timestamps associated with each data point (timestamps associated
        with outlying data points are also removed)
    data : array-like
        data in which to search for outliers
    z_threshold : scalar, optional
        threshold for z score for outliers, by default 4
    absolute_min : scalar, optional
        absolute minimum threshold below which points will be considered
        outliers; None for skip this, by default None
    absolute_max : scalar, optional
        absolute maximum threshold above which points will be considered
        outliers; None for skip this, by default None

    Returns
    -------
    timestamps, data : np.ndarray
        inputs with outlying points removed (points with abs(z) >=
        `z_threshold`, with z computed over all of `data`)
    """
    keep = np.ones_like(data) > 0
    # First, remove absolute threshold out-of-bounds
    if absolute_min is not None:
        keep &= (data >= absolute_min)
    if absolute_max is not None:
        keep &= (data <= absolute_max)
    if z_threshold is not None:
        data_z = zscore(data)
        keep &= (np.abs(data_z) < z_threshold)
    # Alternatively, set to nans, or return outlier index?
    return timestamps[keep], data[keep]

def resample_data(timestamps, data, 
                      fps=120,
                      new_time=None,
                      method='linear_interpolation',
                      remove_nans=True,
                      **kwargs):
    """Resample data to new (by default uniformly spaced) timestamps

    Parameters
    ----------
    timestamps : np.ndarray
        1d timestamps (seconds) associated with `data`
    data : np.ndarray
        values to be resampled (the 'y' or dependent values
        to the `timestamps`' 'x'). May be 1d or 2d array,
        first dimension must match `timestamps`
    fps : scalar, optional
        new sampling rate in Hz (will be uniform), by default 120. Ignored
        if `new_time` is provided.
    new_time : np.ndarray or None, optional
        new array of timestamps, to allow manual specification. If left
        as None, new_time will be np.arange(timestamps[0], timestamps[-1],
        1/fps) (last timestamp excluded)
    method : str, optional
        'linear_interpolation' (`scipy.interpolate.interp1d`) or
        'thin-plate_spline' (`scipy.interpolate.RBFInterpolator`), method
        for interpolation, by default 'linear_interpolation'
    remove_nans : bool, optional
        whether to remove any nans in data before resampling (this will
        fill in those nans with interpolated values), by default True
    **kwargs
        passed to `scipy.interpolate.RBFInterpolator` for
        'thin-plate_spline' (e.g. `smoothing`; `neighbors` defaults to 7);
        ignored for 'linear_interpolation'

    Returns
    -------
    new_time : np.ndarray
        new timestamps; (m,) for 'linear_interpolation', (m, 1) for
        'thin-plate_spline'
    data_out : np.ndarray
        resampled data; same number of dimensions as `data` for
        'linear_interpolation', always 2d (m, k) for 'thin-plate_spline'
    """
    if new_time is None:
        new_time = np.arange(timestamps[0], timestamps[-1], 1/fps)
    # Make 2d for some interpolators
    make_2d = method not in ('linear_interpolation',)
    if make_2d:
        if np.ndim(data) < 2:
            inpt = data.reshape(-1, 1)
        else:
            inpt = data
        t = timestamps.reshape(-1, 1)
        new_time = new_time.reshape(-1, 1)
    else:
        inpt = data
        t = timestamps

    if remove_nans:
        # Remove nans
        if np.ndim(inpt) > 1:
            keep = ~np.any(np.isnan(inpt), axis=1)
        else:
            keep = ~np.isnan(inpt)
        t = t[keep]
        inpt = inpt[keep]

    if method == 'linear_interpolation':
        interp = scipy.interpolate.interp1d(t, inpt, axis=0)
    elif method == 'thin-plate_spline':
        if not 'neighbors' in kwargs:
            kwargs['neighbors'] = 7
            print("Setting neighbors=7 for efficient processing, add a `neigbors=<whatever>` kwarg if you wish to change this")
        interp = scipy.interpolate.RBFInterpolator(t, inpt, **kwargs)
    else:
        raise NotImplementedError(f"Method {method} not available!")
    data_out = interp(new_time)
    return new_time, data_out

def filter_data(timestamps, data, 
                low_cutoff=None,
                high_cutoff=None,
                ):
    """Placeholder for temporal filtering of data; NOT IMPLEMENTED.

    Currently does nothing and returns None. Parameters (timestamps, data,
    and presumably low / high frequency cutoffs) are reserved for a future
    implementation.
    """
    pass