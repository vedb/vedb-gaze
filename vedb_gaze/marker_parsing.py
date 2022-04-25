# Marker parsing 
from . import utils 
import numpy as np
from sklearn import cluster
from scipy.stats import zscore
import copy

def find_duplicates(timestamps, mode='all',):
    """Remove duplicate time values in a series of timestamps
    
    Parameters
    ----------
    timestamps : TYPE
        Description
    mode : str, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    _, b = utils.unique(timestamps)
    aa, bb = np.array(list(b.keys())), np.array(list(b.values()))
    if mode == 'first':
        duplicates = None
    elif mode == 'all':
        duplicates = np.in1d(timestamps, aa[bb > 1])
    return duplicates


def remove_brief_detections(markers, all_timestamps, duration_threshold=0.6):
    """Remove brief marker detections (less than `duration_threshold`) 
    
    This proceeds in two steps:
    
    1) remove duplicate frames, i.e. frames in which two calibration stimuli were detected.
        These are likely to be erroneous detections of markers in the periphery.
    2) set a minimum threshold on the time over which a marker was detected. The logic here
        is that very brief detections lasting << 1 second are not likely to real detections 
        of markers intended to be used for calibration or validations.
    
    Parameters
    ----------
    markers : dict of arrays
        dict of arrays for detected markers (or other quantity)
    all_timestamps : array-like
        all timestamps for relevant clock (probably world camera clock)
    duration_threshold : float, optional
        minimum time allowable for detections
    
    Returns
    -------
    dict of arrays
        marker detections with brief detections filtered out
    
    """
    time_values = markers['timestamp'].copy()
    # Find duplicate timepoints, remove
    duplicates = find_duplicates(time_values)
    time_values_clean = time_values[~duplicates]
    # Check for extremely close values that are not exactly equal
    time_diff_threshold = 1e-8
    # Check for values in marker timestamps that are not in `all_timestamps`
    # THERE SHOULD BE NONE, but SOMETIMES, probably due to rounding errors, 
    # a value in the marker timestamps is VERY slightly different from the 
    # corresponding value in all_timestamps. This code detects those. 
    # Yes this is very annoying.
    potentially_close_times = list(set(time_values_clean) - set(all_timestamps))
    for ptc in potentially_close_times:
        time_diff = np.min(np.abs(all_timestamps - ptc))
        if time_diff < time_diff_threshold:
            ii = np.argmin(np.abs(all_timestamps - ptc))
            jj = list(time_values_clean).index(ptc)
            print("Changed: %.10f" %(time_values_clean[jj]))
            time_values_clean[jj] = all_timestamps[ii]
            print("To:      %.10f" %(time_values_clean[jj]))

    time_index = np.in1d(all_timestamps, time_values_clean)
    # Filter for duration
    onoff = utils.onoff_from_binary(time_index)
    keepers = onoff[:,2] > duration_threshold
    # Convert back to binary selection vector for time points to keep
    onoff_binary = utils.onoff_to_binary(onoff[keepers], len(all_timestamps))
    keep_index = onoff_binary[time_index]
    # Keep only "good" times & positions
    output = dict((key, value[~duplicates][keep_index]) for key, value in markers.items())
    return output


def split_timecourse(*data, max_epoch_gap=15, min_epoch_length=30, verbose=True):
    """Splits data (possibly multiple data streams) with timestamps into multiple 
    segments or epochs if timestamps are greater than `max_epoch_gap` seconds apart.
    
    All splitting is based on the FIRST data dictionary input; all dictionaries must
    be the same size coming in. 
    
    Parameters
    ----------
    *data
        Description
    max_epoch_gap : int, optional
        Description
    min_epoch_length : None, optional
        Description
    verbose : bool, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    if verbose:
        print("== Splitting timecourse ==")
    timestamps = data[0]['timestamp']
    t0 = copy.copy(timestamps[0])
    break_indices, = np.nonzero(np.diff(timestamps) > max_epoch_gap)
    # np.diff shortens index; set correct w/ +1
    break_indices += 1
    break_indices = np.hstack([[0], break_indices, [len(timestamps)]])
    if verbose:
        print('Frame indices for breaks btw timestamps:')
        print(break_indices)
    output = []
    epoch_durations = []
    for st, fin in zip(break_indices[:-1], break_indices[1:]):
        this_epoch = []
        for d in data:
            new_dict = {}
            for k, v in d.items():
                new_dict[k] = v[st:fin]
            this_epoch.append(new_dict)
        epoch_duration = (this_epoch[0]['timestamp'][-1] - this_epoch[0]['timestamp'][0])
        epoch_durations.append(epoch_duration)
        if epoch_duration > min_epoch_length:
            output.append(this_epoch)
    if verbose:
        print('%d epochs found:' %
              (len(break_indices)-2))
        for x, dur in zip(break_indices[1:-1], epoch_durations):
            #print(x)
            #print(timestamps[x])
            print('@ %d min, %.1f s : %.1f s long' % ((timestamps[x]-t0) // 60, (timestamps[x]-t0) % 60, dur))

        print('%d epochs meet duration limits' % len(output))
    return output


def compute_cluster_std(marker, groups, pupil_left=None, pupil_right=None):
    """
    Parameters
    ----------
    marker : TYPE
        Description
    groups : TYPE
        Description
    pupil_left : None, optional
        Description
    pupil_right : None, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    u_groups = np.unique(groups)
    key = 'norm_pos'
    group_stds = []
    for group_number in u_groups:
        tmp = []
        for x in [marker, pupil_left, pupil_right]:
            if x is not None:
                tmp.append(np.std(x[key][groups == group_number], axis=0).mean())
        group_stds.append(tmp)
    output = np.array(group_stds)
    output = np.nan_to_num(output)
    return output


def cluster_marker_points(markers,
                          pupil_left=None,
                          pupil_right=None,
                          cluster_by=("marker:timestamp",
                                      "marker:norm_pos",),
                          cluster_method='DBSCAN',
                          cluster_kw=None,
                          min_cluster_time=1,
                          max_cluster_time=3,
                          aspect_ratio=4/3,
                          max_cluster_std=2.0,
                          normalize_time=True,
                          is_verbose=True,
                          ):
    """Find clusters of points in marker data
    
    NOTE: currently filters out nans and zeros in any channel that is clustered; thus,
    those time points will not be present at all in resulting clusters.
    
    Question: Normalize time? (Do all properties need to be on the same scale for all 
    clustering algorithms?)
    
    Parameters
    ----------
    markers : TYPE
        Description
    pupil_left : TYPE
        Description
    pupil_right : TYPE
        Description
    cluster_by : tuple, optional
        Description
    cluster_method : str, optional
        Description
    cluster_kw : None, optional
        Description
    min_cluster_time : int, optional
        Description
    max_cluster_time : int, optional
        Description
    aspect_ratio : TYPE, optional
        Description
    normalize_time : bool, optional
        Description
    is_verbose : bool, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    # Create an array of data to cluster
    to_cluster = []
    for c in cluster_by:
        marker_type, key = c.split(':')
        if marker_type == 'marker':
            tmp = markers[key].copy()
        elif marker_type == 'pupil_left':
            tmp = pupil_left[key].copy()
        elif marker_type == 'pupil_right':
            tmp = pupil_right[key].copy()
        else:
            raise ValueError(f'unknown field {marker_type} in `cluster_by`')
        if np.ndim(tmp) == 1:
            tmp = tmp[:, np.newaxis]
        if (key == 'timestamp') and normalize_time:
            # Normalize time to put it on same scale as other axes
            # This should make time 0-1 over whatever the epoch time is
            assumed_epoch_time = 90 # Seconds; so time will have consistent
                                    # spacing for all runs to be clustered.
            tmp = (tmp-np.min(tmp)) / assumed_epoch_time #np.ptp(tmp)
            # This helps, for whatever reason. Something about the range of
            # 0-0.5 or 0-1 or so does nto work well with clustering. Unclear why, 
            # perhaps worth exploring.
            tmp += 2
            print('Normalized time range:')
            print(tmp.min(), tmp.max())
        if (key in ('norm_pos', )) and (marker_type == 'marker'):
            tmp *= np.array([[aspect_ratio, 1.0]])
        to_cluster.append(tmp)
    to_cluster = np.hstack(to_cluster)
    # Clustering parameters
    if cluster_kw is None:
        cluster_kw = dict(eps=0.05)
    # Remove NaNs, zeros
    to_kill = np.any(np.isnan(to_cluster), axis=1) | np.any(
        to_cluster == 0, axis=1)
    to_cluster = to_cluster[~to_kill, :]
    # Do clustering
    cluster_fn = getattr(cluster, cluster_method)
    cluster_mapper = cluster_fn(**cluster_kw)
    groups = cluster_mapper.fit_predict(to_cluster)
    unique_groups = np.unique(groups)
    if is_verbose:
        print('Found %d groups' % (len(unique_groups)))
    # Keep clusters that are within a specified interval of length
    ct = markers['timestamp'][~to_kill]
    group_durations = np.array([np.ptp(ct[groups == uv]) for uv in unique_groups])
    print(group_durations)
    keep_groups = (group_durations > min_cluster_time) & (
        group_durations < max_cluster_time)
    appropriate_duration_index = unique_groups[keep_groups]
    if is_verbose:
        print('%d groups after duration filtering' % (keep_groups.sum()))

    keep_groups_i = np.in1d(groups, appropriate_duration_index)
    if keep_groups_i.sum() == 0:
        # No groups meet threshold
        return None
    groups = groups[keep_groups_i]    
    filtered_data = []
    for mk in [markers, pupil_left, pupil_right]:
        if mk is not None:
            mk_dictlist = utils.arraydict_to_dictlist(markers)
            mk_dictlist = utils.filter_list(mk_dictlist, keep_groups_i)
            mk_arraydict = utils.dictlist_to_arraydict(mk_dictlist)
        else:
            mk_arraydict = None
        filtered_data.append(mk_arraydict)
    markers, pupil_left, pupil_right = filtered_data

    # Keep clusters that have a standard deviation lower than a set threshold
    # (This is functional but clumsy, worth a revisit later perhaps)
    unique_groups = np.unique(groups)
    group_stds = compute_cluster_std(markers, groups, pupil_left, pupil_right)
    keep_groups = ~np.any(zscore(group_stds, axis=0) > max_cluster_std, axis=1)
    small_std_index = unique_groups[keep_groups]
    if is_verbose:
        print('%d groups after std filtering' % (keep_groups.sum()))
    keep_groups_i = np.in1d(groups, small_std_index)
    if keep_groups_i.sum() == 0:
        # No groups meet threshold
        return None
    groups = groups[keep_groups_i]
    marker_dictlist = utils.arraydict_to_dictlist(markers)
    marker_dictlist = utils.filter_list(marker_dictlist, keep_groups_i)
    for j, grp in enumerate(groups):
        marker_dictlist[j]['marker_cluster_index'] = grp
    marker_arraydict = utils.dictlist_to_arraydict(marker_dictlist)

    return marker_arraydict


def find_epochs(marker,
                all_timestamps,
                pupil_left=None,
                pupil_right=None,
                max_epoch_gap=15,
                min_epoch_length=30,
                duration_threshold=0.3,
                do_duration_pre_check=True,
                min_n_clusters=5,
                do_cluster_clean=True,
                cluster_method='DBSCAN',
                cluster_by=("marker:timestamp",
                            "marker:norm_pos",),
                cluster_kw=None,
                min_cluster_time=0.3,
                max_cluster_time=3.0,
                max_cluster_std=4.0,
                is_verbose=True,
                ):
    """Find epochs of matched points within lists of pupil detections & marker detections
        
        Parameters
        ----------
        marker : TYPE
            Description
        all_timestamps : array-like
            all timestamps for session on relevant clock (probably world camera clock)
        pupil_left : dict of arrays or None
            pupil detections for this session, if used in finding epochs (None if not)
        pupil_right : dict of arrays or None
            pupil detections for this session, if used in finding epochs (None if not)        
        max_epoch_gap : int, optional
            maximum time gap to consider subsequent marker detections to be within the same
            epoch of (calibration or validation)
        min_epoch_length : int, optional
            minimum time from first to last marker detection to consider an epoch to be
            worth saving
        do_duration_pre_check : bool, optional
            Whether to check for too-short detections that are assumed to be spurious
        duration_threshold : float, optional
            Minimum length for detected markers to be present (if do_duration_pre_check is True)
        min_n_clusters : int, optional
            Description
        error_smoothing_kernels : tuple, optional
            Description
        do_cluster_clean : bool, optional
            Description
        cluster_method : str, optional
            Method for clustering from sklearn.cluster
        cluster_by : tuple, optional
            list of strings of properties over which to cluster. Each string is
            'input_type:property', e.g. 'calibration:size'
        cluster_kw : None, optional
            Description
        min_cluster_time : float, optional
            Description
        max_cluster_time : float, optional
            Description
        """

    # Optionally clean up marker detections to remove duplicate time stamps and too-short detections
    if do_duration_pre_check:
        marker = remove_brief_detections(marker, all_timestamps,
                                         duration_threshold=duration_threshold)

    # Match time points to compare markers w/ eye data
    to_split = [marker]
    # Make window width half median frame rate, by default
    frame_time = np.median(np.diff(all_timestamps))
    window = frame_time / 2
    if pupil_left is not None:
        pupil_left_match = utils.match_time_points(
        marker, pupil_left, window=window)
        to_split.append(pupil_left_match)
    if pupil_right is not None:
        pupil_right_match = utils.match_time_points(
            marker, pupil_right, window=window)
        to_split.append(pupil_right_match)
    
    # Split into multiple epochs of calibration, validation as needed
    epochs = split_timecourse(*to_split,
                              max_epoch_gap=max_epoch_gap,
                              min_epoch_length=min_epoch_length,
                              verbose=True)

    # Clean up w/ clustering
    if do_cluster_clean:
        #epochs_orig = copy.deepcopy(epochs)
        epochs_to_keep = []
        for i, ce in enumerate(epochs):
            print("\n> For epoch %d" % i)
            # Run clustering & filtering by cluster time
            epoch_grpclean = cluster_marker_points(*ce,
                                                   cluster_method=cluster_method,
                                                   cluster_by=cluster_by,
                                                   cluster_kw=cluster_kw,
                                                   min_cluster_time=min_cluster_time,
                                                   max_cluster_time=max_cluster_time,
                                                   max_cluster_std=max_cluster_std,
                                                   is_verbose=is_verbose)
            if epoch_grpclean is None:
                print("All clusters excluded!")
                continue
            grps = epoch_grpclean['marker_cluster_index']
            # Apply min number of clusters as threshold for keeping calibration epochs
            if len(np.unique(grps)) > min_n_clusters:
                epochs_to_keep.append(epoch_grpclean)
                print("Modified length:", len(grps), '(%0.2f%% of original)' %
                      (len(grps) / len(ce[0]['timestamp']) * 100))
        epochs = epochs_to_keep
        n_epochs = len(epochs)
        print("%d epochs after cluster filtering" % n_epochs)
    else: 
        # Return only markers; pupils are redundant, pupil times
        # can be matched with these from full pupil estimates.
        epochs = [ee[0] for ee in epochs]
    return epochs
