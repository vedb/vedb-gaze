# Marker parsing 
"""Filtering, epoch splitting, and clustering of marker detections.

Marker detections (from `marker_detection`) contain spurious and unstable
detections. Functions here remove brief, small, or oblique detections, split
detections into separate epochs (e.g. multiple calibration sessions), and
cluster detections into stable fixations of individual marker positions.

`filter_and_cluster` (and `filter_and_split`) are the functions named by
``config/marker_parsing-<tag>.yaml`` and run by `pipelines.marker_clustering`
(and `pipelines.marker_splitting`). `marker_cluster_stat` is also used to
reduce clusters to single points in `calibration` and `error_computation`.
"""
from . import utils 
import numpy as np
from sklearn import cluster
from scipy.stats import zscore
import copy

def find_duplicates(timestamps, mode='all',):
    """Find duplicated time values in a series of timestamps

    Returns a boolean index; nothing is removed here.

    Parameters
    ----------
    timestamps : array-like
        1D array of timestamps
    mode : str, optional
        'all' to flag every occurrence of a duplicated value, by default 'all'.
        'first' is not implemented and returns None.

    Returns
    -------
    duplicates : array of bool or None
        True for each element of `timestamps` whose value occurs more than once
        (None if `mode` is 'first')
    """
    _, b = utils.unique(timestamps)
    aa, bb = np.array(list(b.keys())), np.array(list(b.values()))
    if mode == 'first':
        duplicates = None
    elif mode == 'all':
        duplicates = np.isin(timestamps, aa[bb > 1])
    return duplicates

def _bimodality_check(data, n_stds_separate=2.5, ):
    """Quick and dirty bimodality check. 
    
    Computes two k-means clusters, finds whether the means for each
    cluster are separated by more than `n_stds_separate`. If so, 
    returns index for larger group only. 
    """
    # Cheapo "fit" with k means
    center2, labels2, fit2 = cluster.k_means(data[:, None], 2)
    # Compute standard deviations for each group
    std2 = [np.std(data[labels2 == ll]) for ll in np.unique(labels2)]
    larger_i = np.argmax(center2)
    smaller_i = np.argmin(center2)
    is_bimodal = (center2[larger_i] - n_stds_separate * std2[larger_i]
                  ) > (center2[smaller_i] + n_stds_separate * std2[smaller_i])
    if is_bimodal:
        keepers = labels2 == larger_i
    else:
        keepers = np.ones_like(data) > 0
    return keepers

    
def remove_brief_detections(markers, all_timestamps, duration_threshold=0.6, is_verbose=False):
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
        minimum duration (seconds) of a run of consecutive frames with
        detections for those detections to be kept, by default 0.6
    is_verbose : bool, optional
        print timestamps that are adjusted for rounding error, by default False
    
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
            if is_verbose:
                print("Changed: %.10f" %(time_values_clean[jj]))
            time_values_clean[jj] = all_timestamps[ii]
            if is_verbose:
                print("To:      %.10f" %(time_values_clean[jj]))

    time_index = np.isin(all_timestamps, time_values_clean)
    # Filter for duration
    onoff = utils.onoff_from_binary(time_index)
    keepers = onoff[:,2] > duration_threshold
    # Convert back to binary selection vector for time points to keep
    onoff_binary = utils.onoff_to_binary(onoff[keepers], len(all_timestamps))
    keep_index = onoff_binary[time_index]
    # Keep only "good" times & positions
    output = dict((key, value[~duplicates][keep_index]) for key, value in markers.items())
    return output


def remove_small_detections(markers,
                            size_std_threshold=None,
                            bimodal_std_threshold=2.5,
                            image_aspect_ratio=4/3,
                            aspect_ratio_threshold=1.2,
                            aspect_ratio_type='x/y',
                            aspect_ratio_keep='less_than_threshold',
                            return_rejects=False,
                            is_verbose=False):
    """Filter out small markers and oblique ellipses 
    
    Such markers are likely to be spurious detections or associated with irregular
    eye behavior, and should be removed.

    Parameters
    ----------
    markers : dict
        Marker data, with fields 'timestamp' and either 'size' (n, 2) (circle
        markers) or 'norm_pos_full_checkerboard' (n, n_corners, 2)
        (checkerboard markers), from which marker size is computed
    size_std_threshold : float, optional
        remove markers with size more than this number of standard deviations
        below the median size, by default None (no size threshold)
    bimodal_std_threshold : float, optional
        if marker sizes are bimodal (two k-means clusters separated by more
        than this many standard deviations), keep only the larger group, by
        default 2.5; None skips this check
    aspect_ratio_threshold : float, optional
        threshold on marker aspect ratio, by default 1.2; None skips this check
    image_aspect_ratio : float, optional
        Aspect ratio of image in which 'norm_pos' is calculated. 'norm_pos' field 
        will be scaled 0-1 for both axes, so if aspect ratio of IMAGE is not 1, aspect
        ratio of markers will not be correct without correcting for image aspect ratio.
    aspect_ratio_type : str, optional
        string indicating how aspect ratio is computed for marker aspect ratio; 
        either 'x/y' (width / height) or 'max/min' (if you don't care about which way 
        is up for marker aspect ratio)
    aspect_ratio_keep : str, optional
        'less_than_threshold' or 'greater_than_threshold': which markers to
        keep relative to `aspect_ratio_threshold`, by default 'less_than_threshold'
    return_rejects : bool, optional
        Flag to return rejected markers instead of kept ones (useful for comparison
        of what is rejected vs what is kept with different settings)
    is_verbose : bool, optional
        print percentage of markers retained, by default False

    Returns
    -------
    filtered_markers : dict of arrays
        markers that pass all criteria (or that fail, if `return_rejects`)
    """
    if 'size' in markers:
        mksz_xy = markers['size'].copy()
    elif 'norm_pos_full_checkerboard' in markers:
        mksz_xy = np.ptp(markers['norm_pos_full_checkerboard'], axis=1)
        mksz_xy[:, 0] *= image_aspect_ratio
    else:
        raise ValueError("Must have 'size' or 'norm_pos_full_checkerboard' parameter in marker dictionary")
    mksz = mksz_xy.mean(1)
    if aspect_ratio_type == 'x/y':
        mkar = mksz_xy[:, 0] / mksz_xy[:, 1]
    elif aspect_ratio_type == 'max/min':
        large_dim = np.max(mksz_xy, axis=1)
        small_dim = np.min(mksz_xy, axis=1)
        mkar = large_dim / small_dim
    median_size = np.median(mksz)
    std_size = np.std(mksz)
    keepers = np.ones_like(markers['timestamp']) > 0
    if bimodal_std_threshold is not None:
        keepers &= _bimodality_check(mksz, n_stds_separate=bimodal_std_threshold)
    if size_std_threshold is not None:
        keepers &= mksz > (median_size - std_size * size_std_threshold)
    if aspect_ratio_threshold is not None:
        if aspect_ratio_keep == 'greater_than_threshold':
            keepers &= mkar > aspect_ratio_threshold
        elif aspect_ratio_keep == 'less_than_threshold':
            keepers &= mkar < aspect_ratio_threshold
    # Select
    lst = utils.arraydict_to_dictlist(markers)
    if return_rejects:
        filtered_markers = utils.dictlist_to_arraydict(
            utils.filter_list(lst, ~keepers))
    else:
        filtered_markers = utils.dictlist_to_arraydict(
            utils.filter_list(lst, keepers))
    if is_verbose:
        print('%.1f%% retained'%(keepers.mean() * 100))
    return filtered_markers


def split_timecourse(*data, max_epoch_gap=15, min_epoch_length=30, max_epoch_length=None, is_verbose=True, absolute_start=None):
    """Splits data (possibly multiple data streams) with timestamps into multiple 
    segments or epochs if timestamps are greater than `max_epoch_gap` seconds apart.
    
    All splitting is based on the FIRST data dictionary input; all dictionaries must
    be the same size coming in. 
    
    Parameters
    ----------
    *data
        one or more dicts of arrays, each with a 'timestamp' field; all arrays
        must have the same length along the first axis
    max_epoch_gap : float, optional
        a gap between consecutive timestamps longer than this (seconds) starts
        a new epoch, by default 15
    min_epoch_length : float, optional
        minimum length (in seconds) for a (marker) epoch to last, by default 30;
        None means 0
    max_epoch_length : float, optional 
        maximum length (in seconds) for a (marker) epoch to last, by default
        None (no maximum)
    is_verbose : bool, optional
        print information about epochs found, by default True
    absolute_start : float
        absolute start time of session (potentially a while before first
        marker was detected) Optional, but is_verbose printout of when 
        markers are detected will not be accurate without this input.
    
    Returns
    -------
    output : list of lists of dicts
        one entry per epoch with duration strictly between `min_epoch_length`
        and `max_epoch_length`; each entry is a list with one dict of arrays
        per input in `data`, cut to that epoch
    """
    if is_verbose:
        print("== Splitting timecourse ==")
    timestamps = data[0]['timestamp']
    if absolute_start is None:
        if is_verbose:
            print("Times of epochs relative to FIRST MARKER DETECTION")
        t0 = copy.copy(timestamps[0])
    else:
        t0 = absolute_start
    if min_epoch_length is None:
        min_epoch_length = 0
    if max_epoch_length is None:
        max_epoch_length = np.inf
    break_indices, = np.nonzero(np.diff(timestamps) > max_epoch_gap)
    # np.diff shortens index; set correct w/ +1
    break_indices += 1
    break_indices = np.hstack([[0], break_indices, [len(timestamps)]])
    if is_verbose:
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
        if (epoch_duration > min_epoch_length) & (epoch_duration < max_epoch_length):
            output.append(this_epoch)
    if is_verbose:
        print('%d epochs found:' % (len(break_indices)-2))
        for x, dur in zip(break_indices[1:-1], epoch_durations):
            print('@ %d min, %.1f s : %.1f s long' % ((timestamps[x]-t0) // 60, (timestamps[x]-t0) % 60, dur))
        if np.isinf(max_epoch_length):
            max_length_str = 'inf'
        else:
            max_length_str = '%d'%max_epoch_length
        print('%d epochs meet duration limit (%d-%s seconds)' % (len(output), min_epoch_length, max_length_str))
    return output


def marker_cluster_stat(markers, fn=np.nanmedian, clusters=None, field='norm_pos', return_all_fields=True):
    """compute statistic (`fn`) for a given `field` for all clusters in data

    clusters can be provided; if they are not, this relies on `marker_cluster_index` field of input

    Parameters
    ----------
    markers : dict
        inputs (markers), with fields containing at least `field` kwarg
    fn : function, optional
        function to call on each cluster, by default np.nanmedian
    clusters : array-like, optional
        list or array of cluster indices, by default None
    field : str, optional
        field within `markers` for which to compute `fn`, by default 'norm_pos'
    return_all_fields : bool, optional
        whether to return all fields in `markers` (True) or just the computed statistic (False),
        by default True

    Returns
    -------
    out : dict of arrays or array
        If `return_all_fields` is False, an array with `fn` applied (along
        axis 0) to `field` for each cluster, one row per unique cluster index
        (sorted). If True, a dict of arrays with, for each cluster, the values
        of all fields at the single data point whose `field` value is closest
        to the cluster statistic (requires a 2D `field`).
    """
    if clusters is None:
        if 'marker_cluster_index' in markers:
            clusters = markers['marker_cluster_index']
        else:
            raise ValueError(("Please provide `clusters` kwarg if input dict \n"
                              "does not have `'marker_cluster_index'` field"))

    tmp_field_value = np.array([fn(markers[field][clusters == ci], axis=0)
                       for ci in np.unique(clusters)])
    if return_all_fields:
        mk_index = [np.argmin(np.abs(np.array(markers[field]) - mk_pos).mean(1))
                    for mk_pos in tmp_field_value]
        out = {}
        for k, v in markers.items():
            out[k] = np.array([v[ti] for ti in mk_index])
        return out
    else:
        return np.array(tmp_field_value)

def cluster_marker_points(markers,
                          pupil_left=None,
                          pupil_right=None,
                          cluster_by=("marker:timestamp",
                                      "marker:norm_pos",),
                          cluster_method='DBSCAN',
                          cluster_kw=None,
                          min_cluster_time=0.3,
                          max_cluster_time=5,
                          aspect_ratio=4/3,
                          max_cluster_std=2.0,
                          max_marker_movement=None,
                          max_pupil_movement=None,
                          normalize_time=True,
                          cut_cluster_outliers=True, 
                          min_n_clusters=5,
                          is_verbose=True,
                          ):
    """Find clusters of points in marker data

    Clusters marker detections (e.g. in time and position, so that each
    cluster is one period of fixation of one marker location), then removes
    clusters that are too short, too long, or too variable.

    Note that a 'marker_cluster_index' field is added in place to `markers`
    (and to `pupil_left` / `pupil_right` if given).
    
    Parameters
    ----------
    markers : dict of arrays
        marker detections, with at least the fields named in `cluster_by`
        and 'timestamp', 'norm_pos'
    pupil_left : dict of arrays, optional
        left pupil data matched in time to `markers`, by default None; needed
        only if referenced in `cluster_by` or if `max_pupil_movement` is set
    pupil_right : dict of arrays, optional
        right pupil data matched in time to `markers`, by default None (as
        for `pupil_left`)
    cluster_by : tuple, optional
        strings 'source:field' (source is 'marker', 'pupil_left' or
        'pupil_right') giving the features to cluster on, by default
        ("marker:timestamp", "marker:norm_pos")
    cluster_method : str, optional
        name of a clustering class in `sklearn.cluster`, by default 'DBSCAN'
    cluster_kw : dict, optional
        kwargs for the clustering class, by default None, which gives
        dict(eps=0.05) for DBSCAN and {} otherwise
    min_cluster_time : float, optional
        minimum duration (seconds) of a cluster to keep it, by default 0.3
    max_cluster_time : float, optional
        maximum duration (seconds) of a cluster to keep it, by default 5
    aspect_ratio : float, optional
        image aspect ratio (width / height) used to scale the x coordinate of
        'norm_pos', by default 4/3
    max_cluster_std : float, optional
        maximum mean (over x, y) standard deviation of aspect-corrected marker
        'norm_pos' within a cluster, by default 2.0; None skips this check
    max_marker_movement : float, optional
        maximum range (peak-to-peak) of aspect-corrected marker 'norm_pos' in
        x and in y within a cluster, by default None (no check)
    max_pupil_movement : float, optional
        maximum range of left and right pupil 'norm_pos' within a cluster, by
        default None (no check)
    normalize_time : bool, optional
        rescale 'timestamp' features as (t - min(t)) / 90 + 2 so that time is on
        a scale comparable to normalized position, by default True
    cut_cluster_outliers : bool, optional
        Whether to remove clusters with label of -1 (clustering
        algorithm has determined these to be outliers)
    min_n_clusters : int, optional
        if fewer clusters than this remain after filtering, return None, by
        default 5
    is_verbose : bool, optional
        print cluster counts and durations, by default True
    
    Returns
    -------
    markers_clustered : dict of arrays or None
        `markers` restricted to points in retained clusters, including the
        'marker_cluster_index' field; None if too few clusters remain

    Raises
    ------
    ValueError
        if any clustering feature contains NaNs or zeros
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
            # This should make time APPROXIMATELY 0-1.X over whatever the epoch time is
            assumed_epoch_time = 90 # Seconds; so time will have consistent
                                    # spacing for all runs to be clustered.
            tmp = (tmp-np.min(tmp)) / assumed_epoch_time #np.ptp(tmp)
            # This +2 helps, for whatever reason. Something about the range of
            # 0-0.5 or 0-1 or so does not work well with clustering. Unclear why, 
            # perhaps worth exploring. A thought: perhaps differentiating the
            # range of time from the range of spatial position is what helps?
            # (spatial position, for norm_pos, is [0,1.33]; time, with this
            # addition, is [2,~3.5])
            tmp += 2
            #print('Normalized time range:')
            #print(tmp.min(), tmp.max())
        if (key in ('norm_pos', )) and (marker_type == 'marker'):
            tmp *= np.array([[aspect_ratio, 1.0]])
        to_cluster.append(tmp)
    to_cluster = np.hstack(to_cluster)
    # Clustering parameters
    if cluster_kw is None:
        if cluster_method=='DBSCAN':
            # Default params for default method:
            cluster_kw = dict(eps=0.05)
        else:
            # IDK what I should do, do nothing:
            cluster_kw = {}
    # Check for NaNs, zeros
    to_kill = np.any(np.isnan(to_cluster), axis=1) | \
              np.any(to_cluster == 0, axis=1)
    if np.any(to_kill):
        raise ValueError("zeros or nans must be removed before clustering!")
    # Do clustering
    cluster_fn = getattr(cluster, cluster_method)
    cluster_mapper = cluster_fn(**cluster_kw)
    groups = cluster_mapper.fit_predict(to_cluster)
    unique_groups = np.unique(groups)
    if is_verbose:
        print('Found %d groups, w/ durations:' % (len(unique_groups)))
    # Add 'marker_cluster_index' field to all dicts
    markers['marker_cluster_index'] = groups
    if pupil_left is not None:
        pupil_left['marker_cluster_index'] = groups
    if pupil_right is not None:
        pupil_right['marker_cluster_index'] = groups
    # Compute various quantities for clusters, assure that clusters are within acceptable ranges for each    
    if cut_cluster_outliers:
        # Remove clusters with -1 label, these are outliers
        keep_clusters_binary = unique_groups >= 0
    else:
        keep_clusters_binary = np.ones_like(unique_groups) > 0
    # Keep clusters that are within a specified interval of time
    group_durations = marker_cluster_stat(markers, field='timestamp', 
                                         fn=np.ptp, return_all_fields=False)
    if is_verbose:
        print(group_durations)
    if min_cluster_time is not None:
        keep_clusters_binary &= (group_durations > min_cluster_time)
    if max_cluster_time is not None:
        keep_clusters_binary &= (group_durations < max_cluster_time)
    if is_verbose:
        print('%d groups after duration filtering' % (keep_clusters_binary.sum()))
    # Keep clusters that are below a threshold for jitter / movement of markers or pupils
    if max_marker_movement is not None:
        group_jitter = marker_cluster_stat(markers, field='norm_pos', 
                                           fn=np.ptp, return_all_fields=False)
        # Account for aspect ratio
        group_jitter[:, 0] *= aspect_ratio
        # Assure that both X and Y position change are less than threshold
        keep_clusters_binary &= np.all(group_jitter < max_marker_movement, axis=1)
        if is_verbose:
            print('%d groups after marker jitter filtering' %
              (keep_clusters_binary.sum()))
    if max_pupil_movement is not None:
        pupil_jitter_left = marker_cluster_stat(pupil_left, field='norm_pos',
                                               fn=np.ptp, return_all_fields=False)
        keep_clusters_binary &= np.all(pupil_jitter_left < max_pupil_movement, axis=1)
        pupil_jitter_right = marker_cluster_stat(pupil_right, field='norm_pos',
                                                fn=np.ptp, return_all_fields=False)
        keep_clusters_binary &= np.all(pupil_jitter_right < max_pupil_movement, axis=1)
        if is_verbose:
            print('%d groups after pupil jitter filtering' %
                  (keep_clusters_binary.sum()))
    # Keep clusters that have a standard deviation lower than a set threshold
    if max_cluster_std is not None:
        marker_cluster_stds = marker_cluster_stat(markers, field='norm_pos',
                                           fn=np.std, return_all_fields=False)
        marker_cluster_stds[:, 0] *= aspect_ratio
        keep_clusters_binary &= (marker_cluster_stds.mean(1) < max_cluster_std)
        if is_verbose:
            print('%d groups after marker std filtering' %
                  (keep_clusters_binary.sum()))
    keep_cluster_numbers = unique_groups[keep_clusters_binary]
    if len(keep_cluster_numbers) < min_n_clusters:
        # Too few clusters meet required parameters
        return None
    keep_clusters_i = np.isin(groups, keep_cluster_numbers)
    if keep_clusters_i.sum() == 0:
        # No groups meet threshold
        return None
    else:
        return utils.filter_arraydict(markers, keep_clusters_i)


def find_epochs(marker,
                all_timestamps,
                pupil_left=None,
                pupil_right=None,
                max_epoch_gap=15,
                min_epoch_length=30,
                max_epoch_length=150,
                do_duration_pre_check=True,
                duration_threshold=0.3,
                size_std_threshold=None,
                bimodal_std_threshold=2.5, 
                aspect_ratio_threshold=1.75,
                aspect_ratio_keep='less_than_threshold',
                aspect_ratio_type='max/min',
                image_aspect_ratio=4/3,
                do_cluster_clean=True,
                min_n_clusters=5,
                cluster_method='DBSCAN',
                cluster_by=("marker:timestamp",
                            "marker:norm_pos",),
                cluster_kw=None,
                cut_cluster_outliers=True,
                min_cluster_time=0.3,
                max_cluster_time=300,
                max_cluster_std=4.0,
                max_marker_movement=0.1, # 10% of screen vertically
                max_pupil_movement=None,
                is_verbose=True,
                ):
    """Find epochs of matched points within lists of pupil detections & marker detections

        Filters marker detections (`remove_brief_detections`,
        `remove_small_detections`), splits them into epochs
        (`split_timecourse`), and optionally cleans each epoch by clustering
        (`cluster_marker_points`). Not referenced by the shipped config
        files; see `filter_and_cluster` / `filter_and_split`.
        
        Parameters
        ----------
        marker : dict of arrays
            marker detections (output of a `marker_detection` function)
        all_timestamps : array-like
            all timestamps for session on relevant clock (probably world camera clock)
        pupil_left : dict of arrays or None
            pupil detections for this session, if used in finding epochs (None if not).
            Not implemented: anything other than None raises NotImplementedError.
        pupil_right : dict of arrays or None
            as `pupil_left`
        max_epoch_gap : float, optional
            maximum time gap (seconds) to consider subsequent marker detections to be
            within the same epoch of (calibration or validation), by default 15
        min_epoch_length : float, optional
            minimum time (seconds) from first to last marker detection to consider an
            epoch to be worth saving, by default 30
        max_epoch_length : float, optional
            maximum epoch duration (seconds), by default 150
        do_duration_pre_check : bool, optional
            Whether to check for too-short detections that are assumed to be spurious
        duration_threshold : float, optional
            Minimum length for detected markers to be present (if do_duration_pre_check is True)
        size_std_threshold, bimodal_std_threshold, aspect_ratio_threshold, aspect_ratio_keep, aspect_ratio_type, image_aspect_ratio
            passed to `remove_small_detections`
        do_cluster_clean : bool, optional
            whether to clean each epoch by clustering, by default True
        min_n_clusters : int, optional
            minimum number of clusters for an epoch to be kept, by default 5.
            Note that epochs are kept only with MORE than this many clusters.
        cluster_method : str, optional
            Method for clustering from sklearn.cluster
        cluster_by : tuple, optional
            list of strings of properties over which to cluster. Each string is
            'input_type:property', e.g. 'marker:norm_pos'
        cluster_kw, cut_cluster_outliers, min_cluster_time, max_cluster_time, max_cluster_std, max_marker_movement, max_pupil_movement
            passed to `cluster_marker_points`
        is_verbose : bool, optional
            print progress information, by default True

        Returns
        -------
        epochs : list of dicts
            one dict of arrays of marker detections per retained epoch (with
            'marker_cluster_index' if `do_cluster_clean`)
        """

    # Optionally clean up marker detections to remove duplicate time stamps and too-short detections
    if do_duration_pre_check:
        marker = remove_brief_detections(marker, all_timestamps,
                                         duration_threshold=duration_threshold)
    marker = remove_small_detections(marker, 
                                     size_std_threshold=size_std_threshold,
                                     bimodal_std_threshold=bimodal_std_threshold,
                                     image_aspect_ratio=image_aspect_ratio,
                                     aspect_ratio_threshold=aspect_ratio_threshold,
                                     aspect_ratio_type=aspect_ratio_type,
                                     aspect_ratio_keep=aspect_ratio_keep,
                                     )
    # Match time points to compare markers w/ eye data
    to_split = [marker]
    # Make window width half median frame rate, by default
    frame_time = np.median(np.diff(all_timestamps))
    window = frame_time / 2
    if pupil_left is not None:
        raise NotImplementedError(("If you wish to use pupil detections to help find clusters\n"
                                   "of markers, you need to filter both pupils and markers by\n"
                                   "the EXISTENCE of pupil detections (can't be any NaNs or 0s\n"
                                   "in pupil detections, this messes up clustering). This is \n"
                                   "not implemented yet!"))
        pupil_left_match = utils.match_time_points(
        marker, pupil_left, window=window)
        to_split.append(pupil_left_match)
    if pupil_right is not None:
        raise NotImplementedError(("If you wish to use pupil detections to help find clusters\n"
                                   "of markers, you need to filter both pupils and markers by\n"
                                   "the EXISTENCE of pupil detections (can't be any NaNs or 0s\n"
                                   "in pupil detections, this messes up clustering). This is \n"
                                   "not implemented yet!"))
        pupil_right_match = utils.match_time_points(
            marker, pupil_right, window=window)
        to_split.append(pupil_right_match)
    
    # Split into multiple epochs of calibration, validation as needed
    epochs = split_timecourse(*to_split,
                              max_epoch_gap=max_epoch_gap,
                              min_epoch_length=min_epoch_length,
                              max_epoch_length=max_epoch_length,
                              is_verbose=is_verbose)

    # Clean up w/ clustering
    if do_cluster_clean:
        #epochs_orig = copy.deepcopy(epochs)
        epochs_to_keep = []
        for i, ce in enumerate(epochs):
            if is_verbose:
                print("\n> For epoch %d" % i)
            # Run clustering & filtering by cluster time
            epoch_grpclean = cluster_marker_points(*ce,
                                                   cluster_method=cluster_method,
                                                   cluster_by=cluster_by,
                                                   cluster_kw=cluster_kw,
                                                   min_cluster_time=min_cluster_time,
                                                   max_cluster_time=max_cluster_time,
                                                   max_cluster_std=max_cluster_std,
                                                   max_marker_movement=max_marker_movement,  # 10% of screen vertically
                                                   max_pupil_movement=max_pupil_movement,
                                                   cut_cluster_outliers=cut_cluster_outliers,
                                                   min_n_clusters=min_n_clusters,
                                                   is_verbose=is_verbose)
            if epoch_grpclean is None:
                if is_verbose:
                    print("All clusters excluded!")
                continue
            grps = epoch_grpclean['marker_cluster_index']
            # Apply min number of clusters as threshold for keeping calibration epochs
            if len(np.unique(grps)) > min_n_clusters:
                epochs_to_keep.append(epoch_grpclean)
                if is_verbose:
                    print("Modified length:", len(grps), '(%0.2f%% of original)' %
                        (len(grps) / len(ce[0]['timestamp']) * 100))
        epochs = epochs_to_keep
        n_epochs = len(epochs)
        if is_verbose:
            print("%d epochs after cluster filtering" % n_epochs)
    else: 
        # Return only markers; pupils are redundant, pupil times
        # can be matched with these from full pupil estimates.
        epochs = [ee[0] for ee in epochs]
    return epochs


def filter_and_split(marker,
                all_timestamps,
                max_epoch_gap=15,
                min_epoch_length=30,
                max_epoch_length=150,
                do_duration_pre_check=True,
                duration_threshold=0.3,
                size_std_threshold=None,
                bimodal_std_threshold=2.5, 
                aspect_ratio_threshold=1.75,
                aspect_ratio_keep='less_than_threshold',
                aspect_ratio_type='max/min',
                image_aspect_ratio=4/3,
                is_verbose=True,
                ):
    """Filter marker detections and split into multiple epochs as needed

    Runs `remove_brief_detections` (optional), `remove_small_detections`,
    and `split_timecourse`. Intended to be called by
    `pipelines.marker_splitting` (``config/marker_parsing-split_*.yaml``).

    Parameters
    ----------
    marker : dict of arrays
        marker detections (output of a `marker_detection` function)
    all_timestamps : array-like
        all timestamps for the world camera
    max_epoch_gap, min_epoch_length, max_epoch_length
        passed to `split_timecourse`
    do_duration_pre_check : bool, optional
        whether to run `remove_brief_detections`, by default True
    duration_threshold : float, optional
        passed to `remove_brief_detections`, by default 0.3
    size_std_threshold, bimodal_std_threshold, aspect_ratio_threshold, aspect_ratio_keep, aspect_ratio_type, image_aspect_ratio
        passed to `remove_small_detections`
    is_verbose : bool, optional
        passed to `split_timecourse`, by default True

    Returns
    -------
    None
        NOTE: the computed `epochs` are currently not returned (there is no
        return statement), so this function always returns None.
    """
    if do_duration_pre_check:
        marker = remove_brief_detections(marker, all_timestamps,
                                         duration_threshold=duration_threshold)
    marker = remove_small_detections(marker, 
                                     size_std_threshold=size_std_threshold,
                                     bimodal_std_threshold=bimodal_std_threshold,
                                     image_aspect_ratio=image_aspect_ratio,
                                     aspect_ratio_threshold=aspect_ratio_threshold,
                                     aspect_ratio_type=aspect_ratio_type,
                                     aspect_ratio_keep=aspect_ratio_keep,
                                     )
    
    # Split into multiple epochs of calibration, validation as needed
    epochs = split_timecourse(marker,
                              max_epoch_gap=max_epoch_gap,
                              min_epoch_length=min_epoch_length,
                              max_epoch_length=max_epoch_length,
                              is_verbose=is_verbose)

def filter_and_cluster(marker,
                all_timestamps,
                do_duration_pre_check=True,
                duration_threshold=0.3,
                size_std_threshold=None,
                bimodal_std_threshold=2.5, 
                aspect_ratio_threshold=1.75, # for circles
                aspect_ratio_keep='less_than_threshold',
                aspect_ratio_type='max/min',
                image_aspect_ratio=4/3,
                min_n_clusters=5,
                cluster_method='DBSCAN',
                cluster_by=("marker:timestamp",
                            "marker:norm_pos",),
                cluster_kw=None,
                cut_cluster_outliers=True,
                min_cluster_time=0.3,
                max_cluster_time=300,
                max_cluster_std=4.0,
                max_marker_movement=0.1, # 10% of screen vertically
                max_pupil_movement=None,
                is_verbose=True,):
    """Filter marker detections for one epoch and cluster them

    Runs `remove_brief_detections` (optional), `remove_small_detections`,
    then `cluster_marker_points` (markers only, no pupil data). This is the
    function named in ``config/marker_parsing-cluster_*.yaml`` and run by
    `pipelines.marker_clustering`.

    Parameters
    ----------
    marker : dict of arrays
        marker detections for one epoch (output of a `marker_detection` function)
    all_timestamps : array-like
        all timestamps for the world camera
    do_duration_pre_check : bool, optional
        whether to run `remove_brief_detections`, by default True
    duration_threshold : float, optional
        minimum duration (seconds) of runs of detections, passed to
        `remove_brief_detections`, by default 0.3
    size_std_threshold, bimodal_std_threshold, aspect_ratio_threshold, aspect_ratio_keep, aspect_ratio_type, image_aspect_ratio
        passed to `remove_small_detections` (defaults 1.75, 'less_than_threshold',
        'max/min', 4/3 are set for circle markers)
    min_n_clusters, cluster_method, cluster_by, cluster_kw, cut_cluster_outliers, min_cluster_time, max_cluster_time, max_cluster_std, max_marker_movement, max_pupil_movement
        passed to `cluster_marker_points`
    is_verbose : bool, optional
        passed to `cluster_marker_points`, by default True

    Returns
    -------
    marker_clustered : dict of arrays or None
        filtered markers with a 'marker_cluster_index' field, or None if fewer
        than `min_n_clusters` clusters survive
    """
    
    if do_duration_pre_check:
        marker = remove_brief_detections(marker, all_timestamps,
                                         duration_threshold=duration_threshold)
        
    marker = remove_small_detections(marker, 
                                     size_std_threshold=size_std_threshold,
                                     bimodal_std_threshold=bimodal_std_threshold,
                                     image_aspect_ratio=image_aspect_ratio,
                                     aspect_ratio_threshold=aspect_ratio_threshold,
                                     aspect_ratio_type=aspect_ratio_type,
                                     aspect_ratio_keep=aspect_ratio_keep,
                                     )

    marker_clustered = cluster_marker_points(marker,
                                            cluster_method=cluster_method,
                                            cluster_by=cluster_by,
                                            cluster_kw=cluster_kw,
                                            min_cluster_time=min_cluster_time,
                                            max_cluster_time=max_cluster_time,
                                            max_cluster_std=max_cluster_std,
                                            max_marker_movement=max_marker_movement,  # 10% of screen vertically
                                            max_pupil_movement=max_pupil_movement,
                                            cut_cluster_outliers=cut_cluster_outliers,
                                            min_n_clusters=min_n_clusters,
                                            is_verbose=is_verbose)
    return marker_clustered