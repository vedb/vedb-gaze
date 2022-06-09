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
            if is_verbose:
                print("Changed: %.10f" %(time_values_clean[jj]))
            time_values_clean[jj] = all_timestamps[ii]
            if is_verbose:
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
        Marker data, with fields 'norm_pos', 'size'
    size_std_threshold : float, optional
        number of standard devations below median to remove, by default 2
    aspect_ratio_threshold : float, optional
        Maximum aspect ratio for detected markers
    image_aspect_ratio : float, optional
        Aspect ratio of image in which 'norm_pos' is calculated. 'norm_pos' field 
        will be scaled 0-1 for both axes, so if aspect ratio of IMAGE is not 1, aspect
        ratio of markers will not be correct without correcting for image aspect ratio.
    aspect_ratio_type : str, optional
        string indicating how aspect ratio is computed for marker aspect ratio; 
        either 'x/y' (width / height) or 'max/min' (if you don't care about which way 
        is up for marker aspect ratio)
    return_rejects : bool, optional
        Flag to return rejected markers instead of kept ones (useful for comparison
        of what is rejected vs what is kept with different settings)
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
        Description
    max_epoch_gap : int, optional
        Description
    min_epoch_length : float, optional
        minimum length (in seconds) for a (marker) epoch to last
    max_epoch_length : float, optional 
        maximum length (in seconds) for a (marker) epoch to last
    is_verbose : bool, optional
        Description
    absolute_start : float
        absolute start time of session (potentially a while before first
        marker was detected) Optional, but is_verbose printout of when 
        markers are detected will not be accurate without this input.
    
    Returns
    -------
    TYPE
        Description
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
                          cut_cluster_outliers=False, # Should be True, False for backward compatibility
                          is_verbose=True,
                          ):
    """Find clusters of points in marker data
    
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
    cut_cluster_outliers : bool, optional
        Whether to remove clusters with label of -1 (clustering
        algorithm has determined these to be outliers)
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

    keep_clusters_i = np.in1d(groups, keep_cluster_numbers)
    if keep_clusters_i.sum() == 0:
        # No groups meet threshold
        return None
    else:
        return utils.filter_arraydict(markers, keep_clusters_i)

# Note: Commented out values here should be defaults, but for inertia - 
# some prior calls to this function did not specify defaults but still
# need to re-run without those defaults. Need to update param dicts with
# old defaults before changing these. Eventually.
def find_epochs(marker,
                all_timestamps,
                pupil_left=None,
                pupil_right=None,
                max_epoch_gap=15,
                min_epoch_length=30,
                max_epoch_length=None, # 150
                do_duration_pre_check=True,
                duration_threshold=0.3,
                size_std_threshold=None,
                bimodal_std_threshold=None, # 2.5, 
                aspect_ratio_threshold=None, # 1.75,
                aspect_ratio_keep='less_than_threshold',
                aspect_ratio_type='max/min',
                image_aspect_ratio=4/3,
                do_cluster_clean=True,
                min_n_clusters=5,
                cluster_method='DBSCAN',
                cluster_by=("marker:timestamp",
                            "marker:norm_pos",),
                cluster_kw=None,
                min_cluster_time=0.3,
                max_cluster_time=3.0,
                max_cluster_std=4.0,
                max_marker_movement=None, # 0.10,  # 10% of screen vertically
                max_pupil_movement=None,
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
