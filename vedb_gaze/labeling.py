# vedb_gaze_labeling

# Blink detection WIP
import plot_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import tqdm.notebook
#
import scipy
import scipy.interpolate
import scipy.signal
from scipy.signal import savgol_filter
from scipy.stats import zscore
from scipy.spatial import distance
from numpy.polynomial import polynomial as P
from sklearn.decomposition import PCA

from .utils import onoff_from_binary, time_to_index, resample_data
from .externals.remodnav import EyegazeClassifier
# Need functions to: 
# Smooth (just for gaze) w/ awareness of noise frequency bandwidth
# Resample
# Maybe resample AND smooth in one step
# label saccades
# label blinks
# label VOR
# - optic flow, odometry, ...
#  
try:
    # Eyelid distance will require this.
    import pylids
    has_pylids = True
except ImportError:
    print('No pylids. please install.')
    has_pylids = False
    pass

# Basics

def buffer_onoff(onoff, buffer_time):
    """Take list of onsets and offsets and add time before and after

    Parameters
    ----------
    onoff: array-like
        array or list with tuples, each row (or item of list) should be
        (onset, offset, duration)
    buffer time : scalar, array-like
        if scalar, same `buffer_time` is added before onsets & after offsets
        if tuple, list, or array, should be 2 long, with separate values
        for pre-onset and post-offset buffers

    """
    if isinstance(buffer_time, (list, tuple)):
        pre, post = buffer_time
    else:
        pre = post = buffer_time
    out = []
    for on, off, duration in onoff:
        out.append([on-pre, off+post, duration+pre+post])
    return np.asarray(out)

def remove_outliers(timestamps, data, 
                    z_max=4,
                    z_min=None,
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
    """
    keep = np.ones_like(data) > 0
    # First, remove absolute threshold out-of-bounds
    if absolute_min is not None:
        keep &= (data >= absolute_min)
    if absolute_max is not None:
        keep &= (data <= absolute_max)
    if z_max is not None:
        data_z = zscore(data)
        if z_min is None:
            keep &= (np.abs(data_z) <= z_max)
        else:
            keep &= ((data_z <= z_max) & (data_z >= z_min))
    # Alternatively, set to nans, or return outlier index?
    return timestamps[keep], data[keep]

def compute_eye_velocity(gaze, max_size_deg=125, aspect_ratio=4/3):
    """Compute eye velocity from gaze

    Recommend filter blinks first? Nan out? Do NOT filter, then compare notes
    across blink and saccade detection?

    Weight by confidence?

    """
    t = np.array(gaze['timestamp'])
    x, y = np.array(gaze['norm_pos']).T
    xy = np.array(gaze['norm_pos'])
    confidence = np.array(gaze['confidence'])

    #screen_size_deg = np.array([101, 101 / aspect_ratio])
    screen_size_deg = np.array([max_size_deg, max_size_deg / aspect_ratio])
    # Convert to (approximate) degrees prior to computing velocity
    xy_deg = xy * screen_size_deg
    # Compute eye velocity with gradients over space and time
    delta_pos = np.gradient(xy_deg, axis=0)
    displacement = np.linalg.norm(delta_pos, axis=1)
    delta_t = np.gradient(t)
    eye_velocity = displacement / delta_t
    return eye_velocity


# Blink utility functions
def assure_positive(pca, verbose=False):
    """Assure largest components of a PCA fit are positive
    
    If you can reasonably assume PCs for e.g. 2D data don't 
    rotate beyond 90 degrees, this may be useful
    
    Parameters
    ----------
    pca : sklearn.decomposition.PCA object
        pca object that has already been fit
    verbose : bool
        verbosity setting (True = talkative)
    """
    # Assure positive PCs
    j = np.argmax(np.abs(pca.components_[0]))
    if pca.components_[0][j] < 0:
        if verbose:
            print("flipping 1st PC")
        pca.components_[0] = -pca.components_[0]
    j = np.argmax(np.abs(pca.components_[1]))
    if pca.components_[1][j] < 0:
        if verbose:
            print("flipping 2nd PC")
        pca.components_[1] = -pca.components_[1]
    return pca

def get_major_minor_axes_pca(pupil_data, n_points=1000, assure_positive_pcs=True):
    """Estimates PCA to compute major and minor axes of the eye across all frames
    """
    if not has_pylids:
        raise ImportError("No pylids module, please install.")
    idx = np.linspace(0, len(pupil_data['dlc_kpts_x']), n_points, endpoint=False).astype(int)
    pca_data = [] # np.zeros((2 * int(np.floor(len(pupil_data['dlc_kpts_x']) / nth)), 2))
    for i in idx:
        x = pupil_data['dlc_kpts_x'][i]
        y = pupil_data['dlc_kpts_y'][i]
        u_, l_ = pylids.utils.parse_keypoints(x, y)
        pca_data.append(u_)
        pca_data.append(l_)
    pca_data = np.vstack(pca_data)
    eyelid_pcs = PCA()
    eyelid_pcs.fit(pca_data)
    if assure_positive_pcs:
        eyelid_pcs = assure_positive(eyelid_pcs)
    return eyelid_pcs


def get_eyelid_distance_coarse_to_fine(x_new, coefs_up, coefs_lo, eyelid_resolution_coarse=100, eyelid_resolution_fine=100):
    """Searches for maximum distance b/w eyelids and uses that as the 
    distance b/w eyelids. First finds max distance for 100 points along
    the eyelid (coarse) then searches for 100 points in the neighbourhood 
    (fine) of this point.

    Args:
        x_new (array):   
        coefs_up (array): polynomial coeffs for upper eyelid
        coefs_lo (array): polynomial coeffs for lower eyelid

    Returns:
        dist_eyelids (array): distance b/w eyelids for each frame
    """
    dist_ilid = []
    dist_coarse = np.zeros(eyelid_resolution_coarse)
    dist_fine = np.zeros(eyelid_resolution_fine)
    # Coarse distance estimate
    x_temp = np.linspace(x_new[0], x_new[-1], eyelid_resolution_coarse)
    fit_up = P.polyval(x_temp, coefs_up)
    fit_lo = P.polyval(x_temp, coefs_lo)
    for j in range(eyelid_resolution_coarse):
        dist_coarse[j] = distance.euclidean([x_temp[j], fit_up[j]],
                                            [x_temp[j], fit_lo[j]])
    # Fine sampling near max
    x_temp2 = np.linspace(
        x_temp[np.argmax(dist_coarse)-1], x_temp[np.argmax(dist_coarse)], eyelid_resolution_fine)
    fit_up = P.polyval(x_temp2, coefs_up)
    fit_lo = P.polyval(x_temp2, coefs_lo)
    for k in range(eyelid_resolution_fine):
        dist_fine[k] = distance.euclidean([x_temp2[k], fit_up[k]],
                                          [x_temp2[k], fit_lo[k]])

    dist_ilid = np.append(dist_ilid, np.max(dist_fine))
    return dist_ilid


def get_eyelid_distance(pupil_data, 
                           eyelid_resolution_coarse=100,
                           eyelid_resolution_fine=100,
                           align_eye_pca = True,
                           n_points_pca=1000,
                           idx=None,
                           assure_positive_pcs=True,
                           save_fits=False,
                           coarse_to_fine_estimate=False,
                           progress_bar=tqdm.notebook.tqdm,
                          ):
    """TODO: docstring
    """
    if not has_pylids:
        raise ImportError("No pylids module, please install.")    
    
    if save_fits:
        fits = dict(x=[],
                    y_upper_coef=[],
                    y_lower_coef=[],
                    y_upper_est=[],
                    y_lower_est=[],                
                   )
    if align_eye_pca:
        eyelid_pcs = get_major_minor_axes_pca(pupil_data, 
                                              n_points=n_points_pca, 
                                              assure_positive_pcs=assure_positive_pcs)

    if idx is None:
        idx = np.arange(len(pupil_data['dlc_kpts_x']))

    dst = []
    for j in progress_bar(idx):
        x = pupil_data['dlc_kpts_x'][j]
        y = pupil_data['dlc_kpts_y'][j]
        c = pupil_data['dlc_confidence'][j]
        if x.shape[0] == 48:
            pupil_model_type = 'eyelid_pupil'
        elif x.shape[0] == 32:
            pupil_model_type = 'eyelid'
        if align_eye_pca:
            # rotate eyelid keypoints but maintain original mean, to maintain valid
            # assumptions (we hope) about x and y range
            x_, y_ = (eyelid_pcs.transform(np.vstack([x, y]).T) + eyelid_pcs.mean_).T
        else:
            x_, y_ = x, y
        x_viz_eye, fit_eye_up, fit_eye_lo, corners_x, coefs_up, coefs_lo = \
            pylids.fit_eyelid(x_, y_, c, return_full_eyelid=True, model_type=pupil_model_type)
        if save_fits:
            fits['x'].append(x_viz_eye)
            fits['y_upper_coef'].append(coefs_up)
            fits['y_lower_coef'].append(coefs_lo)
            fits['y_upper_est'].append(fit_eye_up)
            fits['y_lower_est'].append(fit_eye_lo)
        if coarse_to_fine_estimate:
            dst.append(get_eyelid_distance_coarse_to_fine(
                x_viz_eye, coefs_up, coefs_lo,
                eyelid_resolution_coarse=eyelid_resolution_coarse, 
                eyelid_resolution_fine=eyelid_resolution_fine))
        else:
            # These should all be perpendicular to main axis of eye if PCA has been applied.
            # (this is much faster)
            dst.append(np.max(np.abs(fit_eye_up - fit_eye_lo)))
    dst = np.asarray(dst)
    return dst


# Blinks

# Values derived from labeled blinks in GitW
sig_str = 4 * 0.19 # ??
sig_end = 3 * 0.19 # ??
m = 0.02 # mean
negative_velocity_threshold = m - sig_str
positive_velocity_threshold = m + sig_end

def _detect_blinks_eyevel(dist_eyelid, 
    fps=120, 
    min_eye_closing_time = 10,
    max_eye_closing_time = 250,
    max_full_closure_time = 17,
    min_eye_opening_time = 30,
    min_full_blink_time = 16,
    max_full_blink_time = 500,
    negative_velocity_threshold=-2.4,
    positive_velocity_threshold=2.4,
    ): 
    """
    Note: All parameter times in milliseconds
    """
    # Z-score and scale resulting range to -1 to 1
    # eyelid_velocity = pylids.filter_scale_blinks(dist_eyelid)
    # ... or just compute gradient
    # eyelid_velocity is expressed in fraction of max eye change 
    # per second
    if dist_eyelid.max() > 1:
        dist_eyelid = dist_eyelid / dist_eyelid.max()
    eyelid_velocity = np.gradient(dist_eyelid) / (1/fps)
    pred_blink_labels = np.zeros((len(eyelid_velocity),))
    blink_label = 1
    i = 0
    done = False
    while i < (len(eyelid_velocity)-1):
        if eyelid_velocity[i] <= negative_velocity_threshold:
            blink_start = i
            while eyelid_velocity[i] <= negative_velocity_threshold:
                blink_end = i
                i += 1
                if i > (len(eyelid_velocity)-1):
                    done = True
                    break
            if (blink_end-blink_start) * (1000/fps) < max_eye_closing_time and \
               (blink_end-blink_start) * (1000/fps) > min_eye_closing_time and \
                not done:
                blink_mid = i
                while eyelid_velocity[i] > negative_velocity_threshold and \
                      eyelid_velocity[i] < positive_velocity_threshold:
                    blink_end = i
                    i += 1
                    if i > (len(eyelid_velocity)-1):
                        done = True
                        break

                if (blink_mid-blink_end) * (1000/fps) < max_full_closure_time and \
                    not done:
                    blink_last = i
                    while eyelid_velocity[i] > positive_velocity_threshold:
                        blink_end = i
                        i += 1
                        if i > (len(eyelid_velocity)-1):
                            done = True
                            break

                    # and min(eyelid_velocity[blink_start:blink_end])< 0.53:
                    if (blink_end-blink_last) * (1000/fps) > min_eye_opening_time and \
                        (blink_end-blink_start) * (1000/fps) < max_full_blink_time and \
                        (blink_end-blink_start) * (1000/fps) > min_full_blink_time and \
                        not done:
                        pred_blink_labels[blink_start:blink_end] = blink_label
        i += 1

    return pred_blink_labels

def detect_blinks(pupil_data,
                  fps=120,
                  min_eye_closing_time = 10,
                  max_eye_closing_time = 250,
                  max_full_closure_time = 17,
                  min_eye_opening_time = 30,
                  min_full_blink_time = 16,
                  max_full_blink_time = 500,
                  negative_velocity_threshold=-2.4, # fraction of eye per second #-0.02,
                  positive_velocity_threshold=2.4, # fraction of eye per second #0.02,
                  absolute_max_distance = 400,
                  outlier_z_max=4,
                  max_eye_opening=None,
                  idx=None
                 ):
    """
    All closing, opening, blink times in ms
    velocity should be converted to % closure / second
    """
    # Fixed parameters
    resampling_method = 'thin-plate_spline'
    smoothing = 0.001
    neighbors = 7
    
    orig_time = pupil_data['timestamp']
    ts = orig_time.copy()
    if idx is not None:
        ts = ts[idx]
    dst = get_eyelid_distance(pupil_data, idx=idx,)
    # Remove outliers in distance
    ts_, dst_ = remove_outliers(ts, dst,
                                absolute_max=absolute_max_distance, 
                                absolute_min=0,
                                z_max=outlier_z_max,
                                z_min=-np.inf)
    # Resample with slight smoothing in time by thin-plate spline smoothing spline
    ts_, dst_ = resample_data(ts_, dst_, fps=fps, 
                              method=resampling_method,
                              neighbors=neighbors,
                              smoothing=smoothing)
    ts_ = ts_.flatten()
    dst_ = dst_.flatten()
    # Convert distance to proportion of max eye opening for this data
    if max_eye_opening is None:
        max_eye_opening = dst_.max()
    dst_fraction = dst_ / max_eye_opening
    # Find blinks based on eye velocity
    blink_index_resampled_time = _detect_blinks_eyevel(dst_fraction, fps=fps, 
                                                    min_eye_closing_time=min_eye_closing_time,
                                                    max_eye_closing_time=max_eye_closing_time,
                                                    max_full_closure_time=max_full_closure_time,
                                                    min_eye_opening_time=min_eye_opening_time,
                                                    min_full_blink_time=min_full_blink_time,
                                                    max_full_blink_time=max_full_blink_time,
                                                    negative_velocity_threshold=negative_velocity_threshold,
                                                    positive_velocity_threshold=positive_velocity_threshold)
    # Blink index is 
    blink_onoff_resampled_time = onoff_from_binary(blink_index_resampled_time)
    blink_times_resampled_time = [(ts_[st], ts_[fin], dur*1/fps) for st, fin, dur in blink_onoff_resampled_time]
    #tt = np.asarray([(ts_[st], ts_[fin]) for st, fin, dur in blink_onoff_resampled_time])
    #blink_onoff_orig_time = vedb_gaze.utils.time_to_index(tt, ts).astype(int)
    #blink_index_orig_time = vedb_gaze.utils.onoff_to_binary(blink_onoff_orig_time, len(ts))
    
    return dict(timestamp=ts_,
                distance=dst_, 
                timestamp_orig=ts,
                distance_orig=dst,
                blinks_onoff=blink_times_resampled_time,
               )


def detect_blinks_confidence(pupil_data,
                  fps=120,
                  min_confidence = 0.7,
                  min_full_blink_time = 16 / 1000,
                  max_full_blink_time = 500 / 1000,
                  idx=None
                 ):
    """
    if min_confidence is None, use 1 std below median
    All closing, opening, blink times in ms
    velocity should be converted to % closure / second

    """
    # Fixed parameters
    resampling_method = 'thin-plate_spline'
    smoothing = 0.001
    neighbors = 7
    
    orig_time = pupil_data['timestamp']
    conf = pupil_data['confidence']
    ts = orig_time.copy()
    if idx is not None:
        ts = ts[idx]
        conf = conf[idx]
    if min_confidence is None:
        # I think this shoudl be 2 stds below mean...
        min_confidence = np.median(conf) - np.std(conf)
    # Resample with slight smoothing in time by thin-plate spline smoothing spline
    ts_, conf_ = resample_data(ts, conf, fps=fps, 
                              method=resampling_method,
                              neighbors=neighbors,
                              smoothing=smoothing)
    ts_ = ts_.flatten()
    conf_ = conf_.flatten()
    blink_index_resampled_time = conf_ < min_confidence
    # Find blinks based on eye velocity
    #blink_index_resampled_time = detect_blinks_eyevel(conf_, fps=fps, 
    #                                                  negative_velocity_threshold=negative_velocity_threshold,
    #                                                  positive_velocity_threshold=positive_velocity_threshold)
    
    
    # Blink index is 
    blink_onoff_resampled_time = onoff_from_binary(blink_index_resampled_time)
    blink_times_resampled_time = [(ts_[st], ts_[fin], dur*1/fps) for st, fin, dur in blink_onoff_resampled_time]
    #tt = np.asarray([(ts_[st], ts_[fin]) for st, fin, dur in blink_onoff_resampled_time])
    #blink_onoff_orig_time = vedb_gaze.utils.time_to_index(tt, ts).astype(int)
    #blink_index_orig_time = vedb_gaze.utils.onoff_to_binary(blink_onoff_orig_time, len(ts))
    # Filter out too-long or too-short blinks
    blinks_out = []
    for on, off, duration in blink_times_resampled_time:
        # Convert to ms
        dur = duration #* 1000
        if (dur > min_full_blink_time) & (dur < max_full_blink_time):
            blinks_out.append((on, off, duration))
    return dict(timestamp=ts_,
                confidence=conf_, 
                timestamp_orig=ts,
                confidence_orig=conf,
                blinks_onoff=blinks_out,
               )


def get_saccade_rate(onoff_times, timestamps, output_fps=1, orig_fps=120, window=10):
    max_time = np.max(timestamps)
    min_time = np.min(timestamps)
    blink_starts = np.asarray(onoff_times)[:,0]
    half_window = (window / 2) 
    out = []
    for t in np.arange(min_time, max_time, 1 / output_fps):
        if t < min_time + half_window:
            out.append(np.nan)
            continue
        elif t >= max_time - half_window:
            out.append(np.nan)
            continue
        blinks = (blink_starts > (t - half_window)) & (blink_starts < (t+half_window))
        blink_rate = np.sum(blinks) * (60/window)
        out.append(blink_rate)
    return np.asarray(out)

    


# def load_gaze(session, pipeline_tag, 
#               eye='best',
#               resample_to=120,):
    
#     gaze = e.gaze.data['norm_pos']
#     conf = e.gaze.data['confidence']
#     gtime = e.gaze.data['timestamp'] - e.session.start_time
#     # Keep high confidence for now
#     keep = conf > 0.7
#     keep.mean()    


def find_saccades(gaze,
                  session=None,
                  aspect_ratio = 4/3,
                  max_size_deg=125, # Replace me with nonlinear warp of gaze
                  saccade_max_velocity=600,
                  saccade_min_velocity=75, 
                  blink_confidence_threshold=0.85):
    """find saccades and blinks

    Parameters
    ----------
    gaze : dict
        gaze dict with 'norm_pos' (0-1 gaze position estimates in normalized
        world camera coordinates),'timestamps', and 'confidence' fields
    session : str, optional
        string identifier for session in which we are operating, by default None
    aspect_ratio : scalar, optional
        aspect ratio of world camera, by default 4/3
    max_size_deg : scalar, optional
        size in degrees of world camera FOV, by default 125
        for now, assumes this is degrees are linear across world camera image
        (this is a bad assumption and should be revisited at some point to 
        compensate for fisheye world camera lens)
    saccade_min_velocity : scalar, optional
        threshold over which movement is defined as a saccade, by default 75
        TO DO: make me adaptive as in ReModNav
    saccade_max_velocity : scalar, optional
        threshold over which velocity estimate is assumed to be divergent
        (i.e. probably part of a blink)
    blink_confidence_threshold : float, optional
        threshold for gaze confidence used to define blinks, by default 0.85

    Returns
    -------
    Binary arrays labeling saccades and blinks

    See Also
    --------
    find_saccades_remodnav : adaptive-threshold saccade detection (REMoDNaV),
        recommended over this fixed-threshold version
    """
    # Clearer variables
    t = np.array(gaze['timestamp'])
    confidence = np.array(gaze['confidence'])

    eye_velocity = compute_eye_velocity(gaze, max_size_deg=max_size_deg, aspect_ratio=aspect_ratio)
    # Find blinks
    blink_binary = confidence < blink_confidence_threshold
    # Extend blinks with outlying / divergent eye velocities
    velocity_outliers = eye_velocity > saccade_max_velocity
    blink_binary_extended = velocity_outliers | (blink_binary)
    #onoff_blink_extended = onoff_from_binary(blink_binary_extended, return_duration=False)
    # Find saccades
    saccade_binary = eye_velocity > saccade_min_velocity
    #saccade_only_binary = saccade_binary & (~blink_extended)
    
    # For blink rate, if we want that...
    #blink_on = np.zeros_like(blink_extended)
    #blink_on[onoff_blink_extended[:,0]] = 1
    # blink_clips = ClipList.from_binary(blink_binary_extended, t, session=session)
    # change
    # Filter BS 1-frame clips
    # blink_clips.clip_list = [x for x in blink_clips if x.duration > 0]
    # saccade_clips = ClipList.from_binary(saccade_binary, t, session=session)
    # Filter BS 1-frame clips
    # saccade_clips.clip_list = [x for x in saccade_clips if x.duration > 0]
    # return saccade_clips, blink_clips
    return saccade_binary, blink_binary_extended


def find_saccades_remodnav(gaze,
                           fps=120,
                           degrees_horiz=101,
                           degrees_vert=75.75,
                           min_confidence=0.6,
                           max_gap=None,
                           saccade_labels=('SACC', 'ISAC'),
                           # Classifier parameters (REMoDNaV defaults)
                           pursuit_velthresh=2.0,
                           noise_factor=5.0,
                           velthresh_startvelocity=300.0,
                           min_intersaccade_duration=0.04,
                           min_saccade_duration=0.01,
                           max_initial_saccade_freq=2.0,
                           saccade_context_window_length=1.0,
                           max_pso_duration=0.04,
                           min_fixation_duration=0.04,
                           min_pursuit_duration=0.04,
                           lowpass_cutoff_freq=4.0,
                           # Preprocessing parameters (REMoDNaV defaults,
                           # except `savgol_length`)
                           min_blink_duration=0.02,
                           dilate_nan=0.01,
                           median_filter_length=0.05,
                           savgol_length=None,
                           savgol_polyord=2,
                           max_vel=1000.0,
                           ):
    """Find saccades (and fixations, pursuits, and post-saccadic oscillations)
    with the REMoDNaV algorithm

    The classification is done by `vedb_gaze.externals.remodnav`, which is
    derived from the REMoDNaV package (version 1.1.2,
    https://github.com/psychoinformatics-de/remodnav, MIT license) with the
    algorithm unchanged. This function only adapts vedb gaze data to its
    input requirements and its output to vedb_gaze conventions. If you use
    it, please cite:

        Dar, A. H., Wagner, A. S., & Hanke, M. (2021). REMoDNaV: robust
        eye-movement classification for dynamic stimulation. Behavior Research
        Methods, 53(1), 399-414. https://doi.org/10.3758/s13428-020-01428-x

    Steps:
    1. Gaze position (normalized 0-1 world camera coordinates) is converted
       to degrees by scaling by `degrees_horiz` and `degrees_vert`. This
       assumes degrees are linear across the world camera image, which is
       not true for the fisheye world camera lens (same assumption as
       `compute_eye_velocity`).
    2. Samples with confidence below `min_confidence` are set to NaN, which
       REMoDNaV treats as signal loss (e.g. blinks).
    3. Data are linearly resampled to a uniform rate of `fps`, which
       REMoDNaV requires. Resampled points that fall in a gap between
       original samples longer than `max_gap` are set to NaN.
    4. REMoDNaV preprocessing (spike filter, NaN dilation, Savitzky-Golay
       smoothing, velocity computation) and classification are run.

    Note that REMoDNaV was developed for 500-1000 Hz eye trackers; its
    default durations correspond to only a few samples at 120 Hz, and may
    need tuning for vedb data.

    Parameters
    ----------
    gaze : dict
        gaze dict with 'timestamp', 'norm_pos' (0-1 gaze position estimates
        in normalized world camera coordinates), and (optionally)
        'confidence' fields
    fps : scalar, optional
        sampling rate (Hz) to which gaze is resampled before classification,
        by default 120
    degrees_horiz : scalar, optional
        horizontal size of world camera field of view in degrees, by default 101
    degrees_vert : scalar, optional
        vertical size of world camera field of view in degrees, by default 75.75
    min_confidence : scalar or None, optional
        gaze samples below this confidence are treated as missing data, by
        default 0.6. None to keep all samples.
    max_gap : scalar or None, optional
        maximum gap (in seconds) between original samples across which to
        interpolate; resampled points in longer gaps are treated as missing
        data. If None, defaults to 3 / `fps`.
    saccade_labels : tuple, optional
        REMoDNaV event labels to include in `saccades_onoff`, by default
        ('SACC', 'ISAC') (major saccades and saccades found within
        inter-saccade periods). Other labels are 'HPSO', 'LPSO', 'IHPS',
        'ILPS' (high / low velocity post-saccadic oscillations), 'FIXA'
        (fixation), and 'PURS' (pursuit).
    pursuit_velthresh, noise_factor, velthresh_startvelocity,
    min_intersaccade_duration, min_saccade_duration, max_initial_saccade_freq,
    saccade_context_window_length, max_pso_duration, min_fixation_duration,
    min_pursuit_duration, lowpass_cutoff_freq :
        REMoDNaV classifier parameters (velocities in deg/s, durations in
        seconds); see `vedb_gaze.externals.remodnav.EyegazeClassifier` and
        Dar et al. (2021) for descriptions. Defaults match REMoDNaV.
    min_blink_duration, dilate_nan, median_filter_length, savgol_polyord,
    max_vel :
        REMoDNaV preprocessing parameters (durations in seconds); see
        `vedb_gaze.externals.remodnav.EyegazeClassifier.preproc`. Defaults
        match REMoDNaV.
    savgol_length : scalar or None, optional
        Savitzky-Golay filter length in seconds. REMoDNaV's default (0.019 s)
        does not give a valid (odd) window length at 120 Hz, so if None
        (default), the shortest odd window of at least 0.019 s and more than
        `savgol_polyord` samples is used (3 samples at 120 Hz).

    Returns
    -------
    dict with fields:
        timestamp : array
            resampled (uniform) timestamps, on the same clock as
            gaze['timestamp']
        position : array
            (n, 2) preprocessed (filtered) gaze position in degrees from the
            top left of the world camera image
        velocity : array
            gaze velocity in degrees per second
        events : dict of arrays
            all REMoDNaV events, with fields 'label', 'start_time',
            'end_time', 'duration', 'start_x', 'start_y', 'end_x', 'end_y'
            (degrees), 'amp' (degrees), 'peak_vel', 'med_vel', 'avg_vel'
            (deg/s), and 'id'. Times are on the gaze['timestamp'] clock.
        saccades_onoff : array
            (n, 3) array of (onset, offset, duration) for events with labels
            in `saccade_labels`, same format as `blinks_onoff` output of
            blink detection functions.
    """
    t = np.asarray(gaze['timestamp'], dtype=float)
    xy = np.asarray(gaze['norm_pos'], dtype=float) * \
        np.array([degrees_horiz, degrees_vert])
    if (min_confidence is not None) and ('confidence' in gaze):
        xy[np.asarray(gaze['confidence']) < min_confidence] = np.nan

    # Resample to uniform sampling rate; NaNs (low confidence) are not
    # interpolated over, and remain as missing data
    new_time = np.arange(t[0], t[-1], 1 / fps)
    _, xy = resample_data(t, xy, new_time=new_time,
                          method='linear_interpolation', remove_nans=False)
    # Treat resampled points within long gaps in original data as missing
    if max_gap is None:
        max_gap = 3 / fps
    idx = np.clip(np.searchsorted(t, new_time, side='right'), 1, len(t) - 1)
    in_gap = (t[idx] - t[idx - 1]) > max_gap
    xy[in_gap] = np.nan

    if savgol_length is None:
        n = max(int(np.ceil(0.019 * fps)), savgol_polyord + 1)
        if n % 2 == 0:
            n += 1
        # REMoDNaV converts seconds to samples with int(); offset by half a
        # sample to avoid floating point rounding down
        savgol_length = (n + 0.5) / fps

    clf = EyegazeClassifier(
        px2deg=1.0,  # data are already in degrees
        sampling_rate=fps,
        pursuit_velthresh=pursuit_velthresh,
        noise_factor=noise_factor,
        velthresh_startvelocity=velthresh_startvelocity,
        min_intersaccade_duration=min_intersaccade_duration,
        min_saccade_duration=min_saccade_duration,
        max_initial_saccade_freq=max_initial_saccade_freq,
        saccade_context_window_length=saccade_context_window_length,
        max_pso_duration=max_pso_duration,
        min_fixation_duration=min_fixation_duration,
        min_pursuit_duration=min_pursuit_duration,
        lowpass_cutoff_freq=lowpass_cutoff_freq,
    )
    data = np.rec.fromarrays([xy[:, 0].copy(), xy[:, 1].copy()],
                             names=['x', 'y'])
    pp = clf.preproc(data,
                     min_blink_duration=min_blink_duration,
                     dilate_nan=dilate_nan,
                     median_filter_length=median_filter_length,
                     savgol_length=savgol_length,
                     savgol_polyord=savgol_polyord,
                     max_vel=max_vel,
                     )
    events = clf(pp, classify_isp=True, sort_events=True)

    # Convert event times (seconds from first sample) to gaze clock
    for e in events:
        e['start_time'] += new_time[0]
        e['end_time'] += new_time[0]
        e['duration'] = e['end_time'] - e['start_time']
    fields = clf.record_field_names + ['duration']
    events_out = dict((k, np.array([e[k] for e in events])) for k in fields)
    saccades_onoff = np.array([(e['start_time'], e['end_time'], e['duration'])
                               for e in events if e['label'] in saccade_labels])
    saccades_onoff = saccades_onoff.reshape(-1, 3)

    return dict(timestamp=new_time,
                position=np.vstack([pp['x'], pp['y']]).T,
                velocity=pp['vel'],
                events=events_out,
                saccades_onoff=saccades_onoff,
                )


# def plot_at_times(tt, y, time_start, time_end, 
#                   time_units='seconds', ax=None, **kwargs):
#     if ax is None:
#         _, ax = plt.subplots()
#     if time_units in ('seconds', 's'):
#         multiplier = 1
#     elif time_units in ('minutes', 'm'):
#         multiplier = 60
#     st, fin = vedb_store.utils.get_frame_indices(time_start * multiplier, time_end * multiplier, tt)
#     ax.plot(tt[st:fin] / multiplier, y[st:fin], **kwargs)


# Pylids video overlay
# def pylids_label_video(fpath, eye_data, timestamps, st, fin, eye_color=(1, 0,1, 0.2), figsize=(5, 5)):

#     if ax is None:
#         fig, ax = plt.subplots(figsize=figsize)
    
#     ti = (timestamps >= st) & (timestamps <= fin)
#     frame_i, = np.nonzero(ti)
#     st_frame = frame_i[0]
#     fin_frame = frame_i[-1]
#     tmp = eye_data['ellipse'][st_frame]
#     ellipse_data = dict((k, np.array(v) / 400)
#                               for k, v in tmp.items())
#     ev = file_io.load_mp4(fpath, frames=(st_frame, fin_frame))
#     imh = ax.imshow(ev[0])
#     pupil_h = vedb_gaze.visualization.show_ellipse(ellipse_data,
#                                                        center_color=eye_color,
#                                                        facecolor=eye_color +
#                                                        (0.5,),
#                                                        ax=ax)
#     for frame in range(st_frame, fin_frame):
#         # define animation functions?
#         tmp = eye_data['ellipse'][frame]
#         ellipse_data = dict((k, np.array(v) / 400) for k, v in tmp.items()) 
#         pupil_h[0].set_center(ellipse_data_right['center'])
#         pupil_h[0].set_angle(ellipse_data_right['angle'])
#         pupil_h[0].set_height(ellipse_data_right['axes'][1])
#         pupil_h[0].set_width(ellipse_data_right['axes'][0])
#         # Accumulate? Either for hist, or only matched data.
#         pupil_h[1].set_offsets([ellipse_data_right['center']])


def plot_blinks(blinks, 
                blink_buffer=0.25, 
                color_by='duration',
                min_time=0.125,
                max_time=0.3,
                cmap=plt.cm.viridis,
                ax=None,
                alpha=0.3,
                percentiles=(0.1, 99.9),
                ):
    if ax is None:
        fix, ax = plt.subplots()
    #dst_min = np.nanmin(blinks['distance'])
    #dst_max = np.nanmax(blinks['distance'])
    dst_min, dst_max = np.percentile(blinks['distance'], percentiles)
    dst_min = np.maximum(dst_min, 0)
    dst_max = np.minimum(dst_max, 300)
    dst_nrm = Normalize(vmin=dst_min, vmax=dst_max)
    blink_time_orig = blinks['blinks_onoff'].copy()
    blink_time = blinks['blinks_onoff'].copy()
    blink_time[:,0] -= blink_buffer
    blink_time[:,1] += blink_buffer
    #
    # Sort by duration
    duration_idx = np.argsort(blinks['blinks_onoff'][:,2])
    blink_time = blink_time[duration_idx]
    duration = blink_time[:, 2]
    bi = time_to_index(blink_time[:,:2], blinks['timestamp']).astype(int)
    # Sort by closure
    # To come
    # Normalization
    nrm = Normalize(vmin=min_time, vmax=max_time)
    for (on, off), dur in zip(bi, duration):
        dst = dst_nrm(blinks['distance'][on:off])
        tt = blinks['timestamp'][on:off] - blinks['timestamp'][on] - blink_buffer
        ax.plot(tt, dst, color=cmap(nrm(dur)), alpha=alpha)
    ax.vlines(0, 0, 1, ls='--', color='darkgray') 
    _ = ax.set_ylim([0,1])
    _ = ax.set_xlim([-blink_buffer, max_time + blink_buffer])
    plot_utils.open_axes(ax)
    ax.set_ylabel('Eyelid-to-eyelid distance') #\n(proportion of max opening)')
    ax.set_xlabel("Time (s)")
    plot_utils.set_ax_fontsz(ax, lab=11, tk=9, name='Helvetica')        

def detrend_median(data, fps=45, window_seconds=20, impute_mean=(0.5, 0.5)):
    """Perform median detrending on data

    With default settings, removes low-frequency drift in a signal (for fluctuations
    slower than `window_seconds` at the specified `fps`)

    Parameters
    ----------


    """
    med_x = scipy.signal.medfilt(data[:,0], kernel_size=fps * window_seconds + 1)
    med_y = scipy.signal.medfilt(data[:,1], kernel_size=fps * window_seconds + 1)
    out = np.vstack([data[:,0] - med_x,
                     data[:,1] - med_y]).T
    if impute_mean is not None:
        out += np.asarray(impute_mean)
    return out


def get_blink_rate(onoff_times, timestamps, output_fps=1, orig_fps=120, window=10):
    max_time = np.max(timestamps)
    min_time = np.min(timestamps)
    blink_starts = np.asarray(onoff_times)[:,0]
    half_window = (window / 2) 
    out = []
    for t in np.arange(min_time, max_time, 1 / output_fps):
        if t < min_time + half_window:
            out.append(np.nan)
            continue
        elif t >= max_time - half_window:
            out.append(np.nan)
            continue
        blinks = (blink_starts > (t - half_window)) & (blink_starts < (t+half_window))
        blink_rate = np.sum(blinks) * (60/window)
        out.append(blink_rate)
    return np.asarray(out)

def remove_blinks(blinks, *data, buffer=None, replace_with=np.nan, timestamp=None):
    out = []
    onoff = blinks['blinks_onoff']
    if buffer is not None:
        onoff = buffer_onoff(onoff, buffer)
    for d in data:
        if isinstance(d, np.ndarray):
            tmp = d.copy()
        else:
            tmp = np.asarray(d)
        if timestamp is None:
            if len(d) == len(blinks['timestamp']):
                # Resampled data has been provided
                timestamp = blinks['timestamp'].copy()
            elif len(d) == len(blinks['timestamp_orig']):
                timestamp = blinks['timestamp_orig'].copy()
            else:
                raise ValueError("WTF")
        for on, off, duration in onoff:
            ti = (timestamp > on) & (timestamp < off)
            tmp[ti] = replace_with
        out.append(tmp)
    return out
