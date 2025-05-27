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
from sklearn.decomposition import PCA

from .utils import onoff_from_binary
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
except ImportError:
    pass

# Basics

def resample_data(timestamps, data, 
                      fps=120,
                      method='linear_interpolation',
                      remove_nans=True,
                      **kwargs):
    """
    Removes outliers (< max eye image size, std > std_threshold)

    Parameters
    ----------
    timestamps : array
        array of times associated with `data` array
        These are the timestamps you WANT the data to have.
    data : array
        values to be resampled (the 'y' or dependent values
        to the `timestamps`' 'x'). May be 1d or 2d array, 
        first dimension must match `timestamps`
    outlier_thresh : scalar
        threshold for outliers, in stds
    """
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
            print("Hint: this runs much faster with neighbors=7 or some low number")
        interp = scipy.interpolate.RBFInterpolator(t, inpt, **kwargs)
    else:
        raise NotImplementedError(f"Method {method} not available!")
    data_out = interp(new_time)
    return new_time, data_out

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
        if align_eye_pca:
            # rotate eyelid keypoints but maintain original mean, to maintain valid
            # assumptions (we hope) about x and y range
            x_, y_ = (eyelid_pcs.transform(np.vstack([x, y]).T) + eyelid_pcs.mean_).T
        else:
            x_, y_ = x, y
        x_viz_eye, fit_eye_up, fit_eye_lo, corners_x, coefs_up, coefs_lo = \
            pylids.fit_eyelid(x_, y_, c, return_full_eyelid=True)
        if save_fits:
            fits['x'].append(x_viz_eye)
            fits['y_upper_coef'].append(coefs_up)
            fits['y_lower_coef'].append(coefs_lo)
            fits['y_upper_est'].append(fit_eye_up)
            fits['y_lower_est'].append(fit_eye_lo)
        if coarse_to_fine_estimate:
            dst.append(get_eyelid_distance_coarse_to_fine(x_viz_eye, coefs_up, coefs_lo,
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
    negative_velocity_threshold=negative_velocity_threshold,
    positive_velocity_threshold=positive_velocity_threshold,
    ): 
    """
    Note: All parameter times in milliseconds
    """
    # Z-score and scale resulting range to -1 to 1
    # eyelid_velocity = pylids.filter_scale_blinks(dist_eyelid)
    # ... or just compute gradient
    eyelid_velocity = np.gradient(dist_eyelid)
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
                  negative_velocity_threshold=-0.02,
                  positive_velocity_threshold=0.02,
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
                  min_confidence = None,
                  min_full_blink_time = 16,
                  max_full_blink_time = 500,
                  negative_velocity_threshold=None,
                  positive_velocity_threshold=None,
                  idx=None
                 ):
    """
    if min_confidence is None, use 1 std below median
    All closing, opening, blink times in ms
    velocity should be converted to % closure / second

    THIS IS WIP DO NOT USE
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
    blink_onoff_resampled_time = vedb_gaze.utils.onoff_from_binary(blink_index_resampled_time)
    blink_times_resampled_time = [(ts_[st], ts_[fin], dur*1/fps) for st, fin, dur in blink_onoff_resampled_time]
    #tt = np.asarray([(ts_[st], ts_[fin]) for st, fin, dur in blink_onoff_resampled_time])
    #blink_onoff_orig_time = vedb_gaze.utils.time_to_index(tt, ts).astype(int)
    #blink_index_orig_time = vedb_gaze.utils.onoff_to_binary(blink_onoff_orig_time, len(ts))
    # Filter out too-long or too-short blinks
    blinks_out = []
    for on, off, duration in blink_times_resampled_time:
        # Convert to ms
        dur = duration * 1000
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

    


def find_fixation_breaks(data):
    pass

# def load_gaze(session, pipeline_tag, 
#               eye='best',
#               resample_to=120,):
    
#     gaze = e.gaze.data['norm_pos']
#     conf = e.gaze.data['confidence']
#     gtime = e.gaze.data['timestamp'] - e.session.start_time
#     # Keep high confidence for now
#     keep = conf > 0.7
#     keep.mean()    


# def detect_saccades_eyevel(data, 
#     fps=120, 
#     min_full_saccade_time = 16,
#     max_full_saccade_time = 500,
#     velocity_type='xy',
#     velocity_threshold=.2, # Completely made up, placeholder
#     ): 
#     """
#     gaze data must be provided in degrees 
#     Note: All parameter times in milliseconds
#     velocity in degrees per second
#     """
    
    
#     # eye velocity from gradient of position
#     vx = np.gradient(gaze[:,0]) / np.diff(gtime)
#     vy = np.gradient(gaze[:,1]) / np.diff(gtime)
#     if velocity_type == 'x':
#         eyelid_velocity = vx
#     elif velocity_type == 'y':
#         eyelid_velocity = vx
#     elif velocity_type == 'xy':
#         eyelid_velocity = np.linalg.norm(np.vstack([vx, vy]), axis=0)
#     pred_saccade_labels = np.zeros((len(eyelid_velocity),))
#     blink_label = 1
#     i = 0
#     done = False
#     while i < (len(eyelid_velocity)-1):
#         if eyelid_velocity[i] <= negative_velocity_threshold:
#             saccade_start = i
#             while eyelid_velocity[i] <= negative_velocity_threshold:
#                 saccade_end = i
#                 i += 1
#                 if i > (len(eyelid_velocity)-1):
#                     done = True
#                     break
#             if (saccade_end-saccade_start) * (1000/fps) < max_eye_closing_time and \
#                (saccade_end-saccade_start) * (1000/fps) > min_eye_closing_time and \
#                 not done:
#                 saccade_mid = i
#                 while eyelid_velocity[i] > negative_velocity_threshold and \
#                       eyelid_velocity[i] < positive_velocity_threshold:
#                     saccade_end = i
#                     i += 1
#                     if i > (len(eyelid_velocity)-1):
#                         done = True
#                         break

#                 if (saccade_mid-saccade_end) * (1000/fps) < max_full_closure_time and \
#                     not done:
#                     saccade_last = i
#                     while eyelid_velocity[i] > positive_velocity_threshold:
#                         saccade_end = i
#                         i += 1
#                         if i > (len(eyelid_velocity)-1):
#                             done = True
#                             break

#                     # and min(eyelid_velocity[saccade_start:saccade_end])< 0.53:
#                     if (saccade_end-saccade_last) * (1000/fps) > min_eye_opening_time and \
#                         (saccade_end-saccade_start) * (1000/fps) < max_full_saccade_time and \
#                         (saccade_end-saccade_start) * (1000/fps) > min_full_saccade_time and \
#                         not done:
#                         pred_saccade_labels[saccade_start:saccade_end] = saccade_label
#         i += 1

#     return pred_saccade_labels




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
    ClipLists for saccades, blinks
    """    """"""
    # Clearer variables
    t = np.array(gaze['timestamp'])
    confidence = np.array(gaze['confidence'])

    eye_velocity = compute_eye_velocity(gaze, max_size_deg=max_size_deg, aspect_ratio=aspect_ratio)
    # Find blinks
    blink_binary = confidence < blink_confidence_threshold
    onoff_blink = vedb_gaze.utils.onoff_from_binary(blink_binary, return_duration=False)
    # Extend blinks with outlying / divergent eye velocities
    velocity_outliers = eye_velocity > saccade_max_velocity
    blink_binary_extended = velocity_outliers | (blink_binary)
    #onoff_blink_extended = vedb_gaze.utils.onoff_from_binary(blink_binary_extended, return_duration=False)
    # Find saccades
    saccade_binary = eye_velocity > saccade_min_velocity
    #saccade_only_binary = saccade_binary & (~blink_extended)
    
    # For blink rate, if we want that...
    #blink_on = np.zeros_like(blink_extended)
    #blink_on[onoff_blink_extended[:,0]] = 1
    blink_clips = ClipList.from_binary(blink_binary_extended, t, session=session)
    # Filter BS 1-frame clips
    blink_clips.clip_list = [x for x in blink_clips if x.duration > 0]
    saccade_clips = ClipList.from_binary(saccade_binary, t, session=session)
    # Filter BS 1-frame clips
    saccade_clips.clip_list = [x for x in saccade_clips if x.duration > 0]
    return saccade_clips, blink_clips


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
    bi = vedb_gaze.utils.time_to_index(blink_time[:,:2], blinks['timestamp']).astype(int)
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
