# vedb_gaze_labeling

# Blink detection WIP
"""Eye-movement and blink labeling for VEDB gaze and pupil data.

This module operates on outputs of the gaze pipeline (pupil detection and
gaze mapping) to label events in time:

- Blinks: from eyelid distance computed from `pylids` (DeepLabCut) eyelid
  keypoints (`get_eyelid_distance`, `detect_blinks`), or from drops in
  pupil detection confidence (`detect_blinks_confidence`).
- Saccades: by a fixed eye-velocity threshold (`find_saccades`) or with the
  adaptive REMoDNaV algorithm (`find_saccades_remodnav`).
- Helpers to manipulate event (onset, offset, duration) arrays, compute
  event rates, remove blink periods from data, and plot blinks.

Several functions here are work in progress (see notes in docstrings).
"""
import plot_utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import tqdm.notebook
import warnings
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
    onoff : array-like
        array or list with tuples, each row (or item of list) should be
        (onset, offset, duration)
    buffer_time : scalar, list, or tuple
        if scalar, same `buffer_time` is added before onsets & after offsets.
        if tuple or list, should be 2 long, with separate values
        for pre-onset and post-offset buffers. (A numpy array is NOT
        unpacked; it is treated like a scalar.) Same units as `onoff`.

    Returns
    -------
    array
        (n, 3) array of (onset - pre, offset + post, duration + pre + post)
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
    timestamps : np.ndarray
        timestamps associated with each data point (timestamps associated
        with outlying data points are also removed)
    data : np.ndarray
        1D data in which to search for outliers
    z_max : scalar or None, optional
        maximum z score (computed over all of `data`) to keep, by default 4.
        If `z_min` is None, points with abs(z) > `z_max` are removed
        (symmetric threshold). None to skip z-score thresholding.
    z_min : scalar or None, optional
        minimum z score to keep, by default None (use -`z_max`). Set to
        -np.inf to only remove high outliers.
    absolute_min : scalar, optional
        absolute minimum threshold below which points will be considered
        outliers; None for skip this, by default None
    absolute_max : scalar, optional
        absolute maximum threshold above which points will be considered
        outliers; None for skip this, by default None

    Returns
    -------
    timestamps, data : np.ndarray
        inputs with outlying points removed
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

    Gaze position is converted to approximate degrees by scaling normalized
    (0-1) world camera coordinates by the field of view, assuming degrees
    are linear across the world camera image (not true for the fisheye
    lens). Velocity is the norm of the spatial gradient divided by the
    temporal gradient (`np.gradient`), so samples need not be uniform.

    Parameters
    ----------
    gaze : dict
        gaze arraydict with 'timestamp' (seconds), 'norm_pos' ((n, 2),
        normalized 0-1 world camera coordinates), and 'confidence' fields
        ('confidence' is required but currently unused)
    max_size_deg : scalar, optional
        horizontal size of world camera field of view in degrees, by
        default 125
    aspect_ratio : scalar, optional
        world camera aspect ratio (width / height); vertical size in degrees
        is `max_size_deg` / `aspect_ratio`, by default 4/3

    Returns
    -------
    np.ndarray
        (n,) eye velocity in (approximate) degrees per second

    Notes
    -----
    Open questions from development: should blinks be filtered (or set to
    NaN) first? Should velocity be weighted by confidence? Currently neither
    is done.
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

    Returns
    -------
    sklearn.decomposition.PCA object
        the same object, with the sign of the first two components flipped
        (in place) if needed so that the largest-magnitude element of each
        is positive
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

    Upper and lower eyelid keypoints (split by `pylids.utils.parse_keypoints`)
    from `n_points` evenly spaced frames are pooled, and a 2D PCA is fit to
    their (x, y) coordinates. The first PC approximates the eye's long
    (corner-to-corner) axis. Requires `pylids`.

    Parameters
    ----------
    pupil_data : dict
        pupil detection arraydict with 'dlc_kpts_x' and 'dlc_kpts_y' fields
        (per-frame arrays of keypoint coordinates in eye image space)
    n_points : int, optional
        number of frames (evenly spaced across the data) to use, by default
        1000
    assure_positive_pcs : bool, optional
        whether to flip PC signs with `assure_positive`, by default True

    Returns
    -------
    sklearn.decomposition.PCA object
        PCA fit to eyelid keypoint coordinates
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
    """Searches for maximum distance b/w eyelids (for one frame) and uses
    that as the distance b/w eyelids.

    First evaluates the vertical distance between upper and lower eyelid
    polynomials at `eyelid_resolution_coarse` points spanning `x_new`, then
    re-samples `eyelid_resolution_fine` points in the interval between the
    coarse sample before the maximum and the maximum.

    Parameters
    ----------
    x_new : array-like
        x coordinates of the fitted eyelid; only the first and last values
        (the x range) are used
    coefs_up : array-like
        polynomial coefficients for upper eyelid (lowest order first, as
        used by `numpy.polynomial.polynomial.polyval`)
    coefs_lo : array-like
        polynomial coefficients for lower eyelid (same format)
    eyelid_resolution_coarse : int, optional
        number of points for coarse search, by default 100
    eyelid_resolution_fine : int, optional
        number of points for fine search, by default 100

    Returns
    -------
    np.ndarray
        1-element array with the maximum eyelid distance for this frame
        (same units as the eyelid coordinates)

    Notes
    -----
    The fine search interval is [x[argmax - 1], x[argmax]], i.e. only on one
    side of the coarse maximum; if the coarse maximum is at index 0, index
    -1 wraps around to the last coarse sample.
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
    """Compute distance between upper and lower eyelids for each frame

    For each frame, eyelid keypoints are (optionally) rotated into the
    principal axes of the eye, upper and lower eyelid curves are fit with
    `pylids.fit_eyelid`, and the maximum vertical distance between the
    curves is taken as the eyelid distance. Requires `pylids`.

    Parameters
    ----------
    pupil_data : dict
        pupil detection arraydict with 'dlc_kpts_x', 'dlc_kpts_y', and
        'dlc_confidence' fields (per-frame keypoint arrays from pylids).
        Frames with 48 keypoints use the 'eyelid_pupil' model type and
        frames with 32 keypoints use the 'eyelid' model type.
    eyelid_resolution_coarse : int, optional
        number of points for coarse search, by default 100; only used if
        `coarse_to_fine_estimate` is True
    eyelid_resolution_fine : int, optional
        number of points for fine search, by default 100; only used if
        `coarse_to_fine_estimate` is True
    align_eye_pca : bool, optional
        whether to rotate keypoints into the eye's principal axes (see
        `get_major_minor_axes_pca`) before fitting, so that distances are
        measured perpendicular to the long axis of the eye, by default True
    n_points_pca : int, optional
        number of frames used to fit the PCA, by default 1000
    idx : array-like or None, optional
        indices of frames to process, by default None (all frames)
    assure_positive_pcs : bool, optional
        passed to `get_major_minor_axes_pca`, by default True
    save_fits : bool, optional
        whether to accumulate eyelid fits, by default False. NOTE: fits
        are collected but currently not returned.
    coarse_to_fine_estimate : bool, optional
        if True, use `get_eyelid_distance_coarse_to_fine` on the fitted
        polynomials; if False (default, much faster), use the max absolute
        difference between the fitted upper and lower eyelid curves
    progress_bar : callable, optional
        progress bar wrapper for the frame loop, by default
        tqdm.notebook.tqdm

    Returns
    -------
    np.ndarray
        eyelid distance for each frame in `idx`, in eye image keypoint units
        (pixels). Shape (n,), or (n, 1) if `coarse_to_fine_estimate` is True.
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
# NOTE: the module-level thresholds above are not used as defaults by the
# functions below, which default to -2.4 / +2.4 (fraction of max eye
# opening per second).

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
    """Label blinks in a uniformly sampled eyelid distance signal by eyelid velocity

    A blink is a run of samples with eyelid velocity <=
    `negative_velocity_threshold` (closing), followed by samples with
    velocity between the two thresholds (closed), followed by samples with
    velocity >= `positive_velocity_threshold` (opening), with each phase
    and the whole blink satisfying the duration limits below.

    Parameters
    ----------
    dist_eyelid : np.ndarray
        1D eyelid distance, uniformly sampled at `fps`. If its max is > 1,
        it is divided by its max (so it is a fraction of max opening).
    fps : scalar, optional
        sampling rate of `dist_eyelid` in Hz, by default 120
    min_eye_closing_time, max_eye_closing_time : scalar, optional
        exclusive limits on closing phase duration in ms, by default 10, 250
    max_full_closure_time : scalar, optional
        intended maximum duration (ms) of the closed phase, by default 17.
        As implemented this check never rejects a blink (see Notes).
    min_eye_opening_time : scalar, optional
        minimum duration (ms) of the opening phase, by default 30
    min_full_blink_time, max_full_blink_time : scalar, optional
        exclusive limits on whole blink duration in ms, by default 16, 500
    negative_velocity_threshold : scalar, optional
        eyelid velocity (fraction of max opening per second) at or below
        which the eye is considered closing, by default -2.4
    positive_velocity_threshold : scalar, optional
        eyelid velocity (fraction of max opening per second) above which
        the eye is considered opening, by default 2.4

    Returns
    -------
    np.ndarray
        (n,) float array, 1 for samples labeled as blinks, 0 otherwise

    Notes
    -----
    Durations are computed as (number of samples) * 1000 / `fps`. The
    closed-phase check computes (blink_mid - blink_end), which is never
    positive, so `max_full_closure_time` has no effect.
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

def compute_eyelid_distance(pupil_data,
                            fps=120,
                            absolute_max_distance=400,
                            outlier_z_max=4,
                            idx=None,
                            ):
    """Compute eyelid-to-eyelid distance over time, cleaned and resampled

    This is the eyelid distance used by `detect_blinks`: the distance
    between upper and lower eyelid fits (`get_eyelid_distance`), with
    outliers removed (`remove_outliers`) and then resampled to a uniform
    rate with slight smoothing (thin-plate-spline RBF interpolation,
    `utils.resample_data`).

    Parameters
    ----------
    pupil_data : dict
        pylids pupil / eyelid detections with 'timestamp' and eyelid
        keypoints ('dlc_kpts_x', 'dlc_kpts_y', 'dlc_confidence'), as
        produced by pupil detection with a pylids config that estimates
        eyelids (``estimate_eyelids: true``)
    fps : scalar, optional
        sampling rate (Hz) of the resampled output, by default 120
    absolute_max_distance : scalar, optional
        distances above this (pixels) are removed as outliers before
        resampling, by default 400
    outlier_z_max : scalar, optional
        distances with z-score above this are removed as outliers before
        resampling, by default 4
    idx : array-like or None, optional
        frame indices to use, by default None (all frames)

    Returns
    -------
    dict with fields:
        timestamp : array
            (m,) resampled (uniform) timestamps
        distance : array
            (m,) resampled eyelid distance in pixels (not normalized)
        timestamp_orig : array
            original timestamps (selected by `idx`)
        distance_orig : array
            eyelid distance at original timestamps, before outlier removal

    Raises
    ------
    ImportError
        if pylids is not installed (needed to fit eyelids to keypoints)
    KeyError
        if `pupil_data` lacks timestamps or eyelid keypoints
    """
    if not has_pylids:
        raise ImportError("Computing eyelid distance requires the `pylids` package "
                          "(to fit eyelids to keypoints), which is not installed.")
    required = ('timestamp', 'dlc_kpts_x', 'dlc_kpts_y', 'dlc_confidence')
    missing = [k for k in required if k not in pupil_data]
    if len(missing) > 0:
        raise KeyError(f"Eyelid data is missing field(s) {missing} (it has fields "
                       f"{sorted(pupil_data.keys())}). Eyelid distance needs pylids eyelid "
                       "keypoints ('dlc_kpts_x', 'dlc_kpts_y', 'dlc_confidence') and "
                       "'timestamp', i.e. the output of pupil detection with a pylids "
                       "config that estimates eyelids (`estimate_eyelids: true`).")
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
    return dict(timestamp=ts_.flatten(),
                distance=dst_.flatten(),
                timestamp_orig=ts,
                distance_orig=dst,
                )


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
    """Detect blinks from eyelid distance (pylids eyelid keypoints)

    Steps: (1) compute eyelid distance per frame with `get_eyelid_distance`
    (default settings); (2) remove outliers (distance outside
    [0, `absolute_max_distance`] or z score > `outlier_z_max`); (3) resample
    to a uniform `fps` with a smoothing thin-plate spline (smoothing=0.001,
    neighbors=7); (4) express distance as a fraction of `max_eye_opening`;
    (5) label blinks by eyelid velocity with `_detect_blinks_eyevel`.
    Requires `pylids`.

    Parameters
    ----------
    pupil_data : dict
        pupil detection arraydict with 'timestamp', 'dlc_kpts_x',
        'dlc_kpts_y', and 'dlc_confidence' fields
    fps : scalar, optional
        sampling rate (Hz) to which eyelid distance is resampled, by
        default 120
    min_eye_closing_time, max_eye_closing_time, max_full_closure_time,
    min_eye_opening_time, min_full_blink_time, max_full_blink_time : scalar, optional
        duration limits in milliseconds; see `_detect_blinks_eyevel`
    negative_velocity_threshold, positive_velocity_threshold : scalar, optional
        eyelid velocity thresholds in fraction of max eye opening per
        second, by default -2.4 and 2.4
    absolute_max_distance : scalar, optional
        eyelid distances (pixels) above this are removed as outliers, by
        default 400
    outlier_z_max : scalar, optional
        eyelid distances with z score above this are removed as outliers
        (low z scores are not removed), by default 4
    max_eye_opening : scalar or None, optional
        eyelid distance (pixels) treated as fully open, by default None
        (max of the resampled distance)
    idx : array-like or None, optional
        indices of frames in `pupil_data` to use, by default None (all)

    Returns
    -------
    dict with fields:
        timestamp : array
            (m,) resampled (uniform) timestamps
        distance : array
            (m,) resampled eyelid distance in pixels (not normalized)
        timestamp_orig : array
            original timestamps (selected by `idx`)
        distance_orig : array
            eyelid distance at original timestamps, before outlier removal
        blinks_onoff : array
            (n, 3) array of (onset, offset, duration), one row per blink
            (same format as `find_saccades_remodnav` 'saccades_onoff'); onset and offset are
            times on the `timestamp` clock (offset is the first resampled
            time after the blink), duration is in seconds
    """
    eyelid_distance = compute_eyelid_distance(pupil_data,
                                              fps=fps,
                                              absolute_max_distance=absolute_max_distance,
                                              outlier_z_max=outlier_z_max,
                                              idx=idx)
    ts_, dst_ = eyelid_distance['timestamp'], eyelid_distance['distance']
    ts, dst = eyelid_distance['timestamp_orig'], eyelid_distance['distance_orig']
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
                # (n, 3) array of (onset, offset, duration), like `saccades_onoff`
                blinks_onoff=np.asarray(blink_times_resampled_time, dtype=float).reshape(-1, 3),
               )


def detect_blinks_confidence(pupil_data,
                  fps=120,
                  min_confidence = 0.7,
                  min_full_blink_time = 16 / 1000,
                  max_full_blink_time = 500 / 1000,
                  idx=None
                 ):
    """Detect blinks as periods of low pupil detection confidence

    Confidence is resampled to a uniform `fps` with a smoothing thin-plate
    spline (smoothing=0.001, neighbors=7); runs of samples with confidence
    below `min_confidence` are blink candidates, and candidates with
    durations outside (`min_full_blink_time`, `max_full_blink_time`) are
    discarded.

    Parameters
    ----------
    pupil_data : dict
        pupil detection arraydict with 'timestamp' and 'confidence' fields
    fps : scalar, optional
        sampling rate (Hz) to which confidence is resampled, by default 120
    min_confidence : scalar or None, optional
        confidence threshold below which samples are labeled blinks, by
        default 0.7. If None, use 1 std below the median confidence.
    min_full_blink_time : scalar, optional
        minimum blink duration in SECONDS (exclusive), by default 0.016
    max_full_blink_time : scalar, optional
        maximum blink duration in SECONDS (exclusive), by default 0.5
    idx : array-like or None, optional
        indices of samples in `pupil_data` to use, by default None (all)

    Returns
    -------
    dict with fields:
        timestamp : array
            (m,) resampled (uniform) timestamps
        confidence : array
            (m,) resampled confidence
        timestamp_orig : array
            original timestamps (selected by `idx`)
        confidence_orig : array
            original confidence (selected by `idx`)
        blinks_onoff : array
            (n, 3) array of (onset, offset, duration), one row per blink
            (same format as `find_saccades_remodnav` 'saccades_onoff'); onset and offset are
            times on the `timestamp` clock (offset is the first resampled
            time after the blink), duration is in seconds
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
    # (n, 3) array of (onset, offset, duration), like `saccades_onoff`
    blinks_out = np.asarray(blinks_out, dtype=float).reshape(-1, 3)
    return dict(timestamp=ts_,
                confidence=conf_, 
                timestamp_orig=ts,
                confidence_orig=conf,
                blinks_onoff=blinks_out,
               )


def get_saccade_rate(onoff_times, timestamps, output_fps=1, orig_fps=120, window=10):
    """Compute event (e.g. saccade) rate in a sliding window

    Counts event onsets strictly within +/- `window` / 2 of each output
    time. Identical in implementation to `get_blink_rate` (variable names
    refer to blinks).

    Parameters
    ----------
    onoff_times : array-like
        (n, 2 or 3) events, onset times in first column, on the same clock
        as `timestamps` (seconds)
    timestamps : array-like
        timestamps (seconds) spanning the period over which to compute the
        rate; only the min and max are used
    output_fps : scalar, optional
        sampling rate (Hz) of the output rate time series, by default 1
    orig_fps : scalar, optional
        unused, by default 120
    window : scalar, optional
        window length in seconds, by default 10

    Returns
    -------
    np.ndarray
        events per minute at times np.arange(min(timestamps),
        max(timestamps), 1 / `output_fps`) (times are not returned);
        NaN within `window` / 2 of either end
    """
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
        world camera coordinates),'timestamp', and 'confidence' fields
    session : str, optional
        string identifier for session in which we are operating, by default
        None (currently unused)
    aspect_ratio : scalar, optional
        aspect ratio of world camera, by default 4/3
    max_size_deg : scalar, optional
        size in degrees of world camera FOV, by default 125
        for now, assumes this is degrees are linear across world camera image
        (this is a bad assumption and should be revisited at some point to 
        compensate for fisheye world camera lens)
    saccade_min_velocity : scalar, optional
        threshold (deg/s) over which movement is defined as a saccade, by
        default 75
        TO DO: make me adaptive as in ReModNav
    saccade_max_velocity : scalar, optional
        threshold (deg/s) over which velocity estimate is assumed to be
        divergent (i.e. probably part of a blink), by default 600
    blink_confidence_threshold : float, optional
        threshold for gaze confidence used to define blinks, by default 0.85

    Returns
    -------
    saccade_binary : np.ndarray
        (n,) boolean, True where eye velocity > `saccade_min_velocity`
        (NOT excluding blinks / divergent velocities)
    blink_binary_extended : np.ndarray
        (n,) boolean, True where confidence < `blink_confidence_threshold`
        or eye velocity > `saccade_max_velocity`

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


def plot_blink_eyelid_distance(blinks,
                               eyelids=None,
                               blink_buffer=0.25,
                               color_by='duration',
                               min_time=0.125,
                               max_time=0.3,
                               cmap=plt.cm.viridis,
                               ax=None,
                               alpha=0.3,
                               percentiles=(0.1, 99.9),
                               eyelid_distance_kw=None,
                               ):
    """Plot eyelid distance traces for all blinks, aligned to blink onset

    Each blink (plus `blink_buffer` before and after) is plotted as a trace
    of normalized eyelid-to-eyelid distance vs. time from blink onset,
    colored by blink duration. Traces are drawn in order of increasing
    duration.

    Eyelid distance comes from `eyelids`, if given (computed as in
    `detect_blinks`, with `compute_eyelid_distance`), otherwise from the
    'timestamp' and 'distance' fields of `blinks` (present in the output
    of `detect_blinks`). Use `eyelids` to plot blinks found by a method
    that does not compute eyelid distance, e.g. `detect_blinks_confidence`.

    Parameters
    ----------
    blinks : dict
        blink detections with a 'blinks_onoff' field: (n, 3) array (or
        list) of (onset, offset, duration) rows, with times in seconds on
        the eye-camera clock, e.g. output of `detect_blinks` or
        `detect_blinks_confidence`
    eyelids : dict or None, optional
        pylids eyelid detections for the same eye and session ('timestamp',
        'dlc_kpts_x', 'dlc_kpts_y', 'dlc_confidence'); if given, eyelid
        distance is computed from these instead of taken from `blinks`.
        By default None.
    blink_buffer : scalar, optional
        time (seconds) to plot before and after each blink, by default 0.25
    color_by : str, optional
        what to color traces by; only 'duration' is currently supported,
        by default 'duration'
    min_time, max_time : scalar, optional
        blink durations (seconds) mapped to the ends of `cmap`, by default
        0.125 and 0.3; `max_time` also sets the x axis limit
    cmap : matplotlib colormap, optional
        colormap for blink duration, by default plt.cm.viridis
    ax : matplotlib axis or None, optional
        axis into which to plot, by default None (new figure)
    alpha : scalar, optional
        line transparency, by default 0.3
    percentiles : tuple, optional
        percentiles of eyelid distance (clipped to [0, 300]) mapped to 0 and 1
        for normalizing eyelid distance, by default (0.1, 99.9)
    eyelid_distance_kw : dict or None, optional
        keyword arguments for `compute_eyelid_distance` when `eyelids` is
        given (e.g. ``dict(fps=120)``), by default None (its defaults,
        which match those of `detect_blinks`)

    Returns
    -------
    ax : matplotlib axis
        axis with the plot

    Raises
    ------
    KeyError
        if `blinks` has no 'blinks_onoff', or if `eyelids` is None and
        `blinks` has no eyelid distance ('timestamp' and 'distance')
    ValueError
        if inputs are malformed (see messages) or blinks fall outside the
        time range of the eyelid data
    """
    if color_by != 'duration':
        raise ValueError(f"color_by={color_by!r} is not supported; only 'duration' is.")
    if 'blinks_onoff' not in blinks:
        raise KeyError("`blinks` has no 'blinks_onoff' field; expected the output of "
                       "`detect_blinks` or `detect_blinks_confidence`.")
    onoff = np.asarray(blinks['blinks_onoff'], dtype=float)
    if onoff.size == 0:
        onoff = onoff.reshape(0, 3)
    if (onoff.ndim != 2) or (onoff.shape[1] != 3):
        raise ValueError("blinks['blinks_onoff'] must be (n, 3) rows of (onset, offset, "
                         f"duration); got an array of shape {onoff.shape}.")
    # Eyelid distance over time
    if eyelids is not None:
        if eyelid_distance_kw is None:
            eyelid_distance_kw = {}
        eyelid_distance = compute_eyelid_distance(eyelids, **eyelid_distance_kw)
        timestamp, distance = eyelid_distance['timestamp'], eyelid_distance['distance']
    else:
        if ('distance' not in blinks) or ('timestamp' not in blinks):
            raise KeyError("`blinks` has no eyelid distance ('timestamp' and 'distance' "
                           "fields; `detect_blinks` output has them, `detect_blinks_confidence` "
                           "output does not). Pass the pylids eyelid detections for this eye "
                           "as `eyelids=` to compute eyelid distance.")
        timestamp = np.asarray(blinks['timestamp'])
        distance = np.asarray(blinks['distance'])
    if len(timestamp) != len(distance):
        raise ValueError(f"Eyelid distance has {len(distance)} values but {len(timestamp)} "
                         "timestamps; they must match.")
    if len(onoff) > 0:
        t_first, t_last = onoff[:, 0].min(), onoff[:, 1].max()
        if (t_first < timestamp[0]) or (t_last > timestamp[-1]):
            raise ValueError(f"Blinks span {t_first:.2f}-{t_last:.2f} s, outside the eyelid "
                             f"data ({timestamp[0]:.2f}-{timestamp[-1]:.2f} s). Are `blinks` "
                             "and the eyelid data from the same eye and session, on the same clock?")
    if ax is None:
        _, ax = plt.subplots()
    if len(onoff) == 0:
        warnings.warn("No blinks to plot (blinks['blinks_onoff'] is empty).")
    else:
        dst_min, dst_max = np.percentile(distance, percentiles)
        dst_min = np.maximum(dst_min, 0)
        dst_max = np.minimum(dst_max, 300)
        dst_nrm = Normalize(vmin=dst_min, vmax=dst_max)
        blink_time = onoff.copy()
        blink_time[:,0] -= blink_buffer
        blink_time[:,1] += blink_buffer
        # Sort by duration
        duration_idx = np.argsort(onoff[:,2])
        blink_time = blink_time[duration_idx]
        duration = blink_time[:, 2]
        bi = time_to_index(blink_time[:,:2], timestamp).astype(int)
        # Normalization
        nrm = Normalize(vmin=min_time, vmax=max_time)
        for (on, off), dur in zip(bi, duration):
            dst = dst_nrm(distance[on:off])
            tt = timestamp[on:off] - timestamp[on] - blink_buffer
            ax.plot(tt, dst, color=cmap(nrm(dur)), alpha=alpha)
    ax.vlines(0, 0, 1, ls='--', color='darkgray') 
    _ = ax.set_ylim([0,1])
    _ = ax.set_xlim([-blink_buffer, max_time + blink_buffer])
    plot_utils.open_axes(ax)
    ax.set_ylabel('Eyelid-to-eyelid distance') #\n(proportion of max opening)')
    ax.set_xlabel("Time (s)")
    plot_utils.set_ax_fontsz(ax, lab=11, tk=9, name='Helvetica')        
    return ax

def detrend_median(data, fps=45, window_seconds=20, impute_mean=(0.5, 0.5)):
    """Perform median detrending on data

    With default settings, removes low-frequency drift in a signal (for fluctuations
    slower than `window_seconds` at the specified `fps`)

    Parameters
    ----------
    data : np.ndarray
        (n, 2) data (e.g. normalized x, y positions), assumed uniformly
        sampled at `fps`
    fps : int, optional
        sampling rate of `data` in Hz, by default 45
    window_seconds : int, optional
        length of median filter window in seconds, by default 20. The
        kernel size is `fps` * `window_seconds` + 1 samples, which must be
        an odd integer (`scipy.signal.medfilt` requirement).
    impute_mean : tuple or None, optional
        value added back to each column after detrending, by default
        (0.5, 0.5) (center of normalized coordinates). None to leave the
        output centered on zero.

    Returns
    -------
    np.ndarray
        (n, 2) detrended data
    """
    med_x = scipy.signal.medfilt(data[:,0], kernel_size=fps * window_seconds + 1)
    med_y = scipy.signal.medfilt(data[:,1], kernel_size=fps * window_seconds + 1)
    out = np.vstack([data[:,0] - med_x,
                     data[:,1] - med_y]).T
    if impute_mean is not None:
        out += np.asarray(impute_mean)
    return out


def get_blink_rate(onoff_times, timestamps, output_fps=1, orig_fps=120, window=10):
    """Compute blink rate in a sliding window

    Counts blink onsets strictly within +/- `window` / 2 of each output
    time.

    Parameters
    ----------
    onoff_times : array-like
        (n, 2 or 3) blinks, onset times in first column, on the same clock
        as `timestamps` (seconds); e.g. 'blinks_onoff' from `detect_blinks`
    timestamps : array-like
        timestamps (seconds) spanning the period over which to compute the
        rate; only the min and max are used
    output_fps : scalar, optional
        sampling rate (Hz) of the output rate time series, by default 1
    orig_fps : scalar, optional
        unused, by default 120
    window : scalar, optional
        window length in seconds, by default 10

    Returns
    -------
    np.ndarray
        blinks per minute at times np.arange(min(timestamps),
        max(timestamps), 1 / `output_fps`) (times are not returned);
        NaN within `window` / 2 of either end
    """
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
    """Replace data during blinks with a fill value (NaN by default)

    Parameters
    ----------
    blinks : dict
        output of `detect_blinks` or `detect_blinks_confidence`, with
        'blinks_onoff', 'timestamp', and 'timestamp_orig' fields
    *data : array-like
        one or more arrays to clean; first dimension must be time
    buffer : scalar, list, tuple, or None, optional
        time (seconds) added before and after each blink (see
        `buffer_onoff`), by default None (no buffer)
    replace_with : scalar, optional
        value written into blink periods, by default np.nan (so data must
        have a float dtype)
    timestamp : array-like or None, optional
        timestamps for `data`. If None (default), inferred from the length
        of the first data array: blinks['timestamp'] (resampled) or
        blinks['timestamp_orig']. The inferred timestamps are then reused
        for all subsequent data arrays.

    Returns
    -------
    list of np.ndarray
        copies of each `data` array, with samples strictly between blink
        onset and offset set to `replace_with`
    """
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
