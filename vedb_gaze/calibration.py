# Calibrate alternative for vedb

import numpy as np
import matplotlib.pyplot as plt
import pathlib
try:
    import thinplate as tps
except:
    print(("`thinplate` library not found.\n"
           "Please download and install py-thinplate-spline\n"
           "( http://github.com/cheind/py-thin-plate-spline )\n"
           "if you wish to use thin plate splines for calibration.\b"
           ))
import os

from .utils import (
    filter_arraydict,
    match_time_points,
    arraydict_to_dictlist, 
    dictlist_to_arraydict,
    stack_arraydicts,
    read_yaml,
    get_function,
)
from .visualization import colormap_2d
from .marker_parsing import marker_cluster_stat

from .externals.data_processing import (
    _filter_pupil_list_by_confidence,
    _extract_2d_data_monocular,
    _extract_2d_data_binocular,
    _match_data,
)
from .externals import calibrate_2d

from .externals.gaze_mappers import Binocular_Gaze_Mapper

import logging
logger = logging.getLogger(__name__)


BASE_DIR, _ = os.path.split(__file__)
BASE_DIR = pathlib.Path(BASE_DIR)

def parse_plab_data(pupil_list, ref_list, mode="2d", min_calibration_confidence=0.5):
    """Parse lists of dictionaries geneated by pupil labs code in preparation for calibration

    Returns extracted data for calibration and a flag indicating whether 
    there is binocular data present

    Parameters
    ----------
    pupil_list :

    ref_list :
    """

    pupil_list = _filter_pupil_list_by_confidence(
        pupil_list, min_calibration_confidence
    )

    matched_data = _match_data(pupil_list, ref_list)
    (
        matched_binocular_data,
        matched_monocular_data,
        matched_pupil0_data,
        matched_pupil1_data,
    ) = matched_data

    binocular = None
    extracted_data = None
    if mode == "3d":
        if matched_binocular_data:
            binocular = True
            extracted_data = _extract_3d_data(g_pool, matched_binocular_data)
        elif matched_monocular_data:
            binocular = False
            extracted_data = _extract_3d_data(g_pool, matched_monocular_data)

    elif mode == "2d":
        if matched_binocular_data:
            binocular = True
            cal_pt_cloud_binocular = _extract_2d_data_binocular(
                matched_binocular_data)
            cal_pt_cloud0 = _extract_2d_data_monocular(matched_pupil0_data)
            cal_pt_cloud1 = _extract_2d_data_monocular(matched_pupil1_data)
            extracted_data = (
                cal_pt_cloud_binocular,
                cal_pt_cloud0,
                cal_pt_cloud1,
            )
        elif matched_monocular_data:
            binocular = False
            cal_pt_cloud = _extract_2d_data_monocular(matched_monocular_data)
            extracted_data = (cal_pt_cloud,)

    return binocular, extracted_data

### NEW


def calibrate_2d_polynomial(
    cal_pt_cloud, 
    screen_size=(1, 1), 
    max_stds_for_outliers=None,
    max_absolute_error_threshold=35, 
    recursive_outlier_cut=None,
    binocular=False
):
    """
    we do a simple two pass fitting to a pair of bi-variate polynomials
    return the function to map vector
    """
    # fit once using all avaiable data
    model_n = 7
    if binocular:
        model_n = 13

    cal_pt_cloud = np.array(cal_pt_cloud)

    cx, cy, err_x, err_y = calibrate_2d.fit_poly_surface(cal_pt_cloud, model_n)
    err_dist, err_mean, err_rms = calibrate_2d.fit_error_screen(err_x, err_y, screen_size)
    if max_stds_for_outliers is not None:
        print("Errors:")
        print(err_dist.tolist())
        inliers = err_dist < np.median(err_dist) + max_stds_for_outliers * np.std(err_dist)
        err_ok = err_dist <= max_absolute_error_threshold
    else:
        print("Not computing errors based on std..")
        inliers = err_dist <= max_absolute_error_threshold
        err_ok = inliers
    if np.any(err_ok):  # did not disregard all points..
        # fit again disregarding extreme outliers
        cx, cy, new_err_x, new_err_y = calibrate_2d.fit_poly_surface(
            cal_pt_cloud[inliers], model_n
        )
        map_fn = calibrate_2d.make_map_function(cx, cy, model_n)
        new_err_dist, new_err_mean, new_err_rms = calibrate_2d.fit_error_screen(
            new_err_x, new_err_y, screen_size
        )
        print("Second pass errors:")
        print(new_err_dist)
        logger.info(
            "first iteration. root-mean-square residuals: {}, in pixel".format(
                err_rms
            )
        )
        logger.info(
            "second iteration: ignoring outliers. root-mean-square residuals: {} "
            "in pixel".format(new_err_rms)
        )

        used_num = np.sum(inliers)
        complete_num = cal_pt_cloud.shape[0]
        logger.info(
            "used {} data points out of the full dataset {}: "
            "subset is {:.2f} percent".format(
                used_num, complete_num, 100 * float(used_num) / complete_num
            )
        )

        return (
            map_fn,
            inliers,
            ([p.tolist() for p in cx], [p.tolist() for p in cy], model_n),
        )

    else:  
        # did disregard all points. The data cannot be represented by the model in
        # a meaningful way:
        map_fn = calibrate_2d.make_map_function(cx, cy, model_n)
        logger.error(
            "First iteration. root-mean-square residuals: {} in pixel, "
            "this is bad!".format(err_rms)
        )
        logger.error(
            "The data cannot be represented by the model in a meaningful way."
        )
        return (
            map_fn,
            inliers,
            ([p.tolist() for p in cx], [p.tolist() for p in cy], model_n),
        )

### /NEW
def calibrate_2d_monocular(cal_pt_cloud, frame_size, 
                           max_absolute_error_threshold=35,
                           max_stds_for_outliers=None,
                           recursive_outlier_cut=None,
    ):
    method = "monocular polynomial regression"

    map_fn, inliers, params = calibrate_2d_polynomial(
        cal_pt_cloud, frame_size, binocular=False, 
        max_absolute_error_threshold=max_absolute_error_threshold,
        max_stds_for_outliers=max_stds_for_outliers,
        )
    if not inliers.any():
        return method, None

    mapper = "Monocular_Gaze_Mapper"
    args = {"params": params}
    result = {"subject": "start_plugin", "name": mapper, "args": args}
    return method, result


def calibrate_2d_binocular(cal_pt_cloud_binocular, cal_pt_cloud0,
                           cal_pt_cloud1, frame_size):
    method = "binocular polynomial regression"

    map_fn, inliers, params = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud_binocular, frame_size, binocular=True
    )
    if not inliers.any():
        return method, None
    # Right eye
    map_fn, inliers, params_eye0 = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud0, frame_size, binocular=False
    )
    if not inliers.any():
        return method, None
    # Left eye
    map_fn, inliers, params_eye1 = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud1, frame_size, binocular=False
    )
    if not inliers.any():
        return method, None

    mapper = "Binocular_Gaze_Mapper"
    args = {
        "params": params,
        "params_eye0": params_eye0,
        "params_eye1": params_eye1,
    }
    result = {"subject": "start_plugin", "name": mapper, "args": args}
    return method, result


def _fit_rbf_cv(calibration_markers, pupil,
                smoothnesses=np.linspace(-0.001, 10, 100),
                methods=['thin-plate', 'multiquadric', 'linear', 'cubic'],
                ):
    from scipy.interpolate import Rbf as RBF
    """
    Fit an interpolator to the points via LOO x-validation
    @param smoothnesses:		list<float>, smoothnesses to try for interpolations
    @param methods:				list<str>, methods to try
    @param varianceThreshold:	float?, threshold of variance in the calibration positions to throw away
    @param glintVector:			bool, use pupil-glint vector instead of just the pupil position?
    @param searchThreshold:		float, if the error is above this threshold, search over delays/durations. will not search if is 0
    @return:
    """

    pupil_pos = pupil['norm_pos']
    marker_pos = calibration_markers['norm_pos']

    def LeaveOneOutXval(smoothness, method):
        """
        Leave on out estimation, returns RMS deviation from actual points
        @param smoothness:
        @param method:
        @return:
        """
        estimates = np.zeros([len(marker_pos), 2])
        for i in range(len(marker_pos)):
            fit = np.ones((len(marker_pos),)) > 0
            fit[i] = False

            horizontal = RBF(pupil_pos[fit, 0],
                             pupil_pos[fit, 1],
                             marker_pos[fit, 0],
                             function=method,
                             smooth=smoothness)

            vertical = RBF(pupil_pos[fit, 0],
                           pupil_pos[fit, 1],
                           marker_pos[fit, 1],
                           function=method,
                           smooth=smoothness)

            estimates[i, :] = [horizontal(pupil_pos[i, 0],
                                          pupil_pos[i, 1]),
                               vertical(pupil_pos[i, 0],
                                        pupil_pos[i, 1])]

        return np.sqrt(np.mean((estimates - marker_pos) ** 2))

    errors = np.zeros([len(smoothnesses), len(methods)])
    for s in range(len(smoothnesses)):
        for m in range(len(methods)):
            errors[s, m] = LeaveOneOutXval(smoothnesses[s], methods[m])

    s, m = np.unravel_index(errors.argmin(), errors.shape)
    bestSmoothness = smoothnesses[s]
    bestMethod = methods[m]
    bestError = errors[s, m]

    horizontalInterpolater = RBF(pupil_pos[:, 0],
                                 pupil_pos[:, 1],
                                 marker_pos[:, 0],
                                 function=bestMethod,
                                 smooth=bestSmoothness)

    verticalInterpolater = RBF(pupil_pos[:, 0],
                               pupil_pos[:, 1],
                               marker_pos[:, 1],
                               function=bestMethod,
                               smooth=bestSmoothness)

    #if (searchThreshold > 0) and (bestError > searchThreshold):
    #    print('Min error {:.2f} is above threshold of {:.0f} and will be searching'.format(bestError, searchThreshold))
    #    SearchAndFit()
    return horizontalInterpolater, verticalInterpolater, bestMethod, bestSmoothness, bestError

def _fit_tps_gaze(calibration_markers, pupil, lambd=0.01, min_calibration_threshold=0.6, ):

    x, y = calibration_markers['norm_pos'].T
    px, py = pupil['norm_pos'].T
    keep = pupil['confidence'] > min_calibration_threshold
    # X
    data_x = np.vstack([px, py, x]).T
    theta_x = tps.TPS.fit(data_x, lambd=lambd)
    # Y
    data_y = np.vstack([px, py, y]).T
    theta_y = tps.TPS.fit(data_y, lambd=lambd)
    return data_x, theta_x, data_y, theta_y


def _map_tps(pupil, orig_data, mapper):
    """Map pupil coordinates to video coordinates via calculated transform
    
    Note that for thin plate splines, both the X and Y pupil coordinates are used to generate
    the X component of the gaze coordinates, and then both X and Y pupil coordinates
    are used to generate the Y component of the gaze coordinates

    Parameters
    ----------
    pupil : dict
        array of dicts for pupil position. Must contain 'norm_pos'
    mapper
    """
    xp, yp = pupil['norm_pos'].T
    out = tps.TPS.z(np.vstack([xp, yp]).T, orig_data, mapper)
    return out



DEFAULT_LAMBDA_LIST = np.logspace(np.log10(1e-6), np.log10(10), 8*2)
def _fit_tps_gaze_cv(calibration_markers, pupil, lambd_list=DEFAULT_LAMBDA_LIST, 
    min_calibration_threshold=0.6, max_stds_for_outliers=None, recursive_outlier_cut=False):
    """Cross-validated thin-plate spline fit for calibration"""
    keep = pupil['confidence'] > min_calibration_threshold
    mk = filter_arraydict(calibration_markers, keep)
    pup = filter_arraydict(pupil, keep)
    x, y = mk['norm_pos'].T
    px, py = pup['norm_pos'].T
    
    # X, Y data
    data_x = np.vstack([px, py, x]).T
    data_y = np.vstack([px, py, y]).T
    errors = np.zeros((len(lambd_list), len(mk['norm_pos'])))
    all_preds = []
    for ia, lambd in enumerate(lambd_list):
        preds = np.zeros_like(mk['norm_pos'])
        for i in range(len(x)):
            # Leave one out
            fit = np.ones_like(px) > 0
            fit[i] = False
            # do fit
            theta_x = tps.TPS.fit(data_x[fit], lambd=lambd)
            theta_y = tps.TPS.fit(data_y[fit], lambd=lambd)
            # assess fit for subset
            preds[i,0] = tps.TPS.z(np.vstack([px[i], py[i]]).T,
                                   data_x[fit],
                                   theta_x)
            preds[i,1] = tps.TPS.z(np.vstack([px[i], py[i]]).T,
                                   data_y[fit],
                                   theta_y)
        # average or accumulate subset metrics for each lambda
        all_preds.append(preds)
        errors[ia, :] = np.linalg.norm(preds - mk['norm_pos'], axis=1)
    # Check for outliers
    if max_stds_for_outliers is not None:
        mean_errors_per_pt = errors.mean(0)
        mm = np.median(mean_errors_per_pt)
        ss = np.std(mean_errors_per_pt)
        # Only check for outlying HIGH errors
        outliers = mean_errors_per_pt > mm + max_stds_for_outliers * ss
        if np.any(outliers):
            print('Removing %d outlying data points'%np.sum(outliers))
            # (note: only do outliers once)
            cal_cut = filter_arraydict(mk, ~outliers)
            print(len(cal_cut['norm_pos']))
            pupil_cut = filter_arraydict(pup, ~outliers)
            print("Refitting...")
            if recursive_outlier_cut:
                mx_std = max_stds_for_outliers
            else:
                mx_std = None
            return _fit_tps_gaze_cv(cal_cut, pupil_cut, 
                             lambd_list=lambd_list, 
                             min_calibration_threshold=min_calibration_threshold, 
                             max_stds_for_outliers=mx_std)
    else:
        outliers = np.zeros_like(pup['timestamp']) > 1
    # Select lambda based on accumulated error / whatever metric
    best_lambda_i = np.argmin(errors.mean(1))
    # re-fit with that lambda
    theta_x = tps.TPS.fit(data_x, lambd=lambd_list[best_lambda_i])
    theta_y = tps.TPS.fit(data_y, lambd=lambd_list[best_lambda_i])
    # return parameters0
    return (data_x, theta_x, data_y, theta_y), outliers # , errors, all_preds



def get_point_grid(n_points=60,
                   n_horizontal_lines=7,
                   n_vertical_lines=10,
                   st_horizontal=0,
                   fin_horizontal=1,
                   st_vertical=0,
                   fin_vertical=1):
    """Get a list of (x,y) points that form a rectangular grid
    
    Parameters
    ----------
    n_points : int, optional
        Number of points across horizontal axis (vertical points
        are computed from aspect ratio given by 
        `n_lines_<horiz / vert>`)
    n_horizontal_lines : int, optional
        Number of grid lines across the plot horizontally
    n_vertical_lines : int, optional
        Number of grid lines across the plot vertically
    st_horizontal : int, optional
        Description
    fin_horizontal : int, optional
        Description
    st_vertical : int, optional
        Description
    fin_vertical : int, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    aspect_ratio = n_vertical_lines / n_horizontal_lines
    n_points_horizontal = n_points
    n_points_vertical = int(np.ceil(n_points / aspect_ratio))
    # Horizontal lines first
    tt_sparse_h = np.linspace(st_vertical, fin_vertical, n_horizontal_lines)
    tt_dense_h = np.linspace(
        st_horizontal, fin_horizontal, n_points_horizontal)
    xga, yga = np.meshgrid(tt_dense_h, tt_sparse_h)
    # Vertical lines next
    tt_sparse_v = np.linspace(st_horizontal, fin_horizontal, n_vertical_lines)
    tt_dense_v = np.linspace(st_vertical, fin_vertical, n_points_vertical)
    xgb, ygb = np.meshgrid(tt_sparse_v, tt_dense_v)
    # Combine & return
    xg_pts = np.hstack([xga.flatten(), xgb.flatten()])
    yg_pts = np.hstack([yga.flatten(), ygb.flatten()])
    return xg_pts, yg_pts


class Calibration(object):
    def __init__(self,
                 pupil_arrays,
                 calibration_arrays,
                 video_dims,
                 cluster_reduce_fn=np.median,
                 min_calibration_confidence=0.6,
                 calibration_type='monocular_tps_cv',
                 mapping_computed=False,
                 **params,
                 ):
        """Class to compute gaze mapping calibration in a few different ways.
        
        Parameters
        ----------
        pupil_arrays : dictionary of arrays or tuple of 2 
            dictionaries of arrays
        calibration_arrays : dict
            dictionary of arrays for detected calibration markers
        video_dims : tuple
            (video_vdim, video_hdim)
        min_calibration_confidence : float, optional
            Description
        calibration_type : str, optional
            Description
        
        """
        calibration_list = arraydict_to_dictlist(calibration_arrays)

        # Store for later
        self.calibration_list = calibration_list
        self.calibration_type = calibration_type
        self.min_calibration_confidence = min_calibration_confidence
        self.calibration_arrays = calibration_arrays
        self.pupil_arrays = pupil_arrays
        self.cluster_reduce_fn = cluster_reduce_fn
        self.video_dims = video_dims
        self.params = params
        self.mapping_computed = mapping_computed
        # Optionally load if params are not a dict
        if 'tag' in self.params:
            fpath = BASE_DIR / 'config' / \
                ('calibration_%s.yaml' % self.params['tag'])
            cal_params = read_yaml(fpath)
            if 'min_calibration_confidence' in cal_params:
                self.min_calibration_confidence = cal_params.pop(
                    'min_calibration_confidence')
            self.params = cal_params
        # Binocular or monocular calibration
        if isinstance(pupil_arrays, (list, tuple)) and len(pupil_arrays) == 2:
            left_pupil, right_pupil = self.pupil_arrays
            if self.mapping_computed:
                self.pupil_left = left_pupil
                self.pupil_right = right_pupil
            else:
                self.pupil_left = match_time_points(
                    calibration_arrays, left_pupil, )
                self.pupil_right = match_time_points(
                    calibration_arrays, right_pupil, )
                self.pupil_arrays = [self.pupil_left, self.pupil_right]
            self.pupil_list = arraydict_to_dictlist(left_pupil)
            self.pupil_list.extend(arraydict_to_dictlist(right_pupil))
            # Sort both left and right pupils by time. `id` field in each dict
            # will (SHOULD!) still indicate which eye is which. 
            self.pupil_list = sorted(self.pupil_list, key=lambda x: x['timestamp'])
        else:
            if not self.mapping_computed:
                self.pupil_arrays = match_time_points(
                    calibration_arrays, pupil_arrays)
            self.pupil_list = arraydict_to_dictlist(pupil_arrays)

        # Define mapping functions
        if not hasattr(self, 'map_params'):
            self._get_map_params()
        self._get_mapper()

    def _get_map_params(self):
        # Run calibration
        if self.params is None:
            kws = {}
        else:
            kws = self.params
        if self.cluster_reduce_fn is None:
            marker_input = self.calibration_arrays
            pupil_input = self.pupil_arrays
        else:
            # Since we are reducing data points fed to calibration
            # down to 1, perform data selection based on pupil
            # confidence first:
            fn = get_function(self.cluster_reduce_fn)
            keep = self.pupil_arrays['confidence'] > self.min_calibration_confidence
            marker_input_ = filter_arraydict(self.calibration_arrays, keep)
            marker_input = marker_cluster_stat(marker_input_,
                                                fn=fn,
                                                field='norm_pos',
                                                return_all_fields=True,
                                                clusters=None
                                                )

            pupil_input_ = filter_arraydict(self.pupil_arrays, keep)
            pupil_input = {}
            pupil_input['timestamp'] = marker_input['timestamp']
            pupil_input['norm_pos'] = marker_cluster_stat(pupil_input_,
                                                            field='norm_pos',
                                                            fn=fn,
                                                            clusters=marker_input_[
                                                                'marker_cluster_index'],
                                                            return_all_fields=False,
                                                            )
            pupil_input['confidence'] = marker_cluster_stat(pupil_input_,
                                                            field='confidence',
                                                            fn=fn,
                                                            clusters=marker_input_[
                                                                'marker_cluster_index'],
                                                            return_all_fields=False,
                                                            )
        if self.calibration_type == 'monocular_pl':
            # NOTE: zero index for matched_data here is because this is monocular,
            # and matched data only returns a 1-long tuple.
            if 'max_stds_for_outliers' in kws:
                _, outliers = _fit_tps_gaze_cv(marker_input, pupil_input,
                                            min_calibration_threshold=self.min_calibration_confidence, 
                                            **kws)
                marker_input = filter_arraydict(marker_input, ~outliers)
            marker_list_input = arraydict_to_dictlist(marker_input)
            is_binocular, matched_data = parse_plab_data(self.pupil_list,       
                                            marker_list_input, mode='2d',
                                            min_calibration_confidence=self.min_calibration_confidence)
            method, result = calibrate_2d_monocular(
                matched_data[0], frame_size=self.video_dims, **kws)
            self.map_params = result["args"]["params"]
            #cx, cy, n = self.map_params
        elif self.calibration_type == 'binocular_pl':
            is_binocular, matched_data = parse_plab_data(self.pupil_list, self.calibration_list, mode='2d',
                                                               min_calibration_confidence=self.min_calibration_confidence)
            method, result = calibrate_2d_binocular(
                *matched_data, frame_size=self.video_dims)
            self.map_params = result['args']
        elif self.calibration_type in ('monocular_tps', 'monocular_tps_cv'):
            if self.calibration_type == 'monocular_tps':
                self.map_params = _fit_tps_gaze(marker_input, pupil_input,
                                            min_calibration_threshold=self.min_calibration_confidence, 
                                            **kws)
            elif self.calibration_type == 'monocular_tps_cv':
                self.map_params, outliers = _fit_tps_gaze_cv(marker_input, pupil_input,
                                            min_calibration_threshold=self.min_calibration_confidence, 
                                            **kws)
    def _get_mapper(self):
        if self.calibration_type == 'monocular_pl':
            self._mapper = calibrate_2d.make_map_function(
                *self.map_params)
        elif self.calibration_type == 'binocular_pl':
            self._mapper = Binocular_Gaze_Mapper(
                self.map_params["params"],
                self.map_params["params_eye0"],
                self.map_params["params_eye1"])
        elif self.calibration_type in ('monocular_tps', 'monocular_tps_cv'):
            data_x, mapper_x, data_y, mapper_y = self.map_params
            # For thin plate splines, both the X and Y pupil coordinates are used to generate
            # the X component of the gaze coordinates, and then both X and Y pupil coordinates
            # are used to generate the Y component of the gaze coordinates
            def map_xy_tps(pupil_arrays):
                xtps = _map_tps(pupil_arrays, data_x, mapper_x)
                ytps = _map_tps(pupil_arrays, data_y, mapper_y)
                gaze = np.vstack([xtps, ytps]).T
                return gaze
            self._mapper = map_xy_tps

    def map(self, pupil_arrays, return_type='arraydict'):
        # Map gaze to video coordinates
        if self.calibration_type == 'binocular_pl':
            # Create a list of dicts, sorted by timestamps
            #if not isinstance(pupil_arrays, list):
            #    raise ValueError('For binocular calibrations, `pupil_arrays` input must be a list of two dictionaries\n of arrays for left and right eyes.')
            # Left
            pupil_list = arraydict_to_dictlist(pupil_arrays[0])
            # Right
            pupil_list.extend(arraydict_to_dictlist(pupil_arrays[1]))
            pupil_list = sorted(pupil_list, key=lambda x: x['timestamp'])
            # Map w/ pupil's binocular mapper
            gaze = self._mapper.map_batch(pupil_list)
            # Convert back to array of dictionaries & reformat
            gaze_array = dictlist_to_arraydict(gaze)
            #gaze_array['norm_pos'] = gaze_array.pop('norm_pos')
        elif self.calibration_type == 'monocular_pl':
            # Mapper takes two inputs: normalized pupil x and y position
            x, y = pupil_arrays['norm_pos'].T
            gaze = self._mapper([x, y])
            # Transpose output so time is the first dimension
            gaze = np.vstack(gaze).T
            gaze_array = dict(timestamp=pupil_arrays['timestamp'],
                              norm_pos=gaze)
            if 'confidence' in pupil_arrays:
                gaze_array['confidence']=pupil_arrays['confidence']
        elif self.calibration_type in ('monocular_tps', 'monocular_tps_cv'):
            gaze = self._mapper(pupil_arrays)
            gaze_array = dict(timestamp=pupil_arrays['timestamp'],
                              norm_pos=gaze)
            if 'confidence' in pupil_arrays:
                gaze_array['confidence'] = pupil_arrays['confidence']

        if return_type == 'arraydict':
            return gaze_array
        elif return_type == 'dictlist':
            return arraydict_to_dictlist(gaze_array)            
        elif return_type == 'array':
            return gaze_array['norm_pos']

    @property
    def calibration_data(self):
        pp = dict(
            map_params=self.map_params,
            calibration_arrays=self.calibration_arrays,
            pupil_arrays=self.pupil_arrays,
            video_dims=self.video_dims,
            calibration_type=self.calibration_type,
        )
        return pp

    def save(self, fpath):
        """Save critical data to do mappings of pupil to gaze"""
        np.savez(fpath, **self.calibration_data)

    @classmethod
    def _get_fpath(base_dir, folder, eye, calibration_tag, pupil_tag, marker_type, marker_tag, epoch):
        """Get filepath to save, given elements, assuming this is not for database storage"""
        if epoch == 'all':
            fname = f'calibration_{calibration_tag}_{eye}_{pupil_tag}_{marker_type}_epoch{epoch}_{marker_tag}.npz'
        else:
            fname = f'calibration_{calibration_tag}_{eye}_{pupil_tag}_{marker_type}_epoch{epoch:02d}_{marker_tag}.npz'
        fpath = base_dir / folder / fname
        return fpath

    @classmethod
    def load(cls, fpath):        
        ob = cls.__new__(cls)
        tmp = np.load(fpath, allow_pickle=True)
        params = {}
        for k, v in tmp.items():
            if isinstance(v, np.ndarray) and (v.dtype == np.dtype('O')):
                if v.shape == ():
                    params[k] = v.item()
                else:
                    params[k] = v
            else:
                params[k] = v
        # This next line - setting a property before calling __init__ - feels somewhat shady... 
        # in conjunction with the if clause in __init__ above, seems to do what is desired.
        ob.map_params = params.pop('map_params')
        ob.__init__(**params)
        #ob._get_mapper()
        return ob

    def show_calibration(self,
                         im=None,
                         eye='left',
                         n_horizontal_lines=12,
                         n_vertical_lines=None,
                         n_points=40,
                         point_size=3,
                         show_data=False,
                         data_scale=20,
                         sc=0.0,
                         color=((1.0, 0.9, 0),),
                         ax_world='new',
                         ax_eye=None,
                         ):
        """Plot visualization of calibration manifold on eye and 
        
        Parameters
        ----------
        im : None, optional
            Description
        eye : str, optional
            Description
        n_horizontal_lines : int, optional
            Description
        n_vertical_lines : int or None, optional
            if None, computed from n_horizontal_lines and aspect ratio
            of pupil points.
        n_lines_vert : int, optional
            Description
        """

        if ax_eye is not None:
            if isinstance(ax_eye, (list, tuple)):
                ax_left, ax_right = ax_eye
                # recursive call to plot l, r eye grids on l, r axes
        if self.calibration_type == 'binocular_pl':
            # Start with grid of locations for left
            grid_pts_array_l = self._get_grid_points(
                'left', 
                n_points=n_points, 
                sc=sc, 
                n_horizontal_lines=n_horizontal_lines, 
                return_type='arraydict')
            # Fill in `_id`` and `confidence` fields
            grid_pts_array_l['id'] = np.ones_like(
                grid_pts_array_l['timestamp'], dtype=int64)
            grid_pts_array_l['confidence'] = np.ones_like(
                grid_pts_array_l['timestamp'], dtype=float32)
            # Continue with grid of locations for right
            grid_pts_array_r = self._get_grid_points(
                'right', 
                n_points=n_points, 
                sc=sc, 
                n_horizontal_lines=n_horizontal_lines, 
                return_type='arraydict')
            # Fill in `_id`` and `confidence` fields
            grid_pts_array_r['id'] = np.zeros_like(
                grid_pts_array_r['timestamp'], dtype=int64)
            grid_pts_array_r['confidence'] = np.ones_like(
                grid_pts_array_r['timestamp'], dtype=float32)
            if (grid_pts_array_l is None) or (grid_pts_array_r is None):
                grid_array = None
            else:
                gaze_binocular = self.map([grid_pts_array_l, grid_pts_array_r], 
                                          return_type='dictlist')
                gaze = {}
                # Separate left and right eyes
                gaze['left'] = [
                    g for g in gaze_binocular if g['base_data'][0]['id'] == 0]
                gaze['left'] = dictlist_to_arraydict(gaze['left'])
                gaze['right'] = [
                    g for g in gaze_binocular if g['base_data'][0]['id'] == 1]
                gaze['right'] = dictlist_to_arraydict(gaze['right'])
                grid_array = gaze[eye]['norm_pos']
        elif self.calibration_type in ('monocular_pl', 'monocular_tps', 'monocular_tps_cv'):
            grid_pts_list = self._get_grid_points(
                eye, n_points=n_points, sc=sc, n_horizontal_lines=n_horizontal_lines, return_type='arraydict')
            if grid_pts_list is None:
                grid_array = None
            else:
                grid_array = self.map(grid_pts_list, return_type='array')
        x, y = self.calibration_arrays['norm_pos'].T
        cols = colormap_2d(x, y)
        ci = self.pupil_arrays['confidence'] > self.min_calibration_confidence
        if ax_world is not None:
            if ax_world == 'new':
                fig, ax_world = plt.subplots()
            if im is not None:
                ax_world.imshow(im, extent=[0, 1, 1, 0])
            if grid_array is not None:
                ax_world.scatter(*grid_array.T, s=point_size, c=color)
            if show_data:
                sc_h = ax_world.scatter(x[ci], y[ci],
                    s=self.pupil_arrays['confidence'][ci]*data_scale,
                    c=cols[ci],
                    )
            ax_world.axis([0, 1, 1, 0])
        if ax_eye is not None:
            eye_array = self._get_grid_points(
                eye, n_points=n_points, sc=sc, n_horizontal_lines=n_horizontal_lines, return_type='array')
            if eye_array is not None:
                ax_eye.scatter(*eye_array.T, s=point_size, c=color)
            if show_data:
                ax_eye.scatter(*self.pupil_arrays['norm_pos'][ci].T,
                            s=self.pupil_arrays['confidence'][ci] * data_scale,
                            # self.calibration_arrays['marker_cluster_index'][ci],
                            c=cols[ci],
                            #cmap=vedb_gaze.visualization.cluster_cmap,
                            )

            #ax_eye.axis([0, 1, 1, 0])

    def _get_grid_points(self, eye=None, sc=0.0, n_horizontal_lines=10, n_vertical_lines=None, n_points=40, return_type='dictlist', min_possible_eye_points=10):
        """Get grid points for displaying calibration
        
        Parameters
        ----------
        eye : TYPE
            Description
        """
        # ASSUMPTION:
        eye_camera_fps = 120
        if 'binocular' not in self.calibration_type: 
            pupils = self.pupil_arrays
        else:
            pupils = getattr(self, 'pupil_%s' % eye)
        ci = pupils['confidence'] > self.min_calibration_confidence
        if ci.sum() < min_possible_eye_points:
            return None
        x, y = pupils['norm_pos'][ci].T
        vmin, vmax = np.nanmin(y), np.nanmax(y)
        hmin, hmax = np.nanmin(x), np.nanmax(x)
        height = np.ptp(y[~np.isnan(y)])
        width = np.ptp(x[~np.isnan(x)])
        vmin += height * sc
        vmax -= height * sc
        hmin += width * sc
        hmax -= width * sc
        aspect_ratio = width / height
        pts = get_point_grid(
            n_horizontal_lines=n_horizontal_lines,
            n_vertical_lines=int(
                n_horizontal_lines * aspect_ratio) if n_vertical_lines is None else n_vertical_lines,
            st_horizontal=hmin,
            fin_horizontal=hmax,
            st_vertical=vmin,
            fin_vertical=vmax,
            n_points=n_points,
        )
        pts = np.array(pts).T
        grid_pts = dict(norm_pos=pts, timestamp=np.arange(len(pts))/eye_camera_fps)
        if return_type == 'array':
            return pts
        elif return_type == 'arraydict':
            return grid_pts
        elif return_type == 'dictlist':
            grid_pts_list = arraydict_to_dictlist(grid_pts)
            return grid_pts_list
