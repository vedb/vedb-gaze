"""Computation of gaze error at validation markers.

Provides `compute_error`, the function named by ``config/error-<tag>.yaml``
and run by the `pipelines.compute_error` step. Error is the distance between
mapped gaze and validation marker positions, in degrees of visual angle
(approximated from image resolution and camera field of view), and is
interpolated / smoothed across the image.
"""
import numpy as np
from scipy import interpolate

from .utils import match_time_points, get_function
from .marker_parsing import marker_cluster_stat

try:
    import thinplate as tps # library here: 
except ImportError:
    print(("`thinplate` library not found.\n"
           "Please download and install py-thinplate-spline\n"
           "( http://github.com/cheind/py-thin-plate-spline )\n"
           "if you wish to use thin plate splines for calibration.\b"
           ))



def compute_error(marker, 
                  gaze, 
                  method='tps_cv', 
                  error_smoothing_kernels=None, 
                  vertical_horizontal_smooth_error_resolution=(300, 400), 
                  lambd=(1e-06,
                         2.9286445646252375e-06,
                         8.576958985908945e-06,
                         2.5118864315095822e-05,
                         7.356422544596421e-05,
                         0.00021544346900318845,
                         0.000630957344480193,
                         0.0018478497974222907,
                         0.0054116952654646375,
                         0.01584893192461114,
                         0.04641588833612782,
                         0.1359356390878527,
                         0.3981071705534969,
                         1.165914401179831,
                         3.414548873833601,
                         10.0),
                  outlier_stds=4,
                  extrapolate=False, 
                  min_pupil_confidence=0.6, 
                  cluster_reduce_fn=np.median,
                  image_resolution=(2048, 1536),
                  degrees_horiz=101,
                  degrees_vert=75.75,):
    """Compute error at set points and interpolation between those points

    Gaze is matched in time to marker detections, low-confidence points are
    dropped, points are optionally reduced to one per marker cluster, and
    outliers are removed. Error (in degrees) at the remaining marker
    positions is then interpolated over a grid covering the image.

    Parameters
    ----------
    marker : dict of arrays
        validation marker detections, with fields 'timestamp', 'norm_pos'
        (n, 2) and, if `cluster_reduce_fn` is not None, 'marker_cluster_index'
    gaze : dict of arrays
        estimated gaze, with fields 'timestamp', 'norm_pos' (normalized 0-1)
        and 'confidence'. If lengths differ from `marker`, gaze is matched to
        marker timestamps with `utils.match_time_points`.
    method : str, optional
        interpolation of error across the image: 'griddata' (cubic
        interpolation), 'tps' (thin-plate spline with smoothing `lambd`), or
        'tps_cv' (thin-plate spline with `lambd` chosen from a list by
        leave-one-out cross-validation), by default 'tps_cv'
    error_smoothing_kernels : tuple, optional
        kernel size for `cv2.blur` smoothing of the error image; used only for
        method 'griddata', by default None (no smoothing)
    vertical_horizontal_smooth_error_resolution : tuple or float, optional
        (vertical, horizontal) size of the error image grid; a scalar is
        instead a fraction of `image_resolution`. If None, defaults to
        0.25 * `image_resolution`. By default (300, 400).
    lambd : float or sequence of floats, optional
        thin-plate spline smoothing parameter: a single float for 'tps', or a
        sequence of candidate values for 'tps_cv', by default 16 log-spaced
        values from 1e-6 to 10
    outlier_stds : float, optional
        criterion for excluding outlying error estimates; error estimates will be
        excluded if more than this number of standard deviations from the gaze
        error median, by default 4; None skips outlier removal
    extrapolate : bool, optional
        flag for whether to estimate error outside of locations for validation
        markers (i.e. whether to extrapolate), by default False
    min_pupil_confidence : float, optional
        minimum gaze 'confidence' for a point to be used, by default 0.6
    cluster_reduce_fn : callable or str, optional
        function (or importable name, e.g. 'numpy.median') used to reduce
        marker and gaze positions to one point per marker cluster, by default
        np.median; None uses all points
    image_resolution : tuple, optional
        (horizontal, vertical) size of the world video in pixels, by default
        (2048, 1536)
    degrees_horiz : float, optional
        horizontal field of view of the world camera in degrees, by default 101
    degrees_vert : float, optional
        vertical field of view of the world camera in degrees, by default 75.75

    Returns
    -------
    error : dict
        'gaze_err' : error (degrees) at each retained point (or cluster);
        'gaze_err_angle' : direction of each error vector (from marker to
        gaze), radians in (-pi, pi]: 0 = gaze above the marker, increasing
        clockwise on the image (pi/2 = right, +/-pi = below, -pi/2 = left).
        Same convention as `visualization.angle_hist` (which takes degrees:
        ``angle_hist(np.degrees(error['gaze_err_angle']))``). Computed as
        arctan2(dx, -dy) on the pixel error vector (image y increases
        downward). NOTE: before 2026-10, this was arctan2(dx, dy) (0 = below,
        counterclockwise); error files computed earlier use that convention.
        'gaze_err_image' : (vres, hres) interpolated error (degrees), NaN
        outside the validated area unless `extrapolate`; floored at the
        minimum point error;
        'gaze_err_weighted' : mean of `gaze_err_image` weighted by the
        histogram of ALL gaze positions in `gaze`, over the non-NaN region;
        'gaze_fraction_excluded' : fraction of gaze points falling in NaN
        (non-validated) regions of `gaze_err_image`;
        'gaze_time' : marker timestamps of points passing the confidence
        threshold;
        'gaze_matched' : retained gaze positions (normalized);
        'marker' : retained marker positions (normalized);
        'xgrid', 'ygrid' : normalized coordinates of the error image grid.

    Raises
    ------
    ValueError
        if fewer than 4 points remain after filtering, or if
        `cluster_reduce_fn` is set but `marker` has no 'marker_cluster_index'
    """
    # Pixels per degree, coarse estimate for error computation
    # Default degrees are 125 x 111, this assumes all data is collected 
    # w/ standard size, which is not true given new lenses (defaults must be
    # updated for new lenses)
    hppd = image_resolution[0] / degrees_horiz
    vppd = image_resolution[1] / degrees_vert
    # Coarse, so it goes
    ppd = np.mean([vppd, hppd])
    # reduction function, if applicable
    if cluster_reduce_fn is not None:
        cluster_reduce_fn = get_function(cluster_reduce_fn)
    # Marker positions, matched in time
    marker_pos = marker['norm_pos'].copy()
    # Estimated gaze position, in normalized (0-1) coordinates
    gaze_pos = gaze['norm_pos'].copy()
    if len(gaze['timestamp']) != len(marker['timestamp']):
        print('matching time points...')
        gaze_matched = match_time_points(marker, gaze)
        gaze_pos = gaze_matched['norm_pos']
    else:
        gaze_matched = gaze

    # Gaze confidence index (gz_ci)
    gz_ci = gaze_matched['confidence'] > min_pupil_confidence
    marker_pos = marker_pos[gz_ci]
    gaze_pos = gaze_pos[gz_ci]
    
    if cluster_reduce_fn is not None:
        if not 'marker_cluster_index' in marker:
            raise ValueError("No clusters detected, can't perform cluster reduction and cross validation of lambda parameter")
        else:
            clusters = marker['marker_cluster_index'][gz_ci]
            marker_pos = marker_cluster_stat(dict(marker_pos=marker_pos),
                                            fn=cluster_reduce_fn,
                                            field='marker_pos',
                                            return_all_fields=False,
                                            clusters=clusters
                                            )
            gaze_pos = marker_cluster_stat(dict(gaze_pos=gaze_pos),
                                            fn=cluster_reduce_fn,
                                            field='gaze_pos',
                                            return_all_fields=False,
                                            clusters=clusters
                                            )
    
    vp_image = marker_pos * np.array(image_resolution)
    gz_image = gaze_pos * image_resolution
    # Magnitude of error
    gaze_err = np.linalg.norm(gz_image - vp_image, axis=1) / ppd
    if outlier_stds is not None:
        # Remove ridiculous outliers
        ss = np.std(gaze_err)
        mm = np.median(gaze_err)
        outliers = np.abs(gaze_err - mm) > outlier_stds * ss
        # Cull from all variables
        gaze_err = gaze_err[~outliers]
        vp_image = vp_image[~outliers]
        gz_image = gz_image[~outliers]
        marker_pos = marker_pos[~outliers]
        gaze_pos = gaze_pos[~outliers]
    # Check whether we have cut too many markers
    if len(marker_pos) < 4:
        raise ValueError('Too few points to compute error across visual field.')

    # Direction of error: 0 = up, clockwise on the image (image y points down),
    # matching visualization.angle_hist
    err_vector = gz_image - vp_image
    dx, dy = err_vector.T
    gaze_err_angle = np.arctan2(dx, -dy)
    if vertical_horizontal_smooth_error_resolution is None:
        vertical_horizontal_smooth_error_resolution = 0.25
    if not isinstance(vertical_horizontal_smooth_error_resolution, (list, tuple)):
        hres, vres = (np.array(image_resolution) * vertical_horizontal_smooth_error_resolution).astype(int)
    else:
        vres, hres = vertical_horizontal_smooth_error_resolution
    # Interpolate to get error over whole image
    vpix = np.linspace(0, 1, vres)
    hpix = np.linspace(0, 1, hres)
    xg, yg = np.meshgrid(hpix, vpix)
    # Grid interpolation, basic
    if gz_ci.sum() == 0:
        tmp = np.ones_like(xg) * np.nan
    else:
        tmp = interpolate.griddata(marker_pos, np.nan_to_num(gaze_err, nan=np.nanmean(gaze_err)), (xg, yg), method='cubic', fill_value=np.nan)
    if method=='griddata':
        gaze_err_image = tmp
        if error_smoothing_kernels is not None:
            import cv2
            tmp = np.nan_to_num(gaze_err_image, nan=np.nanmax(gaze_err))
            tmp = cv2.blur(tmp, error_smoothing_kernels)
            tmp[np.isnan(gaze_err_image)] = np.nan
            gaze_err_image = tmp
    elif method=='tps':
        x, y = marker_pos.T
        to_fit = np.vstack([x, y, gaze_err]).T
        theta = tps.TPS.fit(to_fit, lambd=lambd)
        gaze_err_image = tps.TPS.z(np.vstack([xg.flatten(), yg.flatten()]).T, to_fit, theta).reshape(*xg.shape)
        if not extrapolate:
            gaze_err_image[np.isnan(tmp)] = np.nan
    elif method == 'tps_cv':
        x, y = marker_pos.T
        to_fit = np.vstack([x, y, gaze_err]).T
        errs = np.zeros((len(lambd),))
        for i, this_lambd in enumerate(lambd):
            #print(f'=== lambda : {lambd} ===')
            err_pred = np.zeros((len(to_fit),))
            for j in range(len(to_fit)):
                cv_keep = np.ones((len(to_fit),)) > 0
                cv_keep[j] = False
                theta = tps.TPS.fit(to_fit[cv_keep], lambd=this_lambd)
                err_pred[j] = tps.TPS.z(to_fit[j,:2], to_fit[cv_keep], theta)
            errs[i] = np.sqrt(np.mean((err_pred - gaze_err)**2))
        lambda_i = np.argmin(errs)
        theta = tps.TPS.fit(to_fit, lambd=lambd[lambda_i])
        gaze_err_image = tps.TPS.z(
            np.vstack([xg.flatten(), yg.flatten()]).T, to_fit, theta).reshape(*xg.shape)
        if not extrapolate:
            gaze_err_image[np.isnan(tmp)] = np.nan
    # Do not allow any error values lower than minimum estimated error
    gaze_err_image = np.maximum(gaze_err_image, np.min(gaze_err))
    # Compute weighted error for whole session
    gx, gy = gaze['norm_pos'].T
    ny, nx = xg.shape
    bin_x = np.linspace(0, 1, nx+1)
    bin_y = np.linspace(0, 1, ny+1)
    hst, bin_x_, bin_y_ = np.histogram2d(gx, gy, [bin_x, bin_y])
    hst = hst.T
    hst_pct = hst / hst.sum()
    total_gaze_points_in_image = hst.sum()
    total_interpolated_gaze_points = np.sum(hst[~np.isnan(gaze_err_image)])
    total_extrapolated_gaze_points = np.sum(hst[np.isnan(gaze_err_image)])
    gaze_err_weighted = np.nansum((hst_pct) * gaze_err_image) / \
        (total_interpolated_gaze_points / total_gaze_points_in_image)
    fraction_excluded = total_extrapolated_gaze_points / total_gaze_points_in_image
    
    return dict(gaze_err=gaze_err, 
                gaze_err_angle=gaze_err_angle,
                gaze_err_image=gaze_err_image,
                gaze_err_weighted=gaze_err_weighted,
                gaze_fraction_excluded=fraction_excluded,
                gaze_time=marker['timestamp'][gz_ci],
                gaze_matched=gaze_pos,
                marker=marker_pos,
                xgrid=xg,
                ygrid=yg)


