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
                  method='tps', 
                  error_smoothing_kernels=None, 
                  vertical_horizontal_smooth_error_resolution=0.25, 
                  lambd=1.0, 
                  extrapolate=False, 
                  min_pupil_confidence=0, 
                  cluster_reduce_fn=None,
                  image_resolution=(2048, 1536),
                  degrees_horiz=125,
                  degrees_vert=111,):
    """Compute error at set points and interpolation between those points

    Parameters
    ----------
    marker : array
        estimated marker position and confidence; needs field 'norm_pos' only ()
    gaze : _type_
        _description_
    method : str, optional
        _description_, by default 'tps'
    error_smoothing_kernels : _type_, optional
        _description_, by default None
    vertical_horizontal_smooth_error_resolution : _type_, optional
        vertical and horizontal resolution of the smoothed error estimate. 
        If None, defaults to 0.25 * `image_resolution`
    lambd : float, optional
        lambda parameter for thin plate spline smoothing, by default 0.001
    extrapolate : bool, optional
        flag for whether to estimate error outside of locations for validation
        markers (i.e. whether to extrapolate), by default False
    min_pupil_confidence : int, optional
        _description_, by default 0
    image_resolution : tuple, optional
        _description_, by default (2048, 1536)
    degrees_horiz : int, optional
        _description_, by default 125
    degrees_vert : int, optional
        _description_, by default 111

    Returns
    -------
    _type_
        _description_
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
    # Angle of error
    err_vector = gz_image - vp_image
    gaze_err_angle = np.arctan2(*err_vector.T)
    if vertical_horizontal_smooth_error_resolution is None:
        vertical_horizontal_smooth_error_resolution = 0.25
    if not isinstance(vertical_horizontal_smooth_error_resolution, (list, tuple)):
        hres, vres = (np.array(image_resolution) * vertical_horizontal_smooth_error_resolution).astype(np.int)
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
            tmp = np.nan_to_num(gaze_err_image, nan=np.nanmax(gaze_err))
            tmp = cv2.blur(tmp, error_smoothing_kernels)
            tmp[np.isnan(gaze_err_image)] = np.nan
            gaze_err_image = tmp
    elif method=='tps':
        x, y = marker_pos.T
        # Left
        to_fit = np.vstack([x, y, gaze_err]).T
        theta = tps.TPS.fit(to_fit, lambd=lambd)
        gaze_err_image = tps.TPS.z(np.vstack([xg.flatten(), yg.flatten()]).T, to_fit, theta).reshape(*xg.shape)
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
                gaze_matched=gaze_pos,
                marker=marker_pos,
                xgrid=xg,
                ygrid=yg)


