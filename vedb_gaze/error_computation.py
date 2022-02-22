import numpy as np
from scipy import interpolate
try:
    import thinplate as tps # library here: 
except ImportError:
    print(("`thinplate` library not found.\n"
           "Please download and install py-thinplate-spline\n"
           "( http://github.com/cheind/py-thin-plate-spline )\n"
           "if you wish to use thin plate splines for calibration.\b"
           ))



def compute_error(marker_pos, 
                  gaze_left, 
                  gaze_right, 
                  method='tps', 
                  error_smoothing_kernels=None, 
                  vhres=None, 
                  lambd=0.001, 
                  extrapolate=False, 
                  confidence_threshold=0, 
                  image_resolution=(2048, 1536),
                  degrees_horiz=125,
                  degrees_vert=111,):
    """Compute error at set points and interpolation between those points

    Parameters
    ----------
    marker : array
        estimated marker position and confidence; needs field 'norm_pos' only ()

    """
    # Pixels per degree, coarse estimate for error computation
    # Default degrees are 125 x 111, this assumes all data is collected 
    # w/ standard size, which is not true given new lenses (defaults must be
    # updated for new lenses)
    hppd = image_resolution[0] / degrees_horiz
    vppd = image_resolution[1] / degrees_vert
    # Coarse, so it goes
    ppd = np.mean([vppd, hppd])
    
    # < This section will vary with input format >
    # Estimated gaze position, in normalized (0-1) coordinates
    gl = gaze_left['position']
    # Gaze left - confidence index (gl_ci)
    gl_ci = gaze_left['confidence'] > confidence_threshold
    gr = gaze_right['position']
    # Gaze right - confidence index (gl_ci)
    gr_ci = gaze_right['confidence'] > confidence_threshold
    # < To here> 

    vp_image = marker_pos * image_resolution
    gl_image = gl * image_resolution
    gr_image = gr * image_resolution

    err_left = np.linalg.norm(gl_image[gl_ci] - vp_image[gl_ci], axis=1) / ppd
    err_right = np.linalg.norm(gr_image[gr_ci] - vp_image[gr_ci], axis=1) / ppd
    if vhres is None:
        hres, vres = (np.array(image_resolution) * 0.25).astype(np.int)
    else:
        vres, hres = vhres
    # Interpolate to get error over whole image
    vpix = np.linspace(0, 1, vres)
    hpix = np.linspace(0, 1, hres)
    xg, yg = np.meshgrid(hpix, vpix)
    # Grid interpolation, basic
    if gl_ci.sum() == 0:
        tmp_l = np.ones_like(xg) * np.nan
    else:
        tmp_l = interpolate.griddata(vp[gl_ci], np.nan_to_num(err_left, nan=np.nanmean(err_left)), (xg, yg), method='cubic', fill_value=np.nan)
    if gr_ci.sum() == 0:
        tmp_r = np.ones_like(xg) * np.nan
    else:
        tmp_r = interpolate.griddata(vp[gr_ci], np.nan_to_num(err_right, nan=np.nanmean(err_right)), (xg, yg), method='cubic', fill_value=np.nan)

    if method=='griddata':
        err_left_image = tmp_l
        err_right_image = tmp_r
        if error_smoothing_kernels is not None:
            tmp_l = np.nan_to_num(err_left_image, nan=np.nanmax(err_left))
            tmp_r = np.nan_to_num(err_right_image, nan=np.nanmax(err_right))
            tmp_l = cv2.blur(tmp_l, error_smoothing_kernels)
            tmp_r = cv2.blur(tmp_r, error_smoothing_kernels)
            tmp_l[np.isnan(err_left_image)] = np.nan
            tmp_r[np.isnan(err_right_image)] = np.nan
            err_left_image = tmp_l
            err_right_image = tmp_r

    elif method=='tps':
        x, y = marker_pos.T
        # Left
        to_fit_l = np.vstack([x[gl_ci], y[gl_ci], err_left]).T
        theta_l = tps.TPS.fit(to_fit_l, lambd=lambd)
        err_left_image = tps.TPS.z(np.vstack([xg.flatten(), yg.flatten()]).T, to_fit_l, theta_l).reshape(*xg.shape)
        # Right
        to_fit_r = np.vstack([x[gr_ci], y[gr_ci], err_right]).T
        theta_r = tps.TPS.fit(to_fit_r, lambd=lambd)
        err_right_image = tps.TPS.z(np.vstack([xg.flatten(), yg.flatten()]).T, to_fit_r, theta_r).reshape(*xg.shape)
        if not extrapolate:
            err_left_image[np.isnan(tmp_l)] = np.nan
            err_right_image[np.isnan(tmp_r)] = np.nan
            
    return dict(left=err_left, 
                right=err_right, 
                left_image=err_left_image, 
                right_image=err_right_image,
                left_marker_pos=marker_pos[gl_ci],
                right_marker_pos=marker_pos[gr_ci],
                xgrid=xg,
                ygrid=yg)

