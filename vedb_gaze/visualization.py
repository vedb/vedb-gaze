from matplotlib import animation
from vedb_gaze.utils import match_time_points
from sklearn import pipeline
import plot_utils
import copy
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import animation, patches, colors, gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize

from .utils import match_time_points

# Colors for clusters
np.random.seed(107)
color_values = np.linspace(0, 1, 256)
np.random.shuffle(color_values)
cluster_cmap = ListedColormap(plt.cm.hsv(color_values))

# 2D colormap
t = np.linspace(0, 1, 256)
x, y = np.meshgrid(t, t)
r = x
g = (x * 0.5 + y * 0.5)
b = y
RdBu_2D = np.dstack([r, np.zeros_like(r), b])[::-1]
BuOr_2D = np.dstack([r, g, b])[::-1]


def gaze_hist(gaze, 
    confidence_threshold=0, 
    cmap='gray_r',
    hist_bins_x=81, 
    hist_bins_y=61, 
    field='norm_pos', 
    ax=None):
    """Make a 2D histogram of gaze positions

    Parameters
    ----------
    gaze : dict
        dict of arrays, with at least fields for `field` (kwarg below) 
        and 'confidence'
    confidence_threshold : float, optional
        minimum confidence for plotted points, by default 0.0
    cmap : str, optional
        color map for 2D histogram, by default 'gray_r'
    hist_bins_x : int, optional
        number of histogram bins on horizontal (x) axis, by default 81
    hist_bins_y : int, optional
        number of histogram bins on vertical (y) axis, by default 61
    field : str, optional
        name of field within gaze dict to plot, by default 'norm_pos' 
        (potentially e.g. 'norm_pos', or 0-1 normalized position by vedb 
        naming conventions)
    ax : matplotlib axis, optional
        axis into which to plot histogram, by default None, which creates
        a new figure & axis

    Returns
    -------
    im_handle
        handle for histogram image plotted
    """

    if ax is None:
        fig, ax = plt.subplots()
    # Compute histogram of gaze to show too
    ci = gaze['confidence'] > confidence_threshold
    x, y = gaze[field][ci].T
    im_kw_hist = dict(
        extent=[0, 1, 1, 0],
        aspect='auto',
        cmap=cmap)
    hst = np.histogram2d(y, x, bins=[np.linspace(
        0, 1, hist_bins_y), np.linspace(0, 1, hist_bins_x)], density=True)
    vmax_hst = np.percentile(hst[0], 97.5)
    hst_im = ax.imshow(hst[0], vmax=vmax_hst, **im_kw_hist)
    return hst_im


def plot_timestamps(timestamps, full_time, start_time=0, ax=None, **kwargs):
    """Plot `timestamps` within `full_time` as indicator variables (ones)

    Parameters
    ----------
    timestamps : array
        timestamps for events to plot
    full_time : array
        array of all possible timestamps
    start_time : int, optional
        time to subtract off `timestamps` (use if start time in `timestamps` differs
        from start time in `full_time`), by default 0
    ax : [type], optional
        axis into which to plot, by default None, which opens a new figure
    """
    xlim = [full_time.min()/60, full_time.max()/60]
    # Get timing for each epoch
    calibration_time_index = np.in1d(full_time, timestamps-start_time)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(full_time / 60, calibration_time_index, **kwargs)
    ax.set_xlim(xlim)


def set_eye_axis_lims(ax, eye):
    if eye == 'right':
        ax.axis([0, 1, 0, 1])
    elif eye == 'left':
        ax.axis([1, 0, 1, 0])


def make_world_eye_axes(eye_left_ax=True, eye_right_ax=True, fig_scale=5):
    """Plot world w/ markers, eyes w/ pupils
    assumes 
    
    """
    # Set up figure & world axis
    nr_gs = 5
    nc_gs = 4
    mx_n = max(nr_gs, nc_gs)
    fig = plt.figure(figsize=(nc_gs / mx_n * fig_scale, nr_gs / mx_n * fig_scale))
    gs = gridspec.GridSpec(nr_gs, nc_gs, wspace = 0.3, hspace = 0.3)
    world_ax = fig.add_subplot(gs[:3,:])
    # Left
    if eye_left_ax:
        eye_left_ax = fig.add_subplot(gs[3:, 0:2])
        set_eye_axis_lims(eye_left_ax, 'left')
    # Right
    if eye_right_ax: 
        eye_right_ax = fig.add_subplot(gs[3:, 2:4])
        set_eye_axis_lims(eye_right_ax, 'right')
    return world_ax, eye_left_ax, eye_right_ax


def make_dot_overlay_animation(
    video_data,
    video_timestamps,
    *dot_data,
    downsampling_function=np.median,
    window=None,
    dot_widths=None,
    dot_colors=None,
    dot_labels=None,
    dot_markers=None,
    figsize=None,
    fps=60,
    accumulate=False,
    **kwargs
):
    """Make an animated plot of dots moving around on a video image

    Useful for showing estimated gaze positions, detected marker positions, etc. Multiple dots
    indicating multiple different quantites, each with different plot (color, marker type, width)
    can be plotted with this function.

    Parameters
    ----------
    video_data : array
        stack of video_data, (time, vdim, hdim, [c]), in a format showable by plt.imshow()
    video_timestamps : array
        [n_frames] timestamps for video frames; optional, see `dot_timestamps`
    dot_data : dict
        [n_dots, n_frames, xy]: locations of dots to plot, either in normalized (0-1 for
        both x and y) coordinates or in pixel coordinates (with pixel dimensions matching
        the size of the video)
    xx dot_timestamps : array
        [n_dots, n_dot_frames] timestamps for dots to plot; optional if a dot location is
        specified for each frame. However, if `dot_timestamps` do not match
        `video_timestamps`, dot_locations are resampled (simple block average) to match
        with video frames using these timestamps.
    
    dot_widths : scalar or list
        size(s) for dots to plot
    dot_colors : matplotlib colorspec (e.g. 'r' or [1, 0, 0]) or list of colorspecs
        colors of dots to plot. 
    dot_labels : string or list of strings
        label per dot (for legend) NOT YET IMPLEMENTED.
    dot_markers : string or list of strings
        marker type for each dot
    figsize : tuple
        Size of figure
    fps : scalar
        fps for resulting animation

    Notes
    -----
    Good tutorial, fancy extras: https://alexgude.com/blog/matplotlib-blitting-supernova/
    """
    from functools import partial

    def prep_input(x, n):
        if not isinstance(x, (list, tuple)):
            x = [x]
        if len(x) == 1:
            x = x * n
        return x
    n_dots = len(dot_data)
    #dot_locs = dot_locations.copy()
    if dot_markers is None:
        dot_markers = "o"

    # Shape
    extent = [0, 1, 1, 0]
    # interval is milliseconds; convert fps to milliseconds per frame
    interval = 1000 / fps
    # Setup
    n_frames, vdim, hdim = video_data.shape[:3]
    im_shape = (vdim, hdim)
    aspect_ratio = hdim / vdim
    if figsize is None:
        figsize = (5 * aspect_ratio, 5)
    # Match up timestamps
    dots_matched = []
    for this_dot in dot_data:
        # not all, because often we use e.g. first and last timestamps of calibration epoch to select video time,
        # so the video time will be 2 frames shorter...
        acceptable_wiggle_room = 5  # frames in dot timestamps that are not in video
        if (np.sum(np.in1d(this_dot['timestamp'], video_timestamps)) - len(this_dot['timestamp'])) < acceptable_wiggle_room:
            print("Timestamps already in video timestamps")
            dots_matched.append(this_dot)
        else:
            tmp = match_time_points(dict(
                timestamp=video_timestamps), this_dot, window=window, fn=downsampling_function)
            dots_matched.append(tmp)
    dots_matched = tuple(dots_matched)
    vframes = []
    for this_dot in dots_matched:
        tmp, = np.nonzero(np.in1d(video_timestamps, this_dot['timestamp']))
        print("n frames detected in video is:")
        print(len(tmp))
        vframes.append(tmp)
    # Plotting args
    #n_dots, n_dot_frames, _ = dot_locs_ds.shape
    dot_colors = prep_input(dot_colors, n_dots)
    dot_widths = prep_input(dot_widths, n_dots)
    dot_markers = prep_input(dot_markers, n_dots)
    dot_labels = prep_input(dot_labels, n_dots)
    # print(dot_markers)
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(video_data[0], extent=extent, cmap="gray", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    dots = []
    for this_dot, dc, dw, dm, dl in zip(dots_matched, dot_colors, dot_widths, dot_markers, dot_labels):
        tmp = plt.scatter(*this_dot['norm_pos'].T,
                          s=dw, c=dc, marker=dm, label=dl)
        dots.append(tmp)
    artists = (im,) + tuple(dots)
    plt.close(fig.number)
    # initialization function: plot the background of each frame

    def init_func(fig, ax, artists):
        for j, d in enumerate(dots):
            d.set_offsets(dots_matched[j]['norm_pos'][:1])
        im.set_array(np.zeros(im_shape))
        return artists

    # animation function. This is called sequentially
    def update_func(i, artists, dots_matched, vframes):

        artists[0].set_array(video_data[i])
        # Loop over dots
        for j, artist in enumerate(artists[1:]):
            if i in vframes[j]:
                dot_i = vframes[j].tolist().index(i)
                if accumulate:
                    _ = artist.set_offsets(dots_matched[j]['norm_pos'][:dot_i])
                else:
                    _ = artist.set_offsets(dots_matched[j]['norm_pos'][dot_i])
            else:
                if not accumulate:
                    _ = artist.set_offsets([-1, -1])
        return artists

    init = partial(init_func, fig=fig, ax=ax, artists=artists)
    update = partial(
        update_func, artists=artists, dots_matched=dots_matched, vframes=vframes
    )
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, func=update, init_func=init, frames=n_frames, interval=interval, blit=True
    )
    return anim


def show_ellipse(ellipse, img=None, ax=None, center_color='r', **kwargs):
    """Show opencv ellipse in matplotlib, optionally with image underlay

    Parameters
    ----------
    ellipse : dict
        dict of ellipse parameters derived from opencv, with fields:
        * center: tuple (x, y)
        * axes: tuple (x length, y length)
        * angle: scalar, in degrees
    img : array
        underlay image to display
    ax : matplotlib axis
        axis into which to plot ellipse
    kwargs : passed to matplotlib.patches.Ellipse
    """
    if ax is None:
        fig, ax = plt.subplots()
    ell = patches.Ellipse(
        ellipse["center"], *ellipse["axes"], angle=ellipse["angle"], **kwargs
    )
    if img is not None:
        ax.imshow(img, cmap="gray")
    ax.add_patch(ell)
    ax.scatter(ellipse["center"][0], ellipse["center"][1], color=center_color)


def colormap_2d(
    data0,
    data1,
    image_cmap=BuOr_2D,
    vmin0=None,
    vmax0=None,
    vmin1=None,
    vmax1=None,
    map_to_uint8=False,
):
    """Map values in two dimensions to color according to a 2D color map image

    Parameters
    ----------
    data0 : array (1d)
        First dimension of data to map
    data1 : array (1d)
        Second dimension of data to map
    image_cmap : array (3d)
        image of values to use for 2D color map

    """
    norm0 = colors.Normalize(vmin0, vmax0)
    norm1 = colors.Normalize(vmin1, vmax1)

    d0 = np.clip(norm0(data0), 0, 1)
    d1 = np.clip(1 - norm1(data1), 0, 1)
    dim0 = np.round(d0 * (image_cmap.shape[1] - 1))
    # Nans in data seemed to cause weird interaction with conversion to uint32
    dim0 = np.nan_to_num(dim0).astype(np.uint32)
    dim1 = np.round(d1 * (image_cmap.shape[0] - 1))
    dim1 = np.nan_to_num(dim1).astype(np.uint32)

    colored = image_cmap[dim1.ravel(), dim0.ravel()]
    # May be useful to map r, g, b, a values between 0 and 255
    # to avoid problems with diff plotting functions...?
    if map_to_uint8:
        colored = (colored * 255).astype(np.uint8)
    return colored


def show_dots(mk, start=0, end=10, size=0.25, fps=30, video='world_camera', accumulate=False, **kwargs):
    """Make an animation of detected markers as dots overlaid on video
    mk is a vedb_store class (MarkerDetection or PupilDetection so far)
    Set vdieo, size, frame rate according to what mk is!
    """
    mk.db_load()
    vtime, vdata = mk.session.load(video, time_idx=(start, end), size=size)
    anim = make_dot_overlay_animation(vdata,
                                      vtime + mk.session.start_time,
                                      mk.data,
                                      fps=fps,
                                      accumulate=accumulate,
                                      **kwargs,
                                      )
    return anim


def average_frames(session, times, to_load='world_camera'):
    """Load and average frames that occur at a specified list of `times`
    
    Parameters
    ----------
    ses : TYPE
        Description
    times : TYPE
        Description
    to_load : str, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    images = []
    for t0 in times:
        timestamp, frames = session.load(to_load, time_idx=[t0, t0 + 1])
        images.append(frames[0])
    return np.array(images).mean(0).astype(frames.dtype)


def show_average_frames(ses, times, n_images=4, ax=None, extent=(0, 1, 1, 0), mode='linear', to_load='world_camera'):
    """Summary
    
    Parameters
    ----------
    ses : TYPE
        Description
    times : TYPE
        Description
    n_images : int, optional
        Description
    ax : None, optional
        Description
    to_load : str, optional
        Description
    """
    if mode == 'linear':
        nn = len(times) // n_images
        if len(times) % nn == 0:
            nn -= 1
        time_samples = times[::nn][:n_images]
    else:
        time_samples = np.random.choice(times, (n_images,), replace=False)
    img = average_frames(ses, time_samples, to_load=to_load)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, extent=extent, aspect='auto')


def plot_error(err, 
    gaze=None, 
    ax=None, 
    cmap='inferno_r', 
    cmap_hist='gray_r',
    levels=(1, 2.5, 5, 10), 
    err_vmax=10, 
    err_vmin=0, 
    plot_image=True, 
    scatter_size=3,
    hist_bins_x=81, 
    hist_bins_y=61, 
    confidence_threshold=0.6):
    """_summary_

    Parameters
    ----------
    err : _type_
        _description_
    eye : str, optional
        _description_, by default 'left'
    gaze : _type_, optional
        _description_, by default None
    ax : _type_, optional
        _description_, by default None
    cmap : str, optional
        _description_, by default 'inferno_r'
    cmap_hist : str, optional
        _description_, by default 'gray_r'
    levels : tuple, optional
        _description_, by default (1, 2.5, 5, 10)
    err_vmax : int, optional
        _description_, by default 10
    err_vmin : int, optional
        _description_, by default 0
    plot_image : bool, optional
        _description_, by default True
    scatter_size : int, optional
        _description_, by default 3
    hist_bins_x : int, optional
        _description_, by default 81
    hist_bins_y : int, optional
        _description_, by default 61
    confidence_threshold : float, optional
        _description_, by default 0.6

    Returns
    -------
    _type_
        _description_
    """
    if ax is None:
        fig, ax = plt.subplots()
    out = []
    if gaze is not None:
        h_hst = gaze_hist(gaze, 
            confidence_threshold=confidence_threshold, 
            cmap=cmap_hist, 
            hist_bins_x=hist_bins_x, 
            hist_bins_y=hist_bins_y,
            ax=ax)
        out.append(h_hst)
    else:
        ci = None
        
    if plot_image:
        im_kw = dict(extent=[0, 1, 1, 0], 
            aspect='auto', 
            cmap='inferno_r', 
            vmin=err_vmin, 
            vmax=err_vmax, 
            alpha=0.3)
        h_im = ax.imshow(err['gaze_err_image'], **im_kw)   
        out.append(h_im)

    contour_kw = dict(levels=levels, cmap=cmap, vmin=err_vmin, vmax=err_vmax)
    h_contour = ax.contour(err['xgrid'], err['ygrid'], err['gaze_err_image'], **contour_kw)
    out.append(h_contour)

    scatter_kw = dict(cmap=cmap, vmin=err_vmin, vmax=err_vmax, s=scatter_size)
    h_scatter = ax.scatter(*err['marker'].T, c=err['gaze_err'], **scatter_kw)
    out.append(h_scatter)
    ax.axis([0, 1, 1, 0])
    return out


def plot_error_markers(markers, gaze,
                                confidence_threshold=0,
                                ses=None,
                                ax=None,
                                n_images_to_average=4,
                                do_gradients=True,
                                # Blue # (0.95, 0.85, 0) # Yellow
                                left_eye_color=(0.0, 0.0, 0.9),
                                # Yellow # (1.00, 0.50, 0) # Orange
                                right_eye_color=(0.95, 0.85, 0),
                                marker_color=(1.0, 0, 0),  # Red
                                ):
    """Plot validation markers and gaze at relevant timepoints
    
    """
    left_eye_cmap = (marker_color, left_eye_color)
    right_eye_cmap = (marker_color, right_eye_color)

    vt = markers['timestamp']
    vp = markers['norm_pos']
    gaze_left_markers, gaze_right_markers = match_time_points(
        markers, gaze['left'], gaze['right'])
    gl = gaze_left_markers['position']
    gl_ci = gaze_left_markers['confidence'] > confidence_threshold
    gr = gaze_right_markers['position']
    gr_ci = gaze_right_markers['confidence'] > confidence_threshold
    if (gl_ci.sum() < 10) or (gr_ci.sum() < 10):
        print("Insufficient number of points retained after assessing pupil confidence for v epoch %d" % j)
        return
    if ax is None:
        fig, ax = plt.subplots()
    # Visualization of markersidation epochs w/ gaze
    if ses is not None:
        show_average_frames(ses, vt, n_images=n_images_to_average, ax=ax)
    if do_gradients:
        plot_utils.gradient_lines(
            vp[gl_ci], gl[gl_ci], cmap=left_eye_cmap, ax=ax)
        plot_utils.gradient_lines(
            vp[gr_ci], gr[gr_ci], cmap=right_eye_cmap, ax=ax)
    ax.set_title('%.1f minutes' % (vt[0]/60))

def plot_error_interpolation(marker, gaze_err, xgrid, ygrid, gaze_err_image, vmin=0, vmax=None, ax=None,
                             cmap='viridis_r', azimuth=60, elevation=30, **kwargs):
    """Plot error surface for gaze in 3D
    
    Parameters
    ----------
    marker : TYPE
        Description
    gaze_err : TYPE
        Description
    xgrid : TYPE
        Description
    ygrid : TYPE
        Description
    gaze_err_image : TYPE
        Description
    vmin : int, optional
        Description
    vmax : None, optional
        Description
    ax : None, optional
        Description
    cmap : str, optional
        Description
    azimuth : int, optional
        Description
    elevation : int, optional
        Description
    """
    if vmax is None:
        vmax = np.nanmax(gaze_err) * 1.1
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.scatter3D(marker[:, 0], marker[:, 1], gaze_err, c=gaze_err,
                 cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot_surface(xgrid, ygrid, gaze_err_image, vmin=vmin,
                    vmax=vmax, alpha=0.66)  # cmap=cmap,
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlim([0.2, 0.8])
    ax.set_ylim([0.2, 0.8])
    ax.set_zlim([0, vmax])
    ax.set_xlabel("X (horizontal)", fontname='Helvetica')
    ax.set_ylabel("Y (vertical)", fontname='Helvetica')
    ax.set_zlabel("Error (degrees)", fontname='Helvetica')
    plot_utils.set_ax_fontsz(ax, lab=12, tk=11, name='Helvetica')





def plot_marker_epoch(ses, marker, eye_left, eye_right,
                      show_image=True, marker_field='norm_pos', sc=2,
                      keep_index=None, confidence_threshold=0, point_size=1,
                      marker_color=((0.2, 0.2, 0.2),), n_images=3,
                      world_ax=None, eye_left_ax=None, eye_right_ax=None, cmap=cluster_cmap):
    """Plot world w/ markers, eyes w/ pupils
    assumes 
    
    Parameters
    ----------
    ses : TYPE
        Description
    marker : TYPE
        Description
    eye_left : TYPE
        Description
    eye_right : TYPE
        Description
    groups : None, optional
        Description
    marker_field : str, optional
        Description
    sc : int, optional
        Description
    keep_index : None, optional
        Description
    """

    n_tp = len(marker['timestamp'])
    if keep_index is None:
        keep_index = np.ones((n_tp, )) > 0
    if eye_left is not None:
        if len(eye_left['norm_pos']) > n_tp:
            eye_left = match_time_points(marker, eye_left, session=ses)
        lc = eye_left['confidence']
        keep_index &= (lc > confidence_threshold)
    if eye_right is not None:
        if len(eye_right['norm_pos']) > n_tp:
            eye_right = match_time_points(marker, eye_right, session=ses)
        rc = eye_right['confidence']
        keep_index &= (rc > confidence_threshold)
    if 'marker_cluster_index' not in marker:
        c = marker_color
        dot_colors = None
        groups = None
    else:
        groups = marker['marker_cluster_index']
        c = groups[keep_index]
        if len(np.unique(c)) == 1:
            # One group only
            c = marker_color
            dot_colors = None
        else:
            dot_colors = cmap
    if 'size' in marker:
        kw = dict(s=marker['size'][keep_index])
    else:
        kw = {}
    if point_size is not None:
        kw.update(s=point_size)
    nr_gs = 5
    nc_gs = 4
    if world_ax is None:
        # Set up figure & world axis
        fig = plt.figure(figsize=(nc_gs * sc, nr_gs * sc))
        gs = gridspec.GridSpec(nr_gs, nc_gs, wspace=0.3, hspace=0.3)
        world_ax = fig.add_subplot(gs[:3, :])
    world_ax.scatter(*marker[marker_field][keep_index,
                     :].T, c=c, cmap=dot_colors, **kw)
    world_ax.axis([0, 1, 1, 0])
    if show_image:
        show_average_frames(
            ses, marker['timestamp'][keep_index] - ses.start_time, ax=world_ax, n_images=n_images)
    # Left
    if eye_left_ax is not False:
        if eye_left_ax is None:
            eye_left_ax = fig.add_subplot(gs[3:, 0:2])
        eye_left_ax.scatter(
            *eye_left['norm_pos'][keep_index, :].T, c=c, cmap=dot_colors, **kw)
        if show_image:
            show_average_frames(ses, marker['timestamp'][keep_index] - ses.start_time,
                                ax=eye_left_ax, to_load='eye_left', n_images=n_images)
        eye_left_ax.axis([1, 0, 1, 0])
    # Right
    if eye_right_ax is not False:
        if eye_right_ax is None:
            eye_right_ax = fig.add_subplot(gs[3:, 2:4])
        eye_right_ax.scatter(
            *eye_right['norm_pos'][keep_index, :].T, c=c, cmap=dot_colors, **kw)
        if show_image:
            show_average_frames(ses, marker['timestamp'][keep_index] - ses.start_time,
                                ax=eye_right_ax, to_load='eye_right', n_images=n_images)
        eye_right_ax.axis([0, 1, 0, 1])
    return world_ax, eye_left_ax, eye_right_ax
    

def plot_epochs_db(ses,
                calibration_epochs,
                validation_epochs,
                calibration_points_all=None,
                validation_points_all=None,
                gaze=None,
                do_gradients=True,
                error=None,
                calibration=None,
                min_confidence_threshold=0,
                ):

    n_epochs = [len(calibration_epochs), len(validation_epochs)]
    # Set up plot
    # Colors
    calib_color = (0.0, 0.8, 0.0)  # Green # (0.0, 0.0, 1.0) # Blue
    val_color = (0.9, 0.0, 0.0)  # Red # (0.0, 0.0, 1.0) # Blue
    left_eye_col = (0.0, 0.0, 0.9)  # Blue # (0.95, 0.85, 0) # Yellow
    right_eye_col = (0.95, 0.85, 0)  # Yellow # (1.00, 0.50, 0) # Orange
    #left_eye_cmap = LinearSegmentedColormap('left', [(1.0, 0, 0), left_eye_col])
    left_eye_cmap = ((1.0, 0, 0), left_eye_col)
    #right_eye_cmap = LinearSegmentedColormap('left', [(1.0, 0, 0), right_eye_col])
    right_eye_cmap = ((1.0, 0, 0), right_eye_col)
    # For plots of marker detections & eyes
    n_images_to_average = 3
    # Grid
    rows_per_epoch = 5  # 3 for video, 2 for eyes / accuracy
    n_rows_top = 5  # validation / calibration; pupil confidence; pupil skewness; pupil images
    n_rows_bottom = 0
    columns = 4 * 2
    ne = np.max(n_epochs)
    nr = rows_per_epoch * ne
    nr_gs = nr + n_rows_top + n_rows_bottom
    nc_gs = columns
    plot_height_inches = sc = 2
    aspect_ratio = 4 / 3
    # 1 - top_row_height / (top_row_height + sc * nr) * 7/8
    pct_top = (nr + 0.5) / nr

    fig = plt.figure(figsize=(nc_gs * sc, nr_gs * sc))
    gs = gridspec.GridSpec(nr_gs, nc_gs, wspace=0.3, hspace=0.6)

    # Plot timing of calibration, validation within session
    ax0a = fig.add_subplot(gs[0, :])
    if calibration_points_all is not None:
        plot_timestamps(
            calibration_points_all.timestamp, ses.world_time, color='gray', alpha=0.3, ax=ax0a,
            label='Calibration (unfilt.)')
    plot_timestamps(np.hstack([ce.timestamp for ce in calibration_epochs]), ses.world_time,
                    color='b', alpha=0.6, ax=ax0a, label='Calibration')
    if validation_points_all is not None:
        plot_timestamps(
            validation_points_all.timestamp, ses.world_time, color='r', alpha=0.3, ax=ax0a)
    if len(validation_epochs) > 0:
        plot_timestamps(np.hstack([ve.timestamp for ve in validation_epochs]), ses.world_time,
                        color='r', alpha=0.6, ax=ax0a, label='Validation')
    ax0a.set_xlabel('Time (minutes)', fontname='Helvetica', fontsize=12)
    ax0a.legend()
    plot_utils.open_axes(ax0a)

    # Plot gaze (i.e. pupil) confidence
    ax0b = fig.add_subplot(gs[1, :])
    value = 0
    xlim = [ses.world_time.min()/60, ses.world_time.max()/60]
    if gaze is not None:
        vertical_extent = dict(left=(1, 0.5),
            right=(0.5, 0),
            both=(1, 0), # unclear if this will work as intended...
            )
        for eye in ['left', 'right']:
            gc = gaze[eye].data['confidence'].copy()[np.newaxis, :]
            gc[gc < min_confidence_threshold] = value
            imh = ax0b.imshow(gc, cmap='inferno', vmin=0,
                vmax=1, aspect='auto', extent=[*xlim, *vertical_extent[eye]])
            ax0b.axis([*xlim, 1, 0])
            ax0b.set_yticks([0.25, 0.75])
            ax0b.set_yticklabels(['R', 'L'])

    # Plot pupil images
    ax0c = fig.add_subplot(gs[2, :])
    n_samples = 20
    start_time = 20
    lht = 0.075
    rht = 0.225
    gap = np.int(np.floor((np.ptp(ses.world_time) - start_time) / n_samples))
    sample_times = np.arange(start_time, gap*n_samples+1, gap)
    sample_times = np.array(
        [s for s in sample_times if np.min(np.abs(ses.world_time-s)) < 1/20])
    el = [ses.load('eye_left', time_idx=(t, t+1), color='gray')[1][0]
          for t in sample_times]
    er = [ses.load('eye_right', time_idx=(t, t+1), color='gray')[1][0]
          for t in sample_times]
    # Flip over right eye for display
    er = [x[::-1, ::-1] for x in er]
    kw = dict(cmap='gray', ax=ax0c, ylim=[0.0, 0.3], imsz=0.5, xlim=xlim)
    plot_utils.plot_ims(sample_times/60, lht *
                        np.ones_like(sample_times), el, **kw)
    plot_utils.plot_ims(sample_times/60, rht *
                        np.ones_like(sample_times), er, **kw)
    ax0c.set_yticks([lht, rht])
    ax0c.set_yticklabels(['L', 'R'])

    # World camera images
    ax0d = fig.add_subplot(gs[3:5, :])
    sample_times_world = sample_times[::4]
    sample_times = np.array(
        [s for s in sample_times if np.min(np.abs(ses.world_time-s)) < 1/20])
    world_cam = [ses.load('world_camera', time_idx=(t, t+1))[1][0]
                 for t in sample_times_world]
    kw = dict(ax=ax0d, ylim=[0.0, 0.3], imsz=1.0, xlim=xlim)
    plot_utils.plot_ims(sample_times_world / 60, (lht+rht) /
                        2 * np.ones_like(sample_times_world), world_cam, **kw)
    ax0d.set_yticks([])

    # Plot calibration
    j = 0
    # Image & detected calibration locations
    st = j * rows_per_epoch + n_rows_top
    ax_ = fig.add_subplot(gs[st:st+3, :4])
    # 3 is height in rows of calibration image
    st_e = j * rows_per_epoch + n_rows_top + 3 # st_e = start row for eyes
    ax_e = dict(
        left = fig.add_subplot(gs[st_e:st_e+2, :2]),
        right = fig.add_subplot(gs[st_e:st_e+2, 2:4])
        )
    ecolor = dict(
        left = (0.0, 0.5, 1.0, 0.3),
        right = (1.0, 0.5, 0.0, 0.3)
        )
    ee = 0
    for eye, calib in calibration.items():
        ki = calib.calibration.pupil_arrays['confidence'] > min_confidence_threshold
        calib.calibration.show_calibration(show_data=True, eye=eye,
            ax_world=ax_, ax_eye=ax_e[eye], color=ecolor[eye])
        # show eye image
        show_average_frames(ses, calib.calibration.pupil_arrays['timestamp'] - ses.start_time, 
            n_images=3, ax=ax_e, to_load='eye_'+ eye)
        set_eye_axis_lims(ax_e[eye], eye)
        # Show world image
        if ee == 0:
            show_average_frames(
                ses, calib.marker_detection.timestamp,
                n_images=1, ax=ax_e, to_load='world_camera')
            ax_.axis([0, 1, 1, 0])
        ee += 1

    for j, ve in enumerate(validation_epochs):
        #groups_val, val, pupil_left_val, pupil_right_val = ve
        vt = ve.timestamp
        vp = ve.data['norm_pos']
        #plv = pupil_left_val['norm_pos']
        #prv = pupil_right_val['norm_pos']
        #gaze_left_val, gaze_right_val = match_time_points(
        #    val, gaze['left'], gaze['right'])
        #gl = gaze_left_val['norm_pos']
        #gl_ci = gaze_left_val['confidence'] > min_confidence_threshold
        #gr = gaze_right_val['norm_pos']
        #gr_ci = gaze_right_val['confidence'] > min_confidence_threshold

        st = j * rows_per_epoch + n_rows_top
        ax_ = fig.add_subplot(gs[st:st+3, 4:8])
        #if (gl_ci.sum() < 10) or (gr_ci.sum() < 10):
        #    print(
        #        "Insufficient number of points retained after assessing pupil confidence for v epoch %d" % j)
        #    continue
        # Visualization of validation epochs w/ gaze
        show_average_frames(ses, vt, n_images=n_images_to_average, ax=ax_)
        #if do_gradients:
        #    plot_utils.gradient_lines(
        #        vp[gl_ci], gl[gl_ci], cmap=left_eye_cmap, ax=ax_)
        #    plot_utils.gradient_lines(
        #        vp[gr_ci], gr[gr_ci], cmap=right_eye_cmap, ax=ax_)
        ax_.plot(*vp.T, 'r.')
        #ax_.plot(*gl[gl_ci].T, '.', color=left_eye_col)
        #ax_.plot(*gr[gr_ci].T, '.', color=right_eye_col)
        ax_.axis([0, 1, 1, 0])
        ax_.axis('off')
        # Gaze accuracy plots for each validation epoch
        # 3 is height in rows of calibration image
        st = j * rows_per_epoch + n_rows_top + 3
        ax_L = fig.add_subplot(gs[st:st+2, 4:6])
        ax_R = fig.add_subplot(gs[st:st+2, 6:8])
        if error is not None:
            if error[j] is None:
                continue
            err_kw = dict(cmap='inferno_r', cmap_hist='gray_r',
                          levels=(1, 2.5, 5, 10), err_vmax=10, err_vmin=0,
                          plot_image=True, scatter_size=3, hist_bins_x=81,
                          hist_bins_y=61, min_confidence_threshold=min_confidence_threshold)
            # Left eye error
            handles = plot_error(error[j], eye='left', gaze=gaze['left'], ax=ax_L,
                                 **err_kw)
            plot_utils.set_axis_lines(ax_L, color=left_eye_col, lw=2)
            #[h_hst, h_im, h_contour, h_scatter]
            hst_im = handles[0]
            fig.colorbar(hst_im, orientation='horizontal',
                         ax=ax_L, aspect=15, fraction=0.075)
            # Rigth eye error
            handles = plot_error(error[j], eye='right', gaze=gaze['right'], ax=ax_R,
                                 **err_kw)
            plot_utils.set_axis_lines(ax_L, color=left_eye_col, lw=2)
            err_im = handles[1]
            try:
                fig.colorbar(err_im, orientation='horizontal',
                             ax=ax_R, aspect=15, fraction=0.075)
            except:
                print("session error is high, colorbar for error contours fails")
            plot_utils.set_axis_lines(ax_R, color=right_eye_col, lw=2)

        for axx in [ax_L, ax_R]:
            axx.axis([0, 1, 1, 0])
            axx.set_xticks([0, 0.5, 1])
            axx.set_xticklabels([-67.5, 0, 67.5])
            axx.set_yticks([0, 0.5, 1])
            axx.set_yticklabels([55.5, 0, -55.5])
    return fig


def run_session_qc(session,
                   min_confidence_threshold=0.6,
                   pupil_tag='plab_default',
                   marker_tag_cal='circles_halfres',
                   marker_tag_val='checkerboard_halfres',
                   marker_epoch_tag_cal='cluster_default',
                   marker_epoch_tag_val='basic_split',
                   calibration_tag='monocular_pl_default',
                   calibration_epoch=0,
                   gaze_tag='default_mapping',
                   error_tag='tps_default',
                   sname=None,
                   dpi=300,
                   re_assign_markers=False,
                   dbi=None,
                   ):
    ### --- Load all session elements --- ###
    session.db_load()
    # All calibration markers
    # try:
    #     mk_cal = dbi.query(1, type='MarkerDetection',
    #         epoch='all',
    #         session=session._id,
    #         tag=marker_tag_cal,
    #         )
    # except:
    #     mk_cal = None
    # # All epochs for calibration
    # try: 
    #     epoch_cal = dbi.query(type='MarkerDetection',
    #         session=session._id,
    #         tag=f'{marker_tag_cal}-{marker_epoch_tag_cal}')
    #     epoch_cal = sorted(epoch_cal, key=lambda x: x.epoch)
    # except:
    #     epoch_cal = None
    # # All epochs for validation
    # try: 
    #     epoch_val = dbi.query(type='MarkerDetection',
    #                           session=session._id,
    #                           tag=f'{marker_tag_val}-{marker_epoch_tag_val}')
    #     epoch_val = sorted(epoch_val, key=lambda x: x.epoch)
    #     # Re-assign epochs?
    #     if re_assign_markers:
    #         if n_cal > 1:
    #             epoch_val.extend(epoch_cal[1:])
    #             epoch_cal = epoch_cal[:1]
    # except:
    #     epoch_val = None
    # # Calibration
    # calibration = {}
    # ctag = f'{marker_tag_val}-{marker_epoch_tag_val}'
    # if 'monocular' in calibration_tag:
    #     eyes = ['left', 'right']
    # else:
    #     eyes = ['both']
    # try:
    #     for eye in eyes:
    #         tmp = dbi.query(1, type='Calibration', 
    #                         tag=calibration_tag, 
    #                         marker_detection=epoch_cal[calibration_epoch]._id,
    #                         eye=eye, 
    #                         epoch=calibration_epoch,
    #                         session=session._id,
    #                         )
    #         tmp.load()
    #         calibration[eye] = tmp
    # except:
    #     raise
    #     calibration = None
    # # Estimated gaze
    # gaze = {}
    # try:
    #     for eye in eyes:
    #         gaze[eye] = dbi.query(1, 
    #             type='Gaze',
    #             session=session._id,
    #             tag=gaze_tag,
    #             calibration=calibration[eye],
    #             eye=eye,
    #             pupil_detections=pupil_tag,)
    # except:
    #     gaze = None
    # # Error
    # try:
    #     err = []
    #     for ve in epoch_val:
    #         for eye in eyes:
    #             tmp = dbi.query(1, type='GazeError', 
    #                 marker_detection=ve,
    #                 gaze=gaze[eye]._id,
    #                 )
    #             err.append(tmp)
    # except:
    #     if len(err) < 1:
    #         err = None
    # Find epochs
    n_cal = len(epoch_cal)
    print(">>> Found %d calibration epochs" % (n_cal))
    n_val = len(epoch_val)
    print(">>> Found %d validation epochs" % (n_val))

    # Plot all
    fig = plot_epochs_db(session,
                      epoch_cal,
                      epoch_val,
                      calibration_points_all=mk_cal,
                      validation_points_all=None,
                      calibration=calibration,
                      gaze=gaze,
                      do_gradients=True,
                      error=err,
                      min_confidence_threshold=min_confidence_threshold,
                      )
    if sname is not None:
        fig.savefig(sname, dpi=dpi)


def gaze_rect(gaze_position, hdim, vdim, ax=None, linewidth=1, edgecolor='r', **kwargs):
    if ax is None:
        ax = plt.gca()
    # Create a Rectangle patch
    gp = gaze_position - np.array([hdim, vdim]) / 2
    rect = patches.Rectangle(gp, hdim, vdim,
                             facecolor='none',
                             edgecolor=edgecolor,
                             linewidth=linewidth,
                             **kwargs)
    # Add the patch to the Axes
    rh = ax.add_patch(rect)
    return rh

# Making gaze centered videos
def _load_gaze_plot_elements(pipeline_elements,
                             frame_idx=None,
                             time_idx=None,
                             world_size=(600, 800),
                             eye_size=(200, 200),
                             crop_size=(384, 384),
                             crop_size_resize=None,
                             tdelta=0.2,  # to help assure range of eye times is wider than range of world times for interpolation
                             ):
    """Example: 
    world_camera, eye_left_ds, eye_right_ds, \
        pupil_left_matched, pupil_right_matched, \
        gaze_matched, gc_video = _load_gaze_plot_elements(session, time_idx=(60*8,60*8+20))

    Parameters
    ----------
    session : _type_
        _description_
    frame_idx : _type_, optional
        _description_, by default None
    time_idx : _type_, optional
        _description_, by default None
    world_size : tuple, optional
        _description_, by default (600, 800)
    eye_size : tuple, optional
        _description_, by default (200, 200)
    eye : str, optional
        _description_, by default 'left'
    pupil_tag : str, optional
        _description_, by default 'plab'
    gaze_tag : str, optional
        _description_, by default 'basic_mapping'
    calibration_tag : str, optional
        _description_, by default 'monocular_pl_default'
    marker_type : str, optional
        _description_, by default 'concentric_circle'
    marker_epoch : int, optional
        _description_, by default 0
    marker_epoch_tag : str, optional
        _description_, by default 'cluster_default'

    Returns
    -------
    _type_
        _description_
    """
    # Load world camera
    if world_size is None:
        print("Skipping world load...")
        world_time = pipeline_elements['session'].world_time.copy()
        if frame_idx is not None:
            world_time = world_time[frame_idx[0]:frame_idx[1]]
        elif time_idx is not None:
            world_time = world_time[(world_time > time_idx[0]) & (world_time < time_idx[1])]
        world_camera = None
    else:
        print('Loading world...')
        world_time, world_camera = pipeline_elements['session'].load(
        'world_camera', frame_idx=frame_idx, time_idx=time_idx, size=world_size)
    
    # Load & temporally downsample eye videos
    if eye_size is None:
        eye_left_ds = None
        eye_right_ds = None
    else:
        eye_left_time, eye_left_vid = pipeline_elements['session'].load('eye_left', time_idx=[
                                            world_time[0]-tdelta, world_time[-1]+tdelta], size=eye_size)
        eye_right_time, eye_right_vid = pipeline_elements['session'].load('eye_right', time_idx=[
                                                world_time[0]-tdelta, world_time[-1]+tdelta], size=eye_size)
        # Potentially revisit downsampling to make this just skip frames until nearest temporal match, so it can be 
        # performed frame-by-frame rather than loaded in blocks...? 
        downsampler_left = interpolate.interp1d(
            eye_left_time, eye_left_vid, assume_sorted=True, kind='nearest', axis=0)
        eye_left_ds = downsampler_left(world_time).astype(np.uint8)
        downsampler_right = interpolate.interp1d(
            eye_right_time, eye_right_vid, assume_sorted=True, kind='nearest', axis=0)
        eye_right_ds = downsampler_right(world_time).astype(np.uint8)

    # Load pupil estimation
    pupil_left = pipeline_elements['pupil']['left'].data
    pupil_right = pipeline_elements['pupil']['right'].data
    pupil_left_matched, pupil_right_matched = match_time_points(dict(timestamp=world_time + pipeline_elements['session'].start_time),
                                                                pupil_left,
                                                                pupil_right,
                                                                )
    # Load gaze estimation
    gaze = {}
    gaze_matched = {}
    for k in pipeline_elements['gaze'].keys():
        if pipeline_elements['gaze'][k] is None:
            gaze[k] = None
            gaze_matched[k] = None
        else:
            gaze[k] = pipeline_elements['gaze'][k].data
            gaze_matched[k] = match_time_points(
                dict(timestamp=world_time + pipeline_elements['session'].start_time), gaze[k])
            # Handle gaze failures here? Not for now...
            #gg = gaze_matched[k]['norm_pos']
            #gaze_matched[k]['norm_pos'][(gg >= 1) | (gg <= 0)] = np.nan
    gc_video = {}
    for k in pipeline_elements['gaze'].keys():
        if crop_size is None:
            gc_video[k] = None
        else:
            gc_video[k] = pipeline_elements['session'].load('world_camera', 
                center=gaze_matched[k]['norm_pos'], 
                crop_size=crop_size,
                size=crop_size_resize,
                frame_idx=frame_idx,
                time_idx=time_idx)

    return world_camera, eye_left_ds, eye_right_ds, pupil_left_matched, pupil_right_matched, gaze_matched, gc_video


def load_pipeline_elements(session,
                           pupil_tag='plab_default',
                           calibration_tag='monocular_tps_default',
                           cal_marker_tag='circles_halfres',
                           cal_marker_epoch_tag='cluster_default',
                           cal_marker_epoch=0,
                           val_marker_tag='checkerboard_halfres',
                           val_marker_epoch_tag='basic_split',
                           gaze_tag='default_mapper',
                           error_tag='smooth_tps_default',
                           dbi=None,
                           is_verbose=True,
                           ):
    if dbi is None:
        dbi = session.dbi
    verbosity = copy.copy(dbi.is_verbose)
    dbi.is_verbose = is_verbose > 1

    if 'monocular' in calibration_tag:
        eyes = ['left', 'right']
        outputs = dict(pupil=dict(left=[], right=[]),
                       calibration=dict(left=[], right=[]),
                       gaze=dict(left=[], right=[]),
                       error=dict(left=[], right=[]),
                       )
    else:
        eyes = ['both']
        outputs = dict(pupil=dict(both=[]),
                       calibration=dict(both=[]),
                       gaze=dict(both=[]),
                       error=dict(both=[]),
                       )
    outputs['session'] = session
    outputs.update(dict(calibration_marker_all=[],
                        validation_marker_all=[],
                        calibration_marker_filtered=[],
                        validation_marker_filtered=[],
                        )
                   )

    try:
        print("> Searching for calibration markers...")
        outputs['calibration_marker_all'] = dbi.query(
            1, type='MarkerDetection', tag=cal_marker_tag, epoch='all', session=session._id)
        print(">> FOUND it")
    except:
        print('>> NOT found')
        outputs['calibration_marker_all'] = None
    try:
        print("> Searching for validation markers...")
        outputs['validation_marker_all'] = dbi.query(
            1, type='MarkerDetection', tag=val_marker_tag, epoch='all', session=session._id)
        print(">> FOUND it")
    except:
        print('>> NOT found')
        outputs['validation_marker_all'] = None
    try:
        print("> Searching for filtered calibration markers...")
        cfiltered_tag = '-'.join([cal_marker_tag, cal_marker_epoch_tag])
        outputs['calibration_marker_filtered'] = dbi.query(
            1, type='MarkerDetection', tag=cfiltered_tag, epoch=cal_marker_epoch, session=session._id)
        print(">> FOUND it")
    except:
        print('>> NOT found')
        outputs['calibration_marker_filtered'] = None
    try:
        print("> Searching for filtered validation markers...")
        vfiltered_tag = '-'.join([val_marker_tag, val_marker_epoch_tag])
        tmp = dbi.query(type='MarkerDetection',
                        tag=vfiltered_tag, session=session._id)
        tmp = sorted(tmp, key=lambda x: x.epoch)
        outputs['validation_marker_filtered'] = tmp
        print(">> FOUND %d" % (len(tmp)))
    except:
        outputs['validation_marker_filtered'] = None

    # Quick way: Find error, derive everything else from that
    all_tags = [pupil_tag, cal_marker_tag, cal_marker_epoch_tag,
                calibration_tag, gaze_tag, val_marker_tag, val_marker_epoch_tag, error_tag]
    # Skip any steps not provided? Likely to cause bugs below
    #all_tags = [x for x in all_tags if x is not None]
    err_tag = '-'.join(all_tags)
    try:
        print("> Searching for error...")
        error_found = True
        err = dict(
            left=dbi.query(1, tag=err_tag, eye='left', session=session._id),
            right=dbi.query(1, tag=err_tag, eye='right', session=session._id),
        )
        print(">> FOUND error...")
    except:
        print(">> NO error found...")
        error_found = False

    for eye in eyes:
        if error_found:
            # Use only epoch 0 error for most of these
            err[eye].db_load()
            outputs['pupil'][eye] = err[eye].gaze.pupil_detection
            if isinstance(outputs['pupil'][eye], list):
                print('>>>', len(outputs['pupil'][eye]), 'pupil detections for %s eye found'%eye)
                outputs['pupil'][eye] = err[eye].gaze.pupil_detection[0]
            outputs['calibration'][eye] = err[eye].gaze.calibration
            outputs['gaze'][eye] = err[eye].gaze
            outputs['error'][eye] = err[eye]
        else:
            try:
                print("> Searching for %s pupil..." % eye)
                outputs['pupil'][eye] = dbi.query(
                    1, type='PupilDetection', tag=pupil_tag, eye=eye, session=session._id)
                print(">> FOUND %s pupil..." % eye)
            except:
                print('>> NOT found')
                outputs['pupil'][eye] = None
            try:
                print("> Searching for %s calibration..." % eye)
                calib_tag_full = '-'.join(all_tags[:4])
                outputs['calibration'][eye] = dbi.query(
                    1, type='Calibration', tag=calib_tag_full, eye=eye, epoch=cal_marker_epoch, session=session._id)
                print(">> FOUND %s calibration..." % eye)
            except:
                print('>> NOT found')
                outputs['calibration'][eye] = None
            try:
                print("> Searching for %s gaze..." % eye)
                gaze_tag_full = '-'.join(all_tags[:5])
                outputs['gaze'][eye] = dbi.query(
                    1, type='Gaze', tag=gaze_tag_full, eye=eye, session=session._id)
                print(">> FOUND %s gaze..." % eye)
            except:
                print('>> NOT found')
                outputs['gaze'][eye] = None
            outputs['error'][eye] = None

    dbi.is_verbose = verbosity

    return outputs
    

def make_gaze_animation(session,
                        time_idx=None,
                        frame_idx=None,
                        crop_size=(384, 384),
                        fps=30,
                        hspace=0.1,
                        wspace=None,
                        eye='left',
                        **pipeline_kw):
    """Make radical gaze animation"""
    if not isinstance(session, (list, tuple)):
        # DELETE ME here for debugging convenience
        pipeline_elements = load_pipeline_elements(
            session, **pipeline_kw, is_verbose=False)
        world, eye_left_ds, eye_right_ds, \
            pupil_left_matched, pupil_right_matched, \
            gaze_matched, gc_video = _load_gaze_plot_elements(
                pipeline_elements, time_idx=time_idx, frame_idx=frame_idx, crop_size=crop_size)
    else:
        world, eye_left_ds, eye_right_ds, \
            pupil_left_matched, pupil_right_matched, \
            gaze_matched, gc_video = session  # _load_gaze_plot_elements(pipeline_elements, time_idx=time_idx, frame_idx=frame_idx)
    n_frames = world.shape[0]
    frame = world[0]
    rect_width = crop_size[0] / frame.shape[1]
    rect_height = crop_size[1] / frame.shape[0]

    fig = plt.figure(figsize=(8, 8 * 13.5/12))  # * 14 / 12))
    gs = gridspec.GridSpec(2, 3, figure=fig,  hspace=hspace, wspace=wspace,
                  height_ratios=[1, 2], width_ratios=[1, 1, 1])
    ax_eye_left = fig.add_subplot(gs[0, 0])
    ax_eye_right = fig.add_subplot(gs[0, 1])
    ax_gc = fig.add_subplot(gs[0, 2])
    ax_vid = fig.add_subplot(gs[1, :])
    ax_vid.axis([0, 1, 1, 0])
    ax_vid.set_xticks([])  # To visual field size?
    ax_vid.set_yticks([])

    # Initialize all plots
    gaze_rect_h = gaze_rect(
        gaze_matched[eye]['norm_pos'][0], rect_width, rect_height, ax=ax_vid, linewidth=3)
    gaze_dot_h = ax_vid.scatter(*gaze_matched[eye]['norm_pos'][0], c='red')
    world_h = ax_vid.imshow(world[0], extent=[0, 1, 1, 0], aspect='auto')
    eye_left_h = ax_eye_left.imshow(eye_left_ds[0], extent=[0, 1, 1, 0])
    pupil_left_h = ax_eye_left.scatter(
        *pupil_left_matched['norm_pos'][0], c='red')
    eye_right_h = ax_eye_right.imshow(eye_right_ds[0], extent=[0, 1, 1, 0])
    gc_h = ax_gc.imshow(gc_video[eye][1][0], extent=[0, 1, 1, 0])

    ax_eye_left.axis([1, 0, 1, 0])
    ax_eye_left.set_xticks([])
    ax_eye_left.set_yticks([])
    ax_eye_right.axis([0, 1, 0, 1])
    ax_eye_right.set_xticks([])
    ax_eye_right.set_yticks([])

    ax_gc.set_xticks([0, 0.5, 1])
    ax_gc.set_xticklabels([-17, 0, 17])
    ax_gc.set_yticks([0, 0.5, 1])
    ax_gc.set_yticklabels([-17, 0, 17])
    ax_gc.grid('on', linestyle=':', color=(0.95, 0.85, 0))
    plt.close(fig)

    def init():
        gaze_rect_h.set_xy(
            [0.5, 0.5] - np.array([rect_width/2, rect_height/2]))
        gaze_dot_h.set_offsets([0.5, 0.5])
        pupil_left_h.set_offsets([0.5, 0.5])
        world_h.set_array(np.zeros_like(frame))
        eye_left_h.set_array(np.zeros_like(eye_left_ds[0]))
        eye_right_h.set_array(np.zeros_like(eye_right_ds[0]))
        gc_h.set_array(np.zeros_like(gc_video[eye][1][0]))
        return [gaze_rect_h, gaze_dot_h, world_h, eye_left_h, eye_right_h, gc_h]

    def animate(i):
        gaze_rect_h.set_xy(
            gaze_matched[eye]['norm_pos'][i] - np.array([rect_width/2, rect_height/2]))
        gaze_dot_h.set_offsets(gaze_matched[eye]['norm_pos'][i])
        pupil_left_h.set_offsets(pupil_left_matched['norm_pos'][i])
        world_h.set_data(world[i])
        eye_left_h.set_data(eye_left_ds[i])
        eye_right_h.set_data(eye_right_ds[i])
        try:
            gc_h.set_data(gc_video[eye][1][i])
        except:
            print("GAAAAAA")
        return [gaze_rect_h, gaze_dot_h, world_h, eye_left_h, pupil_left_h, eye_right_h, gc_h]

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=n_frames, interval=1/fps * 1000, blit=True)
    return anim


