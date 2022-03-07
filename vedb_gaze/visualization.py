import plot_utils
import copy
import numpy as np
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
    dot_locations,
    dot_timestamps=None,
    video_timestamps=None,
    dot_widths=None,
    dot_colors=None,
    dot_labels=None,
    dot_markers=None,
    figsize=None,
    fps=60,
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
    dot_locations : array
        [n_dots, n_frames, xy]: locations of dots to plot, either in normalized (0-1 for
        both x and y) coordinates or in pixel coordinates (with pixel dimensions matching
        the size of the video)
    dot_timestamps : array
        [n_dots, n_dot_frames] timestamps for dots to plot; optional if a dot location is
        specified for each frame. However, if `dot_timestamps` do not match
        `video_timestamps`, dot_locations are resampled (simple block average) to match
        with video frames using these timestamps.
    video_timestamps : array
        [n_frames] timestamps for video frames; optional, see `dot_timestamps`
    dot_widths : scalar or list
        size(s) for dots to plot
    dot_colors : matplotlib colorspec (e.g. 'r' or [1, 0, 0]) or list of colorspecs
        colors of dots to plot. Only allows one color across time for each dot (for now).
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
    dot_locs = dot_locations.copy()
    # Inputs
    if np.ndim(dot_locs) == 2:
        dot_locs = dot_locs[np.newaxis, :]
    if dot_markers is None:
        dot_markers = "o"

    # Shape
    extent = [0, 1, 1, 0]
    # interval is milliseconds; convert fps to milliseconds per frame
    interval = 1000 / fps
    # Setup
    n_frames, y, x = video_data.shape[:3]
    im_shape = (y, x)
    aspect_ratio = x / y
    if figsize is None:
        figsize = (5 * aspect_ratio, 5)
    if np.nanmean(dot_locs) > 1:
        dot_locs /= np.array([x, y])
    # Match up timestamps
    if video_timestamps is not None:
        mean_video_frame_time = np.mean(np.diff(video_timestamps))
        # Need for loop over dots if some dots
        # have different timestamps than others
        tt = np.repeat(dot_timestamps[:, np.newaxis], len(video_timestamps), axis=1)
        tdif = video_timestamps - tt
        t_i, vframe_i = np.nonzero(np.abs(tdif) < (mean_video_frame_time / 2))
        # Downsample dot locations
        vframes = np.unique(vframe_i)
        #print(vframes)
        n_dots_ds = len(vframes)
        #print(n_dots_ds)
        dot_locs_ds = np.hstack(
            [
                np.median(dot_locs[:, t_i[vframe_i == j]], axis=1)[:, None, :]
                for j in vframes
            ]
        )
        # print(dot_locs_ds.shape)
    else:
        dot_locs_ds = dot_locs
        vframes = np.arange(n_frames)
    # Plotting args
    n_dots, n_dot_frames, _ = dot_locs_ds.shape
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
    for dc, dw, dm, dl in zip(dot_colors, dot_widths, dot_markers, dot_labels):
        tmp = plt.scatter(0.5, 0.5, s=dw, c=dc, marker=dm, label=dl)
        dots.append(tmp)
    artists = (im,) + tuple(dots)
    plt.close(fig.number)
    # initialization function: plot the background of each frame
    def init_func(fig, ax, artists):
        for d in dots:
            d.set_offsets([0.5, 0.5])
        im.set_array(np.zeros(im_shape))
        return artists

    # animation function. This is called sequentially
    def update_func(i, artists, dot_locs, vframes):

        artists[0].set_array(video_data[i])
        # Also needs attention if different timecourses for different dots
        for j, artist in enumerate(artists[1:]):
            # may end up being: if i in vframes[j]:
            if i in vframes:
                dot_i = vframes.tolist().index(i)
                _ = artist.set_offsets(dot_locs[j, dot_i])
            else:
                _ = artist.set_offsets([-1, -1])
        return artists

    init = partial(init_func, fig=fig, ax=ax, artists=artists)
    update = partial(
        update_func, artists=artists, dot_locs=dot_locs_ds, vframes=vframes
    )
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, func=update, init_func=init, frames=n_frames, interval=interval, blit=True
    )
    return anim


def show_ellipse(ellipse, img=None, ax=None, **kwargs):
    """Show opencv ellipse in matplotlib, optionally with image underlay

    Parameters
    ----------
    ellipse : dict
        dict of ellipse parameters derived from opencv, with fields:
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
    ax.scatter(ellipse["center"][0], ellipse["center"][1], color="r")


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


def show_dots(mk, start=0, end=10, size=0.25, fps=30, video='world_camera', **kwargs):
    """Make an animation of detected markers as dots overlaid on video
    mk is a vedb_store class (MarkerDetection or PupilDetection so far)
    Set vdieo, size, frame rate according to what mk is!
    """
    mk.db_load()
    vtime, vdata = mk.session.load(video, time_idx=(start, end), size=size)
    anim = make_dot_overlay_animation(vdata, 
                                    mk.data['norm_pos'],
                                    dot_timestamps=mk.timestamp,
                                    video_timestamps=vtime,
                                    fps=30,
                                    **kwargs,
                                    )
    return anim


def average_frames(ses, times, to_load='world_camera'):
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
        timestamp, frames = ses.load(to_load, time_idx=[t0, t0 + 1])
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


def plot_error_interpolation(marker, gaze_err, xgrid, ygrid, gaze_err_image, vmin=0, vmax=None, ax=None,
                             cmap='viridis_r', azimuth=60, elevation=30, **kwargs):
    """Summary
    
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
    n_rows_top = 4  # validation / calibration; pupil confidence; pupil skewness; pupil images
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
            calibration_points_all.timestamp, ses.world_time, color='b', alpha=0.3, ax=ax0a)
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
    ax0d = fig.add_subplot(gs[3, :])
    sample_times_world = sample_times[::2]
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
            set_axis_lines(ax_L, color=left_eye_col, lw=2)
            #[h_hst, h_im, h_contour, h_scatter]
            hst_im = handles[0]
            fig.colorbar(hst_im, orientation='horizontal',
                         ax=ax_L, aspect=15, fraction=0.075)
            # Rigth eye error
            handles = plot_error(error[j], eye='right', gaze=gaze['right'], ax=ax_R,
                                 **err_kw)
            set_axis_lines(ax_L, color=left_eye_col, lw=2)
            err_im = handles[1]
            try:
                fig.colorbar(err_im, orientation='horizontal',
                             ax=ax_R, aspect=15, fraction=0.075)
            except:
                print("session error is high, colorbar for error contours fails")
            set_axis_lines(ax_R, color=right_eye_col, lw=2)

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
                   calibration_tag='plab_default-circles_halfres-cluster_default-monocular_pl_default',
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
    try:
        mk_cal = dbi.query(1, type='MarkerDetection',
            epoch='all',
            session=session._id,
            tag=marker_tag_cal,
            )
    except:
        mk_cal = None
    # All epochs for calibration
    try: 
        epoch_cal = dbi.query(type='MarkerDetection',
            session=session._id,
            tag=f'{marker_tag_cal}-{marker_epoch_tag_cal}')
        epoch_cal = sorted(epoch_cal, key=lambda x: x.epoch)
    except:
        epoch_cal = None
    # All epochs for validation
    try: 
        epoch_val = dbi.query(type='MarkerDetection',
                              session=session._id,
                              tag=f'{marker_tag_val}-{marker_epoch_tag_val}')
        epoch_val = sorted(epoch_val, key=lambda x: x.epoch)
        # Re-assign epochs?
        if re_assign_markers:
            if n_cal > 1:
                epoch_val.extend(epoch_cal[1:])
                epoch_cal = epoch_cal[:1]
    except:
        epoch_val = None
    # Calibration
    calibration = {}
    if 'monocular' in calibration_tag:
        eyes = ['left', 'right']
    else:
        eyes = ['both']
    try:
        for eye in eyes:
            tmp = dbi.query(1, type='Calibration', 
                            tag=calibration_tag, 
                            marker_detection=epoch_cal[calibration_epoch]._id,
                            eye=eye, 
                            epoch=calibration_epoch,
                            session=session._id,
                            )
            tmp.load()
            calibration[eye] = tmp
    except:
        raise
        calibration = None
    # Estimated gaze
    gaze = {}
    try:
        for eye in eyes:
            gaze[eye] = dbi.query(1, 
                type='Gaze',
                session=session._id,
                tag=gaze_tag,
                calibration=calibration[eye],
                eye=eye,
                pupil_detections=pupil_tag,)
    except:
        gaze = None
    # Error
    try:
        err = []
        for ve in epoch_val:
            for eye in eyes:
                tmp = dbi.query(1, type='GazeError', 
                    marker_detection=ve,
                    gaze=gaze[eye]._id,
                    )
                err.append(tmp)
    except:
        if len(err) < 1:
            err = None
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

# Read from config file
gaze_latest = 'plab_default-circles_halfres-cluster_default-monocular_tps_default-default_mapper'
def show_gaze_movie(session, 
    gaze_tag=gaze_latest, 
    eye='both', 
    calibration_epoch=0,
    is_verbose=False,
    ):
    """Make a movie with overlaid gaze trace(s)
    
    """
    # Set verbosity
    verbosity = copy.copy(session.dbi.is_verbose)
    session.dbi.is_verbose = is_verbose
    # Retrieve gaze
    gaze = session.dbi.query(type='Gaze', 
        tag=gaze_tag, 
        calibration_epoch=calibration_epoch,
        session=session._id)
    
