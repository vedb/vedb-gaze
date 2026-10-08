"""Plotting and animation utilities for the vedb_gaze pipeline.

Provides matplotlib-based visualizations for each pipeline stage: 2D
histograms of gaze position, dot overlays on world/eye video (animations),
pupil ellipses, calibration / validation marker positions (including image
crops of markers at cluster centroids), gaze error maps and surfaces, 2D
color maps for position-coded scatter plots, and a multi-panel quality
control figure for a whole session (`plot_session_qc`).

Most functions work on "arraydicts" (dicts of numpy arrays with keys such
as 'timestamp', 'norm_pos', 'confidence'), with positions in normalized
(0-1) image coordinates.

Notes
-----
Depends on the external `plot_utils` and `file_io` packages. Some functions
(`plot_eye_at_marker`, `plot_at_times`) use `vedb_store`, and `show_dots`
expects `vedb_store` objects. Several functions near the end of the module
(`make_gaze_animation`, `show_session`, `label_eye_video`, `plot_at_times`)
reference undefined names and appear not to run as written. Large
triple-quoted blocks in the module hold older, disabled code.
"""
import matplotlib.pyplot as plt
from matplotlib import animation, patches, colors, gridspec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from scipy import interpolate
import numpy as np
import pathlib
import copy
import os

import plot_utils
import file_io
from .options import config
from .marker_parsing import marker_cluster_stat, split_timecourse
from .utils import load_pipeline_elements,  match_time_points
from . import calibration as vedbcalibration

BASE_DIR = pathlib.Path(config.get('paths','base_dir'))
PROC_DIR = pathlib.Path(config.get('paths','proc_dir'))

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
    vmin_x=0,
    vmax_x=1,
    vmin_y=0,
    vmax_y=1, 
    cmax_pct=97.5, 
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
    vmin_x, vmax_x : float, optional
        horizontal range of histogram bins and image extent, by default 0 and 1
    vmin_y, vmax_y : float, optional
        vertical range of histogram bins and image extent, by default 0 and 1
        (y axis is displayed inverted, i.e. image-style, with vmin_y at top)
    cmax_pct : float, optional
        percentile of histogram values used as the upper color limit,
        by default 97.5

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
        extent=[vmin_x, vmax_x, vmax_y, vmin_y],
        aspect='auto',
        cmap=cmap)
    hst = np.histogram2d(y, x, bins=[np.linspace(
        vmin_y, vmax_y, hist_bins_y), np.linspace(vmin_x, vmax_x, hist_bins_x)], density=True)
    vmax_hst = np.percentile(hst[0], cmax_pct)
    hst_im = ax.imshow(hst[0], vmax=vmax_hst, **im_kw_hist)
    return hst_im


def circ_dist(a, b, degrees=True):
    """Compute angle between two angles w/ circular statistics

    Parameters
    ----------
    a : scalar or array
        angle(s) TO WHICH to compute angular (aka rotational) distance
    b : scalar or array
        angle(s) FROM WHICH to compute angular (aka rotational) distance
    degrees : bool
        Whether a and b are in degrees (defaults to True; False means they are in radians)

    Returns
    -------
    dist : scalar or array
        signed angular difference `a - b`, wrapped to [-180, 180] degrees
        (or [-pi, pi] radians if `degrees` is False)
    """
    if degrees:
        a = np.radians(a)
        b = np.radians(b)
    phase = np.e**(1j*a) / np.e**(1j*b)
    dist = np.arctan2(phase.imag, phase.real)
    if degrees:
        dist = np.degrees(dist)
    return dist


abins = np.linspace(-180, 180, 37, endpoint=True)
def angle_hist(angles, bins=abins, ax=None):
    """Plot a circular histogram of angles

    Parameters
    ----------
    angles : array-like
        angles (in degrees) to be histogrammed; wrapped to [-180, 180]
        before binning
    bins : array-like
        bin edges in degrees. Defaults to 36 evenly spaced bins (10 deg each)
    ax : matplotlib axis
        axis into which to plot. If specified, it must hav been
        created with `projection='polar'`. By default None, which creates
        a new polar figure & axis

    Notes
    -----
    Zero degrees is drawn at the top and angles increase clockwise.
    Nothing is returned.
    """
    # Compute pie slices
    angles = circ_dist(angles, 0)
    hh, bedges = np.histogram(angles, bins=bins)

    theta = np.radians(bins[:-1] + np.mean(np.diff(bins))/2)
    radii = hh
    n = len(bins)-1
    width = np.ones(n) * np.pi*2 / n
    #colors = plt.cm.viridis(radii / 10.)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6),
                               subplot_kw=dict(projection='polar'))
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi/2.0)
    ax.bar(theta, radii, width=width, bottom=0.0)


def plot_timestamps(timestamps, full_time, start_time=0, ax=None, **kwargs):
    """Plot `timestamps` within `full_time` as indicator variables (ones)

    Parameters
    ----------
    timestamps : array
        timestamps (seconds) for events to plot
    full_time : array
        array of all possible timestamps (seconds)
    start_time : int, optional
        time to subtract off `timestamps` (use if start time in `timestamps` differs
        from start time in `full_time`), by default 0
    ax : matplotlib axis, optional
        axis into which to plot, by default None, which opens a new figure
    **kwargs
        passed to `ax.plot`

    Notes
    -----
    The x axis is in minutes (`full_time / 60`). Membership is tested with
    exact equality (`np.in1d`), so `timestamps - start_time` must exactly
    match values in `full_time`.
    """
    xlim = [full_time.min()/60, full_time.max()/60]
    # Get timing for each epoch
    calibration_time_index = np.in1d(full_time, timestamps-start_time)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(full_time / 60, calibration_time_index, **kwargs)
    ax.set_xlim(xlim)


def set_eye_axis_lims(ax, eye):
    """Set 0-1 axis limits for an eye image axis, flipped for the left eye

    Parameters
    ----------
    ax : matplotlib axis
        axis to modify
    eye : str
        'right' sets axis to [0, 1, 0, 1]; 'left' sets it to [1, 0, 1, 0]
        (both x and y reversed). Any other value leaves the axis unchanged.
    """
    if eye == 'right':
        ax.axis([0, 1, 0, 1])
    elif eye == 'left':
        ax.axis([1, 0, 1, 0])


def make_world_eye_axes(eye_left_ax=True, eye_right_ax=True, fig_scale=5):
    """Create a figure with a world-camera axis above two eye axes

    Layout is a 5 x 4 grid: the world axis spans the top 3 rows, and the
    left and right eye axes split the bottom 2 rows. Eye axis limits are
    set with `set_eye_axis_lims`.

    Parameters
    ----------
    eye_left_ax : bool, optional
        whether to create the left eye axis, by default True
    eye_right_ax : bool, optional
        whether to create the right eye axis, by default True
    fig_scale : float, optional
        size (inches) of the longer figure dimension, by default 5

    Returns
    -------
    world_ax : matplotlib axis
        axis for the world camera image
    eye_left_ax : matplotlib axis or bool
        axis for the left eye, or False if not created
    eye_right_ax : matplotlib axis or bool
        axis for the right eye, or False if not created
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


def plot_eye_at_marker(session, marker, pupils,
                       ax=None,
                       alpha=1.0,
                       confidence_cmap=plt.cm.viridis,
                       n_fake_clusters=12,
                       is_verbose=True):
    """Plot eye video frames at the world positions of marker clusters

    For each marker cluster, an eye video frame from near the cluster's
    median time is drawn at the cluster's mean marker position (normalized
    0-1 world coordinates), bordered with a color that encodes the median
    pupil confidence during that cluster.

    Parameters
    ----------
    session : vedb_store session object
        must provide `get_video_time(name)` and `load(name, frame_idx=...)`
        for 'eye_left' / 'eye_right'
    marker : dict
        marker arraydict with 'timestamp' and 'norm_pos' fields, and
        optionally 'marker_cluster_index'
    pupils : dict
        pupil arraydict with 'id' (0 = right eye, 1 = left eye) and
        'confidence' fields; `confidence` is indexed by eye video frame
    ax : matplotlib axis, optional
        axis into which to plot, by default None, which creates a new figure
    alpha : float, optional
        opacity of plotted images, by default 1.0
    confidence_cmap : matplotlib colormap, optional
        colormap for image borders by median pupil confidence, by default
        plt.cm.viridis. Confidence values below 0.6 are set to 0.
    n_fake_clusters : int, optional
        if `marker` has no 'marker_cluster_index', markers are split into
        this many equal-size consecutive blocks, by default 12
    is_verbose : bool, optional
        unused, by default True

    Notes
    -----
    Imports `vedb_store`. Note that `plot_session_qc` calls this function
    with a video file path as `session`, which will not work.
    """
    # Get rid of me. Circular dependency.
    import vedb_store
    # marker.db_load()
    # session = marker.session
    #pupils.db_load()
    #eye = pupils.eye
    eye_id = pupils['id'][0]
    eye = ['right','left'][eye_id]
    if ax is None:
        fig, ax = plt.subplots()
        fig.patch.set_color('w')
    clustered = copy.deepcopy(marker)
    if 'marker_cluster_index' not in clustered:
        n_pts = len(clustered['timestamp'])
        clustered['marker_cluster_index'] = np.floor(
            np.linspace(0, n_fake_clusters, n_pts, endpoint=False))

    cluster_centroids = []
    cluster_times = []
    cluster_times_all = []
    for j in np.unique(clustered['marker_cluster_index']):
        mm = np.mean(clustered['norm_pos']
                     [clustered['marker_cluster_index'] == j], axis=0)
        tt = np.median(clustered['timestamp']
                       [clustered['marker_cluster_index'] == j])
        tt_all = clustered['timestamp'][clustered['marker_cluster_index'] == j]
        #conf = np.median(clustered['confidence'][clustered['marker_cluster_index']==j])
        cluster_centroids.append(mm)
        cluster_times.append(tt)
        cluster_times_all.append([tt_all[0], tt_all[-1]])

    eye_time = session.get_video_time('eye_%s' % eye)
    eye_frames = [vedb_store.utils.get_frame_indices(
        ct, ct+1, eye_time)[0] for ct in cluster_times]
    eye_frames_st = [vedb_store.utils.get_frame_indices(
        ct[0], ct[0]+1, eye_time)[0] for ct in cluster_times_all]
    eye_frames_fin = [vedb_store.utils.get_frame_indices(
        ct[1], ct[1]+1, eye_time)[0] for ct in cluster_times_all]
    pupil_conf = pupils['confidence']
    #print(pupil_conf.shape)
    cluster_confidence = [np.median(pupil_conf[st:fin])
                          for st, fin in zip(eye_frames_st, eye_frames_fin)]
    confidence = np.array(cluster_confidence)
    confidence_threshold = 0.6

    confidence[confidence < confidence_threshold] = 0
    colors = [dict(color=confidence_cmap(c), lw=1.5) for c in confidence]
    images = [session.load('eye_%s' % eye, frame_idx=(
        fr, fr+1))[1][0] for fr in eye_frames]
    plot_utils.plot_ims(*np.array(cluster_centroids).T,
                        np.array(images),
                        imsz=0.15,
                        ax=ax,
                        xlim=[0, 1],
                        ylim=[1, 0],
                        im_border=colors,
                        alpha=alpha,)


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
        (displayed with vmin=0, vmax=255)
    video_timestamps : array
        [n_frames] timestamps for video frames
    *dot_data : dict
        one arraydict per set of dots, each with 'timestamp' and 'norm_pos'
        ((n, 2), normalized 0-1 coordinates) fields. Dots are shown on video
        frames whose timestamps exactly match dot timestamps.
    downsampling_function : callable, optional
        passed as `fn` to `match_time_points` if dot timestamps need to be
        resampled to video timestamps, by default np.median
    window : float, optional
        passed to `match_time_points`, by default None (nearest-frame matching)
    dot_widths : scalar or list
        size(s) for dots to plot (scatter `s`), by default None
    dot_colors : matplotlib colorspec (e.g. 'r' or [1, 0, 0]) or list of colorspecs
        colors of dots to plot, by default None
    dot_labels : string or list of strings
        label per dot (passed to scatter `label`; no legend is drawn)
    dot_markers : string or list of strings
        marker type for each dot, by default 'o'
    figsize : tuple
        Size of figure, by default None, which gives (5 * aspect ratio, 5)
    fps : scalar
        fps for resulting animation, by default 60
    accumulate : bool, optional
        if True, each frame shows all earlier dots (up to, but not including,
        the current one; earlier dots stay on frames with no match); if False
        (default), only the current dot is shown and dots are hidden on
        frames without a matching timestamp
    **kwargs
        unused

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        animation with one frame per video frame (the figure itself is closed)

    Notes
    -----
    The check meant to skip resampling when dot timestamps already match
    video timestamps is always True as written, so `match_time_points` is
    never called and dot timestamps are used as given.


    Good tutorial, fancy extras: https://alexgude.com/blog/matplotlib-blitting-supernova/
    """
    from functools import partial

    def prep_input(x, n):
        """Wrap `x` in a list and repeat a single value `n` times."""
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
    im = ax.imshow(video_data[0], extent=extent, cmap="gray", aspect="auto", vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    dots = []
    for this_dot, dc, dw, dm, dl in zip(dots_matched, dot_colors, dot_widths, dot_markers, dot_labels):
        tmp = ax.scatter(*this_dot['norm_pos'].T,
                          s=dw, c=dc, marker=dm, label=dl)
        dots.append(tmp)
    ax.set_xlim([0, 1])
    ax.set_ylim([1, 0])
    artists = (im,) + tuple(dots)
    plt.close(fig.number)
    # initialization function: plot the background of each frame

    def init_func(fig, ax, artists):
        """Blank the image and place each dot set at its first position."""
        for j, d in enumerate(dots):
            d.set_offsets(dots_matched[j]['norm_pos'][:1])
        im.set_array(np.zeros(im_shape))
        return artists

    # animation function. This is called sequentially
    def update_func(i, artists, dots_matched, vframes):
        """Show video frame `i` and update dot positions for that frame."""

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
    img : array, optional
        underlay image to display (in pixel coordinates), by default None
    ax : matplotlib axis, optional
        axis into which to plot ellipse, by default None, which creates a
        new figure
    center_color : matplotlib colorspec, optional
        color of the dot plotted at the ellipse center, by default 'r'
    kwargs : passed to matplotlib.patches.Ellipse

    Returns
    -------
    patch_h : matplotlib.patches.Ellipse
        handle for the ellipse patch
    pt_h : matplotlib PathCollection
        handle for the center dot

    Notes
    -----
    `axes` are passed directly as the Ellipse width and height (i.e. full
    axis lengths, as in opencv `fitEllipse` output).
    """
    if ax is None:
        fig, ax = plt.subplots()
    ell = patches.Ellipse(
        ellipse["center"], *ellipse["axes"], angle=ellipse["angle"], **kwargs
    )
    if img is not None:
        ax.imshow(img, cmap="gray")
    patch_h = ax.add_patch(ell)
    pt_h = ax.scatter(ellipse["center"][0], ellipse["center"][1], color=center_color)
    return patch_h, pt_h

def _set_ellipse(ellipse_h, dot_h, ellipse, frame, eye_video_size=(1, 1)):
    """Update ellipse and center-dot handles to the ellipse for one frame

    Parameters
    ----------
    ellipse_h : matplotlib.patches.Ellipse
        ellipse handle (e.g. from `show_ellipse`)
    dot_h : matplotlib PathCollection
        center dot handle (e.g. from `show_ellipse`)
    ellipse : sequence of dict
        per-frame ellipse dicts with 'center', 'axes', 'angle' fields
    frame : int
        index into `ellipse`
    eye_video_size : scalar or tuple, optional
        size of the video in pixels; center and axes are divided by this to
        normalize the ellipse to 0-1 for plotting. By default (1, 1), which
        does nothing. The angle is not rescaled.

    Returns
    -------
    ellipse_h, dot_h
        the updated handles
    """
    tmp = ellipse[frame]
    ellipse_data = dict((k, np.array(v) / eye_video_size)
                for k, v in tmp.items())
    ellipse_h.set_center(ellipse_data['center'])
    ellipse_h.set_height(ellipse_data['axes'][1])
    ellipse_h.set_width(ellipse_data['axes'][0])
    ellipse_h.set_angle(tmp['angle'])
    # Accumulate?
    dot_h.set_offsets([ellipse_data['center']])
    return ellipse_h, dot_h

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
        image of values to use for 2D color map, (rows, cols, channels);
        by default BuOr_2D (module-level 256 x 256 RGB map)
    vmin0, vmax0 : float, optional
        range of `data0`, mapped to image columns (left to right); by
        default None, which uses the data min / max
    vmin1, vmax1 : float, optional
        range of `data1`, mapped to image rows (bottom to top, i.e. high
        values of `data1` index the first row); by default None, which uses
        the data min / max
    map_to_uint8 : bool, optional
        whether to scale output (assumed 0-1) to 0-255 uint8, by default False

    Returns
    -------
    colored : array
        (n, channels) colors, one per data point. Values outside the ranges
        are clipped; NaNs map to index 0.
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

    Parameters
    ----------
    mk : vedb_store object
        object with `db_load()` and a `session` attribute; passed directly
        as dot data to `make_dot_overlay_animation`, so it must also support
        indexing by 'timestamp' and 'norm_pos'
    start : float, optional
        start time (seconds) of video segment to load, by default 0
    end : float, optional
        end time (seconds) of video segment to load, by default 10
    size : float, optional
        video resize factor passed to `session.load`, by default 0.25
    fps : float, optional
        frame rate of the animation, by default 30
    video : str, optional
        name of video to load from session, by default 'world_camera'
    accumulate : bool, optional
        passed to `make_dot_overlay_animation`, by default False
    **kwargs
        passed to `make_dot_overlay_animation`

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    mk.db_load()
    vtime, vdata = mk.session.load(video, time_idx=(start, end), size=size)
    anim = make_dot_overlay_animation(vdata,
                                      vtime + mk.session.start_time,
                                      mk,
                                      fps=fps,
                                      accumulate=accumulate,
                                      **kwargs,
                                      )
    return anim


def average_frames(session, times, to_load='world_camera'):
    """Load and average frames that occur at a specified list of `times`
    
    Parameters
    ----------
    session : vedb_store session object
        object with a `load(name, time_idx=...)` method returning
        (timestamps, frames)
    times : array-like
        times (seconds) at which to load frames; for each, the first frame
        in [t, t + 1) is used
    to_load : str, optional
        name of video to load, by default 'world_camera'

    Returns
    -------
    array
        mean of the loaded frames, cast to the frames' dtype
    """
    images = []
    for t0 in times:
        timestamp, frames = session.load(to_load, time_idx=[t0, t0 + 1])
        images.append(frames[0])
    return np.array(images).mean(0).astype(frames.dtype)


def show_average_frames(ses, times, n_images=4, ax=None, extent=(0, 1, 1, 0), mode='linear', to_load='world_camera'):
    """Show the average of video frames sampled from a set of times

    Parameters
    ----------
    ses : vedb_store session object
        passed to `average_frames`
    times : array-like
        candidate times (seconds) from which to sample frames
    n_images : int, optional
        number of frames to average, by default 4
    ax : matplotlib axis, optional
        axis into which to plot, by default None, which creates a new figure
    extent : tuple, optional
        image extent for `imshow`, by default (0, 1, 1, 0)
    mode : str, optional
        'linear' takes evenly spaced samples from `times`; anything else
        samples `n_images` times at random without replacement,
        by default 'linear'
    to_load : str, optional
        name of video to load, by default 'world_camera'

    Notes
    -----
    In 'linear' mode the step is `len(times) // n_images`, minus 1 if that
    evenly divides `len(times)`; as written this raises an error (step of
    0, or modulo by zero) whenever `len(times) < 2 * n_images`.
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
    confidence_threshold=0.6,
    label_contours=False,
    contour_fontsize=16):
    """Plot gaze error (degrees) across the world camera field of view

    Optionally layers a 2D histogram of gaze positions, a translucent image
    of the interpolated error surface, error contours, and a scatter of
    error at each validation marker position, all in normalized (0-1)
    world coordinates.

    Parameters
    ----------
    err : dict
        error estimation output with fields 'gaze_err_image' (2D
        interpolated error, degrees), 'xgrid', 'ygrid' (grid coordinates for
        the error image), 'marker' ((n, 2) normalized marker positions) and
        'gaze_err' ((n,) error in degrees at each marker)
    gaze : dict, optional
        gaze arraydict ('norm_pos', 'confidence'); if provided, a gaze
        histogram is drawn with `gaze_hist`, by default None
    ax : matplotlib axis, optional
        axis into which to plot, by default None, which creates a new figure
    cmap : str, optional
        colormap for contours and marker scatter, by default 'inferno_r'
        (the error image always uses 'inferno_r')
    cmap_hist : str, optional
        colormap for gaze histogram, by default 'gray_r'
    levels : tuple, optional
        error levels (degrees) for contours; None or empty skips contours,
        by default (1, 2.5, 5, 10)
    err_vmax : int, optional
        upper color limit for error (degrees), by default 10
    err_vmin : int, optional
        lower color limit for error (degrees), by default 0
    plot_image : bool, optional
        whether to show the error image, by default True
    scatter_size : int, optional
        size of marker scatter points, by default 3
    hist_bins_x : int, optional
        horizontal bins for gaze histogram, by default 81
    hist_bins_y : int, optional
        vertical bins for gaze histogram, by default 61
    confidence_threshold : float, optional
        minimum gaze confidence for histogram, by default 0.6
    label_contours : bool, optional
        whether to label contour lines, by default False
    contour_fontsize : int, optional
        font size for contour labels, by default 16

    Returns
    -------
    list
        plot handles: [histogram image (only if `gaze` given), error image
        or None, contour set or None, marker scatter]
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
    else:
        out.append(None)
    if (levels is not None) and (len(levels) > 0):
        contour_kw = dict(levels=levels, cmap=cmap, vmin=err_vmin, vmax=err_vmax)
        h_contour = ax.contour(err['xgrid'], err['ygrid'], err['gaze_err_image'], **contour_kw)
        if label_contours:
            ax.clabel(h_contour, inline=True, fontsize=contour_fontsize)
        out.append(h_contour)
    else:
        out.append(None)
    scatter_kw = dict(cmap=cmap, vmin=err_vmin, vmax=err_vmax, s=scatter_size)
    h_scatter = ax.scatter(*err['marker'].T, c=err['gaze_err'], **scatter_kw)
    out.append(h_scatter)
    ax.axis([0, 1, 1, 0])
    return out


def plot_error_markers(markers, gaze,
                                confidence_threshold=0,
                                ax=None,
                                do_gradients=True,
                                # Blue # (0.95, 0.85, 0) # Yellow
                                # Yellow # (1.00, 0.50, 0) # Orange
                                eye_color=(0.0, 0.0, 0.9),
                                 # Red
                                marker_color=(1.0, 0, 0), 
                                ):
    """Plot validation markers and gaze at relevant timepoints

    Draws a line from each marker position to the matched gaze position,
    with a color gradient from `marker_color` to `eye_color`, in normalized
    (0-1) world coordinates.

    Parameters
    ----------
    markers : dict or array
        marker arraydict ('timestamp', 'norm_pos'), or an (n, 2) array of
        marker positions
    gaze : dict or array
        gaze arraydict ('timestamp', 'norm_pos', 'confidence'), or an
        (n, 2) array of gaze positions. If both inputs are arrays they are
        used as-is (no confidence filtering). If dicts with different
        lengths, gaze is matched to marker timestamps with
        `match_time_points`.
    confidence_threshold : float, optional
        minimum gaze confidence for points to keep (dict inputs only),
        by default 0
    ax : matplotlib axis, optional
        axis into which to plot, by default None, which creates a new figure
    do_gradients : bool, optional
        whether to draw the marker-to-gaze lines, by default True
    eye_color : tuple, optional
        RGB color at the gaze end of the lines, by default (0.0, 0.0, 0.9)
    marker_color : tuple, optional
        RGB color at the marker end of the lines, by default (1.0, 0, 0)

    Notes
    -----
    Returns None without plotting if fewer than 10 points pass the
    confidence threshold (the message printed in that case references an
    undefined variable `j` and will raise a NameError).
    """
    # left: (0.0, 0.0, 0.9)
    # right: (0.95, 0.85, 0)
    eye_cmap = (marker_color, eye_color)

    
    if isinstance(gaze, np.ndarray) and isinstance(markers, np.ndarray):
        vp = markers
        gl = gaze
    else:
        if len(markers['norm_pos']) != len(gaze['norm_pos']):
            gaze_matched = match_time_points(markers, gaze)
        else:
            gaze_matched = gaze
        gl = gaze_matched['norm_pos']
        vp = markers['norm_pos']
        gl_ci = gaze_matched['confidence'] > confidence_threshold        
        if (gl_ci.sum() < 10):
            print("Insufficient number of points retained after assessing pupil confidence for v epoch %d" % j)
            return
        vp = vp[gl_ci]
        gl = gl[gl_ci]
    if ax is None:
        fig, ax = plt.subplots()
        fig.patch.set_color('w')
    # Visualization of markersidation epochs w/ gaze
    if do_gradients:
        plot_utils.gradient_lines(
            vp, gl, cmap=eye_cmap, ax=ax)
    ax.axis([0, 1, 1, 0])


def load_markers(markers, video_fpath, fn=np.nanmedian, clusters=None, crop_size=(128, 128), tdelta=0.5):
    """Load world video crops centered on each marker cluster

    Parameters
    ----------
    markers : dict
        marker arraydict with 'norm_pos' and 'timestamp' fields (and
        'marker_cluster_index' if `clusters` is not given)
    video_fpath : str or pathlib.Path
        path to world video file, loaded with `file_io.load_video`
    fn : callable, optional
        statistic used to summarize each cluster (via
        `marker_cluster_stat`), by default np.nanmedian
    clusters : array-like, optional
        cluster index per marker, by default None (use
        markers['marker_cluster_index'])
    crop_size : tuple, optional
        size (pixels) of the crop around each marker, also used as output
        size, by default (128, 128)
    tdelta : float, optional
        duration (seconds) of video loaded from each cluster's summary
        time; only the first frame is kept, by default 0.5

    Returns
    -------
    marker_positions : array
        (n_clusters, 2) summary marker positions (normalized 0-1)
    marker_times : array
        (n_clusters,) summary timestamps
    marker_crops : list of array
        one RGB crop per cluster
    """
    mk_cut = marker_cluster_stat(markers, 
                                 fn=fn,
                                 clusters=clusters,
                                 field='norm_pos',
                                 return_all_fields=True)
    marker_times = mk_cut['timestamp']
    marker_positions = mk_cut['norm_pos']
    marker_crops = []
    for mk_pos, mk_t in zip(marker_positions, marker_times):
        wv = file_io.load_video(video_fpath,
                                time_idx=(mk_t, mk_t + tdelta),
                                size=crop_size,
                                crop_size=crop_size,
                                center=mk_pos,
                                color='rgb'
                                )
        marker_crops.append(wv[0])
    return marker_positions, marker_times, marker_crops


def show_clustered_markers(markers, folder, fn=np.nanmedian, clusters=None, 
                           crop_size=(128, 128), tdelta=0.25, ax=None, n_blocks=12, imsz=0.15):
    """Plot world video crops of each marker cluster at its world position

    Parameters
    ----------
    markers : dict
        marker arraydict with 'norm_pos' and 'timestamp' fields
    folder : str or pathlib.Path
        if a pathlib.Path, used directly as the video file path; otherwise
        joined to BASE_DIR (i.e. `BASE_DIR / folder`) and passed as the
        video path to `load_markers`
    fn : callable, optional
        statistic used to summarize each cluster, by default np.nanmedian
    clusters : array-like, optional
        cluster index per marker. If None and `markers` has no
        'marker_cluster_index', markers are split into `n_blocks` equal
        consecutive blocks. By default None
    crop_size : tuple, optional
        crop size in pixels, by default (128, 128)
    tdelta : float, optional
        passed to `load_markers`, by default 0.25
    ax : matplotlib axis, optional
        axis into which to plot, by default None
    n_blocks : int, optional
        number of blocks if clusters must be faked, by default 12
    imsz : float, optional
        displayed size of each crop (axis units), by default 0.15
    """
    if (clusters is None) and ('marker_cluster_index' not in markers):
        n_pts = len(markers['norm_pos'])
        clusters = np.floor(
            np.linspace(0, n_blocks, n_pts, endpoint=False))
    if isinstance(folder, pathlib.Path):
        video_fpath = folder
    else:
        video_fpath = BASE_DIR / folder
    mk_pos, mk_times, mk_frames = load_markers(markers, 
                                               video_fpath, fn=fn,
                                               clusters=clusters, 
                                               crop_size=crop_size,
                                               tdelta=tdelta)

    plot_utils.plot_ims(*np.array(mk_pos).T, mk_frames, imsz=imsz, ylim=[1, 0], xlim=[0, 1], ax=ax)


def plot_error_interpolation_surface(marker, gaze_err, xgrid, ygrid, gaze_err_image, vmin=0, vmax=None, ax=None,
                             cmap='viridis_r', azimuth=60, elevation=30, **kwargs):
    """Plot error surface for gaze in 3D
    
    Parameters
    ----------
    marker : array
        (n, 2) normalized (0-1) validation marker positions
    gaze_err : array
        (n,) gaze error (degrees) at each marker
    xgrid : array
        2D grid of x coordinates for `gaze_err_image`
    ygrid : array
        2D grid of y coordinates for `gaze_err_image`
    gaze_err_image : array
        2D interpolated error surface (degrees)
    vmin : int, optional
        lower color limit, by default 0
    vmax : float, optional
        upper color / z limit, by default None, which uses 1.1 x the max
        of `gaze_err`
    ax : matplotlib 3D axis, optional
        axis into which to plot (must have `projection='3d'`), by default
        None, which creates a new figure
    cmap : str, optional
        colormap for marker scatter (not the surface), by default 'viridis_r'
    azimuth : int, optional
        view azimuth (degrees), by default 60
    elevation : int, optional
        view elevation (degrees), by default 30
    **kwargs
        unused

    Notes
    -----
    x and y limits are fixed at [0.2, 0.8].
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


def plot_session_qc(folder, 
        video_fpath=None,
        pupil=None,
        calibration_marker=None,
        calibration_cluster=None, 
        calibration=None, 
        validation_marker=None,
        validation_cluster=None, 
        gaze=None,
        error=None,
        do_slow_plots = False,
        axs=None, 
        fig_scale=14,
        val_color = (0.9, 0.0, 0.0),
        val_color_lt = (0.9, 0.5, 0.5),
        cal_color = (0.0, 0.9, 0.0),
        cal_color_lt = (0.5, 0.9, 0.5),
        font_kw=None,
        fpath=None,
        close_figure=False,
        pupil_confidence_threshold = 0.7,
        calibration_marker_epoch=0,
        validation_marker_epoch=0,
        do_memory_cleanup=False,
        ):
    """Make quality control plot for VEDB session

    Creates (or fills) a 3 x 4 grid of axes summarizing each pipeline step:
    calibration / validation marker positions, pupil positions during
    calibration / validation for each eye, calibrations, gaze error maps,
    a timeline of pupil confidence and marker detections, and histograms of
    pupil confidence. Steps that are missing (None) or failed are reported
    as text in the corresponding axes.

    Parameters
    ----------
    folder : str or pathlib.Path
        session folder name; used in the figure title and, if
        `video_fpath` is None, to build `BASE_DIR / folder`
    video_fpath : str or pathlib.Path, optional
        world video path, used only for slow plots (`do_slow_plots`),
        by default None
    pupil : dict, optional
        dict with keys `left` and `right` for pupil detection results for
        left and right eyes, respectively. Results can be either dicts
        output by vedb_gaze.pupil_detection_pl.detect_pupils() or 
        a PupilDetection class returned from the `vedb_store` database.
        by default None
    calibration_marker : dict or vedb_store.MarkerDetection class, optional
        output of calibration marker detection, either a dict (directly from output
        of vedb_gaze.marker_detection.detect_<marker> function) or a class loaded
        from vedb_store database. Must contain 'norm_pos' field with normalized (0-1)
        locations of detected markers and 'timestamp' field with timestamps.
        by default None
    calibration_cluster : dict or vedb_store.MarkerDetection class, optional
        output of calibration marker filtering (to remove spurious detections). Either 
        a dict (directly from output of vedb_gaze.marker_parsing.find_epochs function)
        or a MarkerDetection class loaded from vedb_store database. Must contain 'norm_pos' field with normalized (0-1)
        locations of detected markers and 'timestamp' field with timestamps.
        by default None
    calibration : dict, optional
        dict with keys `left` and `right` (or `both`, for binocular) 
        for calibrations computed for left and right eyes, respectively.
        Values can be either `vedb_gaze.calibration.Calibration` instances or
        vedb_store Calibration objects (which are `.load()`-ed and their
        `.calibration` attribute used). Only 'left' and 'right' are plotted.
        by default None
    validation_marker : list, optional
        per-epoch validation marker detections (dicts or vedb_store
        objects with 'norm_pos' and 'timestamp'); the element at
        `validation_marker_epoch` is used, by default None
    validation_cluster : list, optional
        per-epoch filtered / clustered validation markers; the element at
        `validation_marker_epoch` is used, by default None
    gaze : dict, optional
        dict of {'left': gaze_left, 'right': gaze_right} or None, by default None
        `gaze_left` and `gaze_right` here are outputs of gaze estimation, either
        vedb_store objects or dictionaries (as output by `vedb_gaze.gaze_mapping.gaze_mapper`)
    error : dict, optional
        dict of {'left': error_left, 'right': error_right} or None, by default None
        `error_left` and `error_right` here are lists (one per validation
        epoch) of outputs of error estimation, either
        vedb_store objects or dictionaries (as output by `vedb_gaze.error_computation.compute_error`);
        the element at `validation_marker_epoch` is plotted with `plot_error`
    do_slow_plots : bool, optional
        if True, show video crops of markers and eye images at markers
        (`show_clustered_markers`, `plot_eye_at_marker`) instead of
        scatter plots, by default False
    axs : array of matplotlib axes, optional
        3 x 4 array of axes into which to plot, by default None, which
        creates a new figure
    fig_scale : int, optional
        figure size scale (inches) for a new figure, by default 14
    val_color : tuple, optional
        timeline color for filtered validation markers, by default (0.9, 0.0, 0.0)
    val_color_lt : tuple, optional
        timeline color for all detected validation markers, by default (0.9, 0.5, 0.5)
    cal_color : tuple, optional
        timeline color for filtered calibration markers, by default (0.0, 0.9, 0.0)
    cal_color_lt : tuple, optional
        timeline color for all detected calibration markers, by default (0.5, 0.9, 0.5)
    font_kw : dict, optional
        font keyword args for titles / text, by default None, which uses
        dict(fontname='Helvetica', fontsize=14)
    fpath : str or pathlib.Path, optional
        if given, figure is saved here (dpi=100), by default None
    close_figure : bool, optional
        whether to close the figure at the end, by default False
    pupil_confidence_threshold : float, optional
        threshold shown on pupil confidence histograms and used to compute
        the percent of pupil samples kept, by default 0.7
    calibration_marker_epoch : int, optional
        unused, by default 0
    validation_marker_epoch : int, optional
        index of the validation epoch to plot, by default 0
    do_memory_cleanup : bool, optional
        if True, set `_data = None` on non-dict (vedb_store) inputs after
        plotting to free memory, by default False

    Notes
    -----
    Nothing is returned. Timeline x axis is in minutes; the figure title
    gives session duration from the max pupil timestamp.
    """
    # Lazy functions         
    def check_status(x):
        """Determine whether an input failed or was never run"""
        # Recursive call for dicts
        if isinstance(x, dict) and (('left' in x) or ('right' in x) or ('both' in x)):
            return dict((k, check_status(v)) for k, v in x.items())
        if x is None:
            return 'not run'
        failed = check_failed(x)
        if failed:
            return 'failed'
        else:
            return 'ok'

    def check_failed(x):
        """Return True if `x` is None, empty, or flagged as failed."""
        if x is None:
            return True
        if isinstance(x, dict):
            list_values = []
            for vv in x.values():
                if isinstance(vv, np.ndarray) and isinstance(vv.tolist(), list):
                    list_values.append(len(vv))
                elif isinstance(vv, list):
                    list_values.append(len(vv))
            print(list_values)
            failed = all([v == 0 for v in list_values])
        elif isinstance(x, list):
            failed = (len(x) == 0) or np.all([check_failed(x_) for x_ in x])
        else:
            if hasattr(x, 'failed'):
                failed = x.failed
            else:
                failed = False # ??
        return failed

    def disp_fail(msg, ax, title=None, axis=(0, 1, 0, 1), **font_kw):
        """Write a centered failure message (and optional title) in `ax`."""
        ax.text(0.5, 0.5, msg, ha='center', va='center', **font_kw)
        ax.axis(axis)
        if title is not None:
            ax.set_title(title)
    try:
        max_time_seconds = np.max([v['timestamp'].max() for v in pupil.values()])
    except:
        max_time_seconds = 0

            
    # Inputs
    if font_kw is None:
        # Defaults (kwargs should not be dicts or other mutable types, so substitute here)
        font_kw = dict(fontname='Helvetica', fontsize=14)

    if axs is None:
        ar = 4/3
        fig, axs = plt.subplots(3, 4, figsize=(ar * fig_scale, 0.75 * fig_scale),)
        fig.patch.set_color('w')
        fig.suptitle('%s: %.1f mins'%(folder, max_time_seconds / 60),
                    **font_kw)
    else:
        fig = axs[0,0].figure
    if video_fpath is None:
        video_fpath = BASE_DIR / folder
    # Check status of all steps
    steps = ['pupil',
             'calibration_marker',
             'calibration_cluster',
             'calibration',
             'validation_marker',
             'validation_cluster',
             'gaze',
             'error']
    status = {}
    for step in steps:
        if step in ['validation_marker','validation_cluster']:
            this_step = locals()[step][validation_marker_epoch]
        elif step in ['error']:
            if error is None:
                this_step = None
            else:
                this_step = dict((ee, error[ee][validation_marker_epoch]) for ee in error.keys())
        else:
            this_step = locals()[step]
        status[step] = check_status(this_step)

    # Plot
    if status['calibration_cluster'] in ('not run', 'failed'):
        # Filtering of calibration markers failed; try fallback plot of unfiltered marker position
        if status['calibration_marker'] in ('not run', 'failed'):
            # Detection has failed entirely
            msg = 'detection: %s\nfiltering: %s'%(status['calibration_marker'],
                                                  status['calibration_cluster'])
            disp_fail(msg, axs[0,0], title='Calibration Markers', axis=[0, 1, 0, 1])
            mk_for_eyes = None
            title = 'Calibration Markers'
        else:
            # Raw detection OK, just filtering failed
            title='Calibration Markers\n(raw, filtering %s)'%(status['calibration_cluster'])
            if do_slow_plots:
                show_clustered_markers(calibration_marker,
                                                           video_fpath, 
                                                           n_blocks=8,
                                                           ax=axs[0,0]
                                                          )
            else:
                cols_mkc = colormap_2d(*calibration_marker['norm_pos'].T)
                axs[0,0].scatter(*calibration_marker['norm_pos'].T, 
                                 c=cols_mkc, alpha=0.1, )
            mk_for_eyes = calibration_marker
    else:
        # Calibration markers detected and filtered correctly.
        if do_slow_plots:
            show_clustered_markers(calibration_cluster,
                                                       video_fpath, 
                                                       ax=axs[0,0]
                                                      )
        else:
            cols_mkc = colormap_2d(*calibration_cluster['norm_pos'].T)
            
            axs[0,0].scatter(*calibration_cluster['norm_pos'].T, c=cols_mkc)
        title = 'Calibration Markers\n(filtered)'
        mk_for_eyes = calibration_cluster
    # For whatever circumstance, set axis
    axs[0,0].axis([0, 1, 1, 0])
    axs[0,0].set_yticks([])
    axs[0,0].set_xticks([])
    axs[0,0].set_title(title, **font_kw)
    
    # Pupils during calibration
    if status['pupil'] in ('failed', 'not run') or \
        status['calibration_marker'] in ('failed', 'not run'):
        msg_both = 'Pupil detection: %s\nCalibration marker%s'%(status['pupil'],
            status['calibration_marker'])
        disp_fail(msg_both, axs[1, 0], title='Left eye position\n during calibration', 
            axis=[1, 0, 1, 0],)
        disp_fail(msg_both, axs[2, 0], title='Right eye position\n during calibration', 
            axis=[0, 1, 0, 1],)
    else:
        # At least both pupils haven't failed.
        # To do: catch case for which pupil detection fails entirely for one eye. Seems unlikely.
        if do_slow_plots and (mk_for_eyes is not None):
            plot_eye_at_marker(video_fpath, mk_for_eyes, pupil['left'], ax=axs[1, 0])
            plot_eye_at_marker(video_fpath, mk_for_eyes, pupil['right'], ax=axs[2, 0])
        else:
            if mk_for_eyes is not None:
                pl_match = match_time_points(mk_for_eyes, pupil['left'], )
                axs[1, 0].scatter(*pl_match['norm_pos'].T, c=cols_mkc, s=pl_match['confidence']*10)
                pr_match = match_time_points(mk_for_eyes, pupil['right'], )
                axs[2, 0].scatter(*pr_match['norm_pos'].T, c=cols_mkc, s=pr_match['confidence']*10)
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                axs[1, 1].set_aspect('equal', 'box')
                axs[1, 0].axis([1, 0, 1, 0])
                axs[1, 0].set_title('Left eye position\n during calibration', **font_kw)
                axs[2, 0].set_xticks([])
                axs[2, 0].set_yticks([])
                axs[2, 1].set_aspect('equal', 'box')
                axs[2, 0].axis([0, 1, 0, 1])
                axs[2, 0].set_title('Right eye position\n during calibration', **font_kw)
            
    # Validation 
    if status['validation_cluster'] in ('not run', 'failed'):
        # Filtering of validation markers failed; try fallback plot of unfiltered marker position
        if status['validation_marker'] in ('not run', 'failed'):
            # Detection has failed entirely
            msg = 'detection: %s\nfiltering: %s'%(status['validation_marker'],
                                                  status['validation_cluster'])
            disp_fail(msg, axs[0,2], title='Validation Markers', axis=[0, 1, 0, 1])
            mk_for_eyes_v = None
            title = 'Validation Markers'
        else:
            # Raw detection OK, just filtering failed
            title='Validation Markers\n(raw, filtering %s)'%(status['validation_cluster'])
            if do_slow_plots:
                show_clustered_markers(validation_marker,
                                                           session, 
                                                           n_blocks=8,
                                                           ax=axs[0,2]
                                                          )
            else:
                cols_mkv = colormap_2d(*validation_marker['norm_pos'].T)
                axs[0,2].scatter(*validation_marker['norm_pos'].T, 
                                 c=cols_mkv, alpha=0.1, )
            mk_for_eyes_v = validation_marker
    else:
        # validation markers detected and filtered correctly.
        if do_slow_plots:
            show_clustered_markers(validation_cluster[validation_marker_epoch],
                                                       video_fpath, 
                                                       ax=axs[0,2]
                                                      )
        else:
            cols_mkv = colormap_2d(*validation_cluster[validation_marker_epoch]['norm_pos'].T)
            
            axs[0,2].scatter(*validation_cluster[validation_marker_epoch]['norm_pos'].T, c=cols_mkv)
        title = 'Validation Markers\n(filtered)'
        mk_for_eyes_v = validation_cluster[validation_marker_epoch]
    # For whatever circumstance, set axis
    axs[0,2].axis([0, 1, 1, 0])
    axs[0,2].set_yticks([])
    axs[0,2].set_xticks([])
    axs[0,2].set_title(title, **font_kw)
    
    # Pupils during validation
    if status['pupil'] in ('failed', 'not run') or \
        status['validation_marker'] in ('failed', 'not run'):
        msg_both = 'Pupil detection: %s\nvalidation marker%s'%(status['pupil'],
            status['validation_marker'])
        disp_fail(msg_both, axs[1, 2], title='Left eye position\n during validation', 
            axis=[1, 0, 1, 0],)
        disp_fail(msg_both, axs[2, 2], title='Right eye position\n during validation', 
            axis=[0, 1, 0, 1],)
    else:
        # At least both pupils haven't failed.
        # To do: catch case for which pupil detection fails entirely for one eye. Seems unlikely.
        if do_slow_plots and (mk_for_eyes_v is not None):
            plot_eye_at_marker(video_fpath, mk_for_eyes_v, pupil['left'], ax=axs[1, 2])
            plot_eye_at_marker(video_fpath, mk_for_eyes_v, pupil['right'], ax=axs[2, 2])
        else:
            if mk_for_eyes_v is not None:
                pl_match = match_time_points(mk_for_eyes_v, pupil['left'], )
                axs[1, 2].scatter(*pl_match['norm_pos'].T, c=cols_mkv, s=pl_match['confidence']*10)
                pr_match = match_time_points(mk_for_eyes_v, pupil['right'], )
                axs[2, 2].scatter(*pr_match['norm_pos'].T, c=cols_mkv, s=pr_match['confidence']*10)
                axs[1, 2].set_xticks([])
                axs[1, 2].set_yticks([])
                axs[1, 1].set_aspect('equal', 'box')
                axs[1, 2].axis([1, 0, 1, 0])
                axs[1, 2].set_title('Left eye position\n during validation', **font_kw)
                axs[2, 2].set_xticks([])
                axs[2, 2].set_yticks([])
                axs[2, 1].set_aspect('equal', 'box')
                axs[2, 2].axis([0, 1, 0, 1])
                axs[2, 2].set_title('Right eye position\n during validation', **font_kw)

    # Calibrations
    if status['calibration'] in ('not run', 'failed'):
        disp_fail('Calibration %s'%status['calibration'], axs[1, 1], **font_kw)
        disp_fail('Calibration %s'%status['calibration'], axs[2, 1], **font_kw)
    else:
        for lr, ax, cal_colors in zip(['left', 'right'], axs[1:3, 1], ['steelblue', 'orange']):
            if status['calibration'][lr] in ('failed', 'not run'):
               disp_fail(status['calibration'][lr], ax=ax, **font_kw)
            else:
                if not isinstance(calibration[lr], vedbcalibration.Calibration):
                    # Is a vedb_store Calibration object, loaded from database, which can
                    # itself load an instance of a vedb_gaze Calibration
                    calibration[lr].load()
                    this_cal = calibration[lr].calibration
                else:
                    this_cal = calibration[lr]
                this_cal.show_calibration(sc=0, ax_world=ax, color=cal_colors)
                
            ax.set_xticks([])
            ax.set_yticks([])
        
    # Error
    if status['error'] in ('not run', 'failed'):
        disp_fail('Errror calculation %s'%status['error'], axs[1, 3], **font_kw)
        disp_fail('Errror calculation %s'%status['error'], axs[2, 3], **font_kw)
    else:
        for lr, ax in zip(['left', 'right'], axs[1:3,3]):
            if status['error'][lr] in ('failed', 'not run'):
                disp_fail('Errror calculation %s'%status['error'][lr], ax, **font_kw)
            else:
                plot_error(error[lr][validation_marker_epoch], 
                                                gaze=gaze[lr],
                                                ax=ax)
                ax.set_title('Err: med=%.2f, wt=%.2f'%(np.median(error[lr][validation_marker_epoch]['gaze_err']),
                                                            error[lr][validation_marker_epoch]['gaze_err_weighted']))
            ax.set_xticks([])
            ax.set_yticks([])
        
    
    # Timeline plot:
    # A. Pupil confidence
    
    #if pupil['left'] is not None:
    for offset, lr in enumerate(['right','left']):
        if (status['pupil'] not in ('failed', 'not run')) and (status['pupil'][lr] not in ('failed', 'not run')):
            # Find epochs within pupils
            pupil_epochs = split_timecourse(pupil[lr], 
                                                max_epoch_gap=1)
            for pe in pupil_epochs:
                ptime = pe[0]['timestamp'] / 60
                axs[0, 1].imshow(pe[0]['confidence'][None,:], 
                                extent=[ptime[0], ptime[-1], offset + 0.6, offset + 1.4],
                                aspect='auto',
                                cmap='gray_r',
                                vmin=0.6, vmax=1.0,
                                )
    # B. All detected calibration markers
    if status['calibration_marker'] not in ('failed', 'not run'):
        axs[0, 1].scatter(calibration_marker['timestamp'] / 60, 
                        np.ones_like(calibration_marker['timestamp']) * 4,
                        c=cal_color_lt, alpha=0.05,
                        )
    # C. Filtered calibration markers
    if status['calibration_cluster'] not in ('failed', 'not run'):
        axs[0, 1].scatter(calibration_cluster['timestamp'] / 60, 
                          np.ones_like(calibration_cluster['timestamp']) * 3,
                          c=cal_color, alpha=0.05,
                         )
    # D. All detected validation markers
    if status['validation_marker'] not in ('failed', 'not run'):
        axs[0, 1].scatter(validation_marker[validation_marker_epoch]['timestamp'] / 60, 
                          np.ones_like(validation_marker[validation_marker_epoch]['timestamp']) * 6,
                          c=val_color_lt, alpha=0.05,
                         )
    # E. Filtered validation markers
    if status['validation_cluster'] not in ('failed', 'not run'):
        axs[0, 1].scatter(validation_cluster[validation_marker_epoch]['timestamp'][validation_marker_epoch] / 60, 
                          np.ones_like(validation_cluster[validation_marker_epoch]['timestamp'][validation_marker_epoch]) * 5,
                          c=val_color, alpha=0.05,
                         )
    
    axs[0, 1].set_ylim([0, 7])
    axs[0, 1].set_xlim([0, max_time_seconds / 60])
    axs[0, 1].set_yticks(range(1, 7))
    axs[0, 1].set_yticklabels(['$Pup_R$', '$Pup_L$',
                               '$C_{filt}$','$C_{all}$', 
                               '$V_{filt}$','$V_{all}$'])
    axs[0, 1].patch.set_color((0.92, 0.92, 0.92))
    axs[0, 1].set_title("Timeline", **font_kw)
    a = 1
    
    # % gaze kept plot
    if status['pupil'] not in ('failed', 'not run'):
        bins = np.linspace(-0.1, 1.1, 101)
        if status['pupil']['left'] not in ('failed', 'not run'):
            plot_utils.histline(pupil['left']['confidence'], bins=bins,
                                ax=axs[0, 3], color='steelblue')
            lpct = np.mean(pupil['left']['confidence'] > pupil_confidence_threshold) * 100
        else:
            lpct = 0
        if status['pupil']['right'] not in ('failed', 'not run'):
            plot_utils.histline(pupil['right']['confidence'], bins=bins,
                                ax=axs[0, 3], color='orange')
            rpct = np.mean(pupil['right']['confidence'] > pupil_confidence_threshold) * 100
        else:
            rpct = 0
        yl = axs[0, 3].get_ylim()
        axs[0, 3].vlines(pupil_confidence_threshold, yl[0], yl[1]*0.6, ls='--', lw=2, color='k')
        axs[0, 3].text(pupil_confidence_threshold, yl[1]*0.8, 'L: %0.1f%% kept\nR: %0.1f%%kept'%(
                lpct, rpct), ha='center', va='center', **font_kw)
        axs[0, 3].set_yticks([])
        axs[0, 3].set_xticks([0, 0.5, 1])
        axs[0, 3].set_title('Pupil Confidence', **font_kw)

    # Clear data to save memory
    if do_memory_cleanup:
        for d in [calibration_marker, 
                  calibration_cluster,
                  validation_marker,
                  validation_cluster,
                 ]:
            if (d is not None) and not (isinstance(d, dict)):
                d._data = None
        for d in [calibration,
                  pupil,
                  gaze,
                  error,
                 ]:
            if d is not None:
                for lr in ['left', 'right']:
                    if (d[lr] is not None) and not (isinstance(d[lr], dict)):
                        d[lr]._data = None
    if fpath is not None:
        fig.savefig(fpath, dpi=100)
    if close_figure:
        plt.close(fig)




def gaze_rect(gaze_position, hdim, vdim, ax=None, linewidth=1, edgecolor='r', **kwargs):
    """Draw an unfilled rectangle centered on a gaze position

    Parameters
    ----------
    gaze_position : array-like
        (x, y) center of the rectangle, in axis units (e.g. normalized 0-1)
    hdim : float
        rectangle width, in the same units
    vdim : float
        rectangle height, in the same units
    ax : matplotlib axis, optional
        axis into which to plot, by default None (current axis)
    linewidth : float, optional
        edge line width, by default 1
    edgecolor : matplotlib colorspec, optional
        edge color, by default 'r'
    **kwargs
        passed to `matplotlib.patches.Rectangle`

    Returns
    -------
    matplotlib.patches.Rectangle
        handle for the added rectangle
    """
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

'''
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
    pupil_left = pipeline_elements['pupil']['left']
    pupil_right = pipeline_elements['pupil']['right']
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
            gaze[k] = pipeline_elements['gaze'][k]
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
'''

"""
def make_gaze_animation(session,
                        time_idx=None,
                        frame_idx=None,
                        crop_size=(384, 384),
                        fps=30,
                        hspace=0.1,
                        wspace=None,
                        eye='left',
                        **pipeline_kw):
    ""Make radical gaze animation. 
    Animations can be displayed in jupyter notebooks, but this only works for short videos
    because it's very demanding on memory (all video frames must be loaded, so the length
    of the animation you can create depends on how much RAM your copmuter has. Keep it short!
    
    For longer videos, see `render_gaze_video()`
    ""
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


def render_gaze_video(session,
                     start_time,
                     end_time,
                     sname,
                     elements_to_render=(
                         'pupil', 'gaze', 'calibration_cluster'),
                     gaze_to_show=('left', 'right'),
                     accumulate=False,
                     tmp_dir=None,
                     axs=None,
                     figscale=6,
                     frame_step=1,
                     dpi=100,
                     cleanup_files=True,
                     progress_bar=None,
                     **pipeline):
    ""Render video of world camera and eyes
    
    
    ""
    # Remove this dependency
    import vedb_store
    import file_io
    import pathlib
    import cv2
    if progress_bar is None:
        progress_bar = lambda x, total=None: x
    #pipeline = load_pipeline_elements(
    #    session, dbi=dbi, **pipeline_kw)

    # Colors
    eye_left_color = (1.0, 0.5, 0.0)  # orange
    eye_right_color = (0.0, 0.5, 1.0)  # cyan
    cal_mk_color = (0.3, 0.9, 0.3)
    val_mk_color = (0.9, 0.3, 0.3)
    error_cmap = plt.cm.inferno_r
    error_vmin = 0
    error_vmax = 10
    # Manage time
    start_frame, end_frame = vedb_store.utils.get_frame_indices(
        start_time, end_time, session.world_time)
    # Account for possibility that whole eye video was not analyzed to find pupils
    start_frame_eye_left_vid, _ = vedb_store.utils.get_frame_indices(
        start_time, end_time, session.get_video_time('eye_left') - session.start_time)
    start_frame_eye_left, _ = vedb_store.utils.get_frame_indices(
        start_time, end_time, _get_timestamp(pipeline['pupil']['left'], session))
    start_frame_eye_right_vid, _ = vedb_store.utils.get_frame_indices(
        start_time, end_time, session.get_video_time('eye_right') - session.start_time)
    start_frame_eye_right, _ = vedb_store.utils.get_frame_indices(
        start_time, end_time, _get_timestamp(pipeline['pupil']['right'], session))
    # Manage where to save
    if tmp_dir is None:
        tmp_dir = os.path.expanduser(
            '~/Desktop/test_render_eye/')  # TEMP make me a tmpdir
    sdir = pathlib.Path(tmp_dir)
    # Set up axes
    if axs is None:
        ax_world, ax_left, ax_right = make_world_eye_axes(
            fig_scale=figscale)
    else:
        ax_world, ax_left, ax_right = axs
    fig = ax_world.figure
    # Get videos & timestamps from session
    world_vid = session.get_video_handle('world_camera')
    eye_left_vid = session.get_video_handle('eye_left')
    eye_right_vid = session.get_video_handle('eye_right')
    world_time = session.get_video_time('world_camera')
    eye_left_time = session.get_video_time('eye_left')
    eye_right_time = session.get_video_time('eye_right')
    # Load first images, establish plots
    world_vid.VideoObj.set(1, start_frame)
    success, world_im = world_vid.VideoObj.read()
    eye_left_vid.VideoObj.set(1, start_frame_eye_left_vid)
    success, eye_left = eye_left_vid.VideoObj.read()
    eye_right_vid.VideoObj.set(1, start_frame_eye_right_vid)
    success, eye_right = eye_right_vid.VideoObj.read()
    wim = ax_world.imshow(cv2.cvtColor(world_im, cv2.COLOR_BGR2RGB), extent=[
                          0, 1, 1, 0], aspect='auto')
    ax_world.set_xticks([])
    ax_world.set_yticks([])
    elim = ax_left.imshow(eye_left, extent=[0, 1, 1, 0])
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    erim = ax_right.imshow(eye_right, extent=[0, 1, 1, 0])
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    # Plot extra elements if desired
    if 'pupil' in elements_to_render:
        tmp = _use_data(pipeline['pupil']['left'])['ellipse'][start_frame_eye_left]
        ellipse_data_left = dict((k, np.array(v) / 400)
                                 for k, v in tmp.items())
        pupil_left = show_ellipse(ellipse_data_left,
                                  center_color=eye_left_color,
                                  facecolor=eye_left_color +
                                  (0.5,),
                                  ax=ax_left)
        tmp = _use_data(pipeline['pupil']['right'])['ellipse'][start_frame_eye_right]
        ellipse_data_right = dict((k, np.array(v) / 400)
                                  for k, v in tmp.items())
        pupil_right = show_ellipse(ellipse_data_right,
                                   center_color=eye_right_color,
                                   facecolor=eye_right_color +
                                   (0.5,),
                                   ax=ax_right)
    if 'gaze' in elements_to_render:
        # Accumulate w gaze hist, set max beforehand?
        gaze_left_data = _use_data(pipeline['gaze']['left'])['norm_pos']
        gaze_right_data = _use_data(pipeline['gaze']['right'])['norm_pos']
        if 'average' in gaze_to_show:
            gaze_avg_0 = (gaze_left_data[0] + gaze_right_data[0]) / 2
            gaze_avg = ax_world.scatter(
                *gaze_avg_0, color=(1.0, 1.0, 1.0, 1.0))
        if 'left' in gaze_to_show:
            gaze_left = ax_world.scatter(*gaze_left_data[0],
                                         color=eye_left_color,
                                         zorder=10,
                                         )
        if 'right' in gaze_to_show:
            gaze_right = ax_world.scatter(*gaze_right_data[0],
                                          color=eye_right_color,
                                          zorder=10,
                                          )
        ax_world.axis([0, 1, 1, 0])
    if 'calibration_cluster' in elements_to_render:
        calib_marker_pos = _use_data(pipeline['calibration_cluster'])['norm_pos']
        calib_marker_time = _use_data(pipeline['calibration_cluster'])['timestamp']
        calib_mk = ax_world.scatter(*calib_marker_pos.T, color=cal_mk_color)
        calib_mk.set_offsets(calib_marker_pos[:0])
    if 'validation_cluster' in elements_to_render:
        # Change variable names...
        val_marker_pos = [_use_data(x)['norm_pos']
                          for x in pipeline['validation_cluster']]
        val_marker_times = [_use_data(x)['timestamp']
                            for x in pipeline['validation_cluster']]
        val_mk = [ax_world.scatter(*vm_pos.T,
                                   c=val_mk_color)
                  for j, vm_pos in enumerate(val_marker_pos)]
        for v_mk, vm_pos in zip(val_mk, val_marker_pos):
            v_mk.set_offsets(vm_pos[:0])

    if 'error' in elements_to_render:
        # Change variable names...
        vm_pos_all_for_err = _use_data(pipeline['validation_marker'])['norm_pos']
        vm_time_all_for_err = _use_data(pipeline['validation_marker'])['timestamp'].tolist(
        )
        err_marker_pos_left = [_use_data(x)['marker']
                               for x in pipeline['error']['left']]
        err_gaze_pos_left = [_use_data(x)['gaze_matched']
                             for x in pipeline['error']['left']]
        err_marker_pos_right = [_use_data(x)['marker']
                                for x in pipeline['error']['right']]
        err_gaze_pos_right = [_use_data(x)['gaze_matched']
                              for x in pipeline['error']['right']]
        nrm_err = Normalize(vmin=error_vmin, vmax=error_vmax, clip=True)
        ecol_left = [_use_data(x)['gaze_err'] for x in pipeline['error']['left']]
        ecol_right = [_use_data(x)['gaze_err'] for x in pipeline['error']['right']]
        err_mk_left = [ax_world.scatter(*e_pos.T,
                                        c=ecol_left[j],
                                        cmap=error_cmap,
                                        vmin=error_vmin,
                                        vmax=error_vmax)
                       for j, e_pos in enumerate(err_marker_pos_left)]
        err_mk_right = [ax_world.scatter(*e_pos.T,
                                         c=ecol_right[j],
                                         cmap=error_cmap,
                                         vmin=error_vmin,
                                         vmax=error_vmax)
                        for j, e_pos in enumerate(err_marker_pos_right)]
        for v_mkl, vm_posl in zip(err_mk_left, err_marker_pos_left):
            v_mkl.set_offsets(vm_posl[:0])
        for v_mkr, vm_posr in zip(err_mk_right, err_marker_pos_right):
            v_mkr.set_offsets(vm_posr[:0])
        # Draw connector line
        line_h_left = ax_world.plot([0, 1], [0, 1], lw=3, color='r')[0]
        line_h_left.set_data([[np.nan, np.nan], [np.nan, np.nan]])
        line_h_right = ax_world.plot([0, 1], [0, 1], lw=3, color='r')[0]
        line_h_right.set_data([[np.nan, np.nan], [np.nan, np.nan]])

    # Track indices for each; time will move faster for eye videos,
    # which are sampled at a higher rate.
    world_frame = start_frame
    eye_left_frame = start_frame_eye_left_vid
    eye_left_frame_data = start_frame_eye_left
    eye_right_frame = start_frame_eye_right_vid
    eye_right_frame_data = start_frame_eye_right
    total_iters = np.ceil((end_frame-start_frame) / frame_step)
    last_frame = copy.copy(start_frame)
    for j in progress_bar(range(start_frame, end_frame, frame_step), total=total_iters):
        # Read world video frame
        world_time_this_frame = world_time[j]
        #while last_frame <= j:
        success, world_im = world_vid.VideoObj.read()
        while eye_left_time[eye_left_frame] < world_time_this_frame:
            eye_left_frame += 1
            eye_left_frame_data += 1
            success, eye_left = eye_left_vid.VideoObj.read()
        while eye_right_time[eye_right_frame] < world_time_this_frame:
            eye_right_frame += 1
            eye_right_frame_data += 1
            success, eye_right = eye_right_vid.VideoObj.read()
        wim.set_data(cv2.cvtColor(world_im, cv2.COLOR_BGR2RGB))
        elim.set_data(eye_left)
        erim.set_data(eye_right)
        # Plot extra elements if desired
        if 'pupil' in elements_to_render:
            tmpl = _use_data(pipeline['pupil']['left'])['ellipse'][eye_left_frame_data]
            ellipse_data_left = dict((k, np.array(v) / 400)
                                     for k, v in tmpl.items())
            pupil_left[0].set_center(ellipse_data_left['center'])
            pupil_left[0].set_height(ellipse_data_left['axes'][1])
            pupil_left[0].set_width(ellipse_data_left['axes'][0])
            pupil_left[0].set_angle(tmpl['angle'])
            # Accumulate?
            pupil_left[1].set_offsets([ellipse_data_left['center']])

            tmpr = _use_data(pipeline['pupil']['right'])['ellipse'][eye_right_frame_data]
            ellipse_data_right = dict((k, np.array(v) / 400)
                                      for k, v in tmpr.items())
            pupil_right[0].set_center(ellipse_data_right['center'])
            pupil_right[0].set_height(ellipse_data_right['axes'][1])
            pupil_right[0].set_width(ellipse_data_right['axes'][0])
            pupil_right[0].set_angle(tmpr['angle'])
            # Accumulate? Either for hist, or only matched data.
            pupil_right[1].set_offsets([ellipse_data_right['center']])
        if 'gaze' in elements_to_render:
            # Accumulate? Either for hist, or only matched data.
            tmp_gl = _use_data(pipeline['gaze']['left'])['norm_pos'][eye_left_frame_data]
            tmp_gr = _use_data(pipeline['gaze']['right'])['norm_pos'][eye_right_frame_data]
            if 'average' in gaze_to_show:
                # Consider weighting this average; for now, just average
                tmp_avg = (tmp_gl + tmp_gr) / 2
                gaze_avg.set_offsets([tmp_avg])
            if 'left' in gaze_to_show:
                gaze_left.set_offsets([tmp_gl])
            if 'right' in gaze_to_show:
                gaze_right.set_offsets([tmp_gr])
        if 'calibration_cluster' in elements_to_render:
            if world_time_this_frame in calib_marker_time:
                cmi = calib_marker_time.tolist().index(world_time_this_frame)
                calib_mk.set_offsets(calib_marker_pos[:cmi])
        if 'validation_cluster' in elements_to_render:
            vm_epoch_i = np.array(
                [world_time_this_frame in vm_t for vm_t in val_marker_times])
            if np.any(vm_epoch_i):
                this_val_epoch, = np.nonzero(vm_epoch_i)
                this_val_epoch = this_val_epoch[0]
                vmi = val_marker_times[this_val_epoch].tolist().index(
                    world_time_this_frame)
                val_mk[this_val_epoch].set_offsets(
                    val_marker_pos[this_val_epoch][:vmi])
        if 'error' in elements_to_render:
            # check if time matches a validation detection timestamp
            if world_time_this_frame in vm_time_all_for_err:
                all_val_pos_i = vm_time_all_for_err.index(
                    world_time_this_frame)
                val_pos_rn = tuple(vm_pos_all_for_err[all_val_pos_i])
                # Check if this position is in any of our error marker position lists
                # Tuple-ify for indexing
                err_marker_pos_left_tup = [
                    [tuple(y) for y in x] for x in err_marker_pos_left]
                err_marker_pos_right_tup = [
                    [tuple(y) for y in x] for x in err_marker_pos_right]
                err_epoch_i_left = np.array(
                    [val_pos_rn in vm_t for vm_t in err_marker_pos_left_tup])
                err_epoch_i_right = np.array(
                    [val_pos_rn in vm_t for vm_t in err_marker_pos_right_tup])
                if 'left' in gaze_to_show:
                    if np.any(err_epoch_i_left):
                        this_err_epoch, = np.nonzero(err_epoch_i_left)
                        this_err_epoch = this_err_epoch[0]
                        emi = err_marker_pos_left_tup[this_err_epoch].index(
                            val_pos_rn)
                        err_mk_left[this_err_epoch].set_offsets(
                            err_marker_pos_left[this_err_epoch][:emi])
                        # Set data for line connecting gaze and marker
                        x0, y0 = err_marker_pos_left[this_err_epoch][emi]
                        x1, y1 = err_gaze_pos_left[this_err_epoch][emi]
                        line_h_left.set_data([[x0, x1], [y0, y1]])
                        line_h_left.set_color(error_cmap(
                            nrm_err(ecol_left[this_err_epoch][emi])))
                    else:
                        line_h_left.set_data(
                            [[np.nan, np.nan], [np.nan, np.nan]])
                if 'right' in gaze_to_show:
                    if np.any(err_epoch_i_right):
                        this_err_epoch, = np.nonzero(err_epoch_i_right)
                        this_err_epoch = this_err_epoch[0]
                        emi = err_marker_pos_right_tup[this_err_epoch].index(
                            val_pos_rn)
                        err_mk_right[this_err_epoch].set_offsets(
                            err_marker_pos_right[this_err_epoch][:emi])
                        # Set data for line connecting gaze and marker
                        x0, y0 = err_marker_pos_right[this_err_epoch][emi]
                        x1, y1 = err_gaze_pos_right[this_err_epoch][emi]
                        line_h_right.set_data([[x0, x1], [y0, y1]])
                        line_h_right.set_color(error_cmap(
                            nrm_err(ecol_right[this_err_epoch][emi])))
                    else:
                        line_h_right.set_data(
                            [[np.nan, np.nan], [np.nan, np.nan]])
        ax_world.axis([0, 1, 1, 0])
        fig.savefig(sdir / (f'fr{j:07d}.png'), dpi=dpi)
    world_vid.VideoObj.release()
    eye_left_vid.VideoObj.release()
    eye_right_vid.VideoObj.release()
    files = sorted(sdir.glob('*png'))
    fps = int(
        np.round(1/np.mean(np.diff(world_time[start_frame:end_frame]))))
    print(fps)
    file_io.write_movie_from_frames(files, sname, fps=fps,
                                    progress_bar=progress_bar)
    if cleanup_files:
        for f in files:
            f.unlink()
"""

def make_gaze_animation(paths,
                        rect_size=(600,600),
                        start_time=0,
                        end_time=5,
                        # pupil_str = 'pupil_detection-%s-pylids_eyelids_pupils_v2.npz',
                        # gaze_str = 'gaze-%s-default_mapper-monocular_tps_cv_cluster_median_conf75_cut3std-c0eb1e655a0f58e7905b.npz',
                        fps=30, 
                        world_size_factor=0.25,
                        hspace=0.1,
                        wspace=None,
                        eye_left_color=(1.0, 0.5, 0.0),  # orange
                        eye_right_color=(0.0, 0.5, 1.0),  # cyan
                        raise_error=False,
                    ):
    """Make radical gaze animation

    paths must contain:
    pupil_left
    pupil_right
    gaze_left
    gaze_right
    world
    world_timestamps
    eye_left
    eye_right
    eye_timestamps_left
    eye_timestamps_right

    Intended layout: left eye, right eye and gaze-centered world crop on
    top, full world video with gaze rectangles / dots below.

    Parameters
    ----------
    paths : dict
        file paths (pathlib.Path for pupil / gaze .npz files, which are
        checked with `.exists()`); 'eye_left', 'eye_right' and 'world'
        appear to be indexable with the file path at index 1
    rect_size : tuple, optional
        size (pixels) of the gaze-centered crop / rectangle, by default
        (600, 600)
    start_time, end_time : float, optional
        unused, by default 0 and 5
    fps : float, optional
        animation frame rate, by default 30
    world_size_factor : float, optional
        resize factor for world video, by default 0.25
    hspace, wspace : float, optional
        GridSpec spacing, by default 0.1 and None
    eye_left_color : tuple, optional
        RGB color for left-eye gaze / pupil, by default (1.0, 0.5, 0.0)
    eye_right_color : tuple, optional
        RGB color for right-eye gaze / pupil, by default (0.0, 0.5, 1.0)
    raise_error : bool, optional
        unused, by default False

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation

    Notes
    -----
    Appears to be a partially ported method: it references undefined
    names (`self`, `ses`, `get_session_info`, `utils`) and will raise a
    NameError as written. Uses module-level globals to track eye frames.
    """
    from matplotlib.gridspec import GridSpec
    global eye_left_frame
    global eye_left_image
    global eye_right_frame    
    global eye_right_image
    eye_video_size = 400 # x 400, square
    si = get_session_info(self.session)
    
    try:
        pl = paths['pupil_left']
        gl = paths['gaze_left']
        pr = paths['pupil_right']
        gr = paths['gaze_right']
        include_left_eye = 'eye_left' in paths
        include_right_eye = 'eye_right' in paths
        
        if include_left_eye:
            eye_left_frame = 0
            eye_left_time = np.load(paths['eye_timestamps_left'])
            eye_left_vid = file_io.VideoCapture(paths['eye_left'][1])
        if include_right_eye:
            eye_right_frame = 0
            eye_right_time = np.load(paths['eye_timestamps_right'])
            eye_right_vid = file_io.VideoCapture(paths['eye_right'][1])
        
        _, vh, vw, _ = file_io.list_array_shapes(paths['world'][1])
        n_frames = len(self(dict(timestamp=ses.world_time))['timestamp'])
        #frame = world[0]
        rect_width = rect_size[0] / vw
        rect_height = rect_size[1] / vh
        ar = vw / vh

        fig = plt.figure(figsize=(8, 8 * 13.5/12)) #* 14 / 12))
        gs = GridSpec(2,3, figure=fig,  hspace=hspace, wspace=wspace,
                    height_ratios=[1, 2], width_ratios=[1, 1, 1])
        ax_eye_left = fig.add_subplot(gs[0,0])
        ax_eye_right = fig.add_subplot(gs[0,1])
        ax_gc = fig.add_subplot(gs[0, 2])
        ax_vid = fig.add_subplot(gs[1,:])
        ax_vid.axis([0, 1, 1, 0])
        ax_vid.set_xticks([]) # To visual field size?
        ax_vid.set_yticks([])

        # Initialize all plots
        if gl.exists():
            gaze_left = dict(np.load(gl))
            gaze_rect_pixel_size = rect_size
            # FIX ME
            wt = ses.world_time[self.binary(ses.world_time)]
            g_matched = utils.match_time_points(dict(timestamp=wt), gaze_left)
            _, gaze_centered_video = wt, wv = self.load('world', 
                                                        center=g_matched['norm_pos'],
                                                        crop_size=gaze_rect_pixel_size)

        if gr.exists():
            gaze_right = dict(np.load(gr))
        world_time, world = self.load('world', size=world_size_factor)
        world_h = ax_vid.imshow(world[0], extent=[0, 1, 1, 0], aspect='auto')
        # For now: choose best, don't rely on left.
        if gl.exists():
            gaze_rect_lh = gaze_rect(gaze_left['norm_pos'][0], rect_width, rect_height, ax=ax_vid, linewidth=3, edgecolor=eye_left_color)
            gaze_dot_lh = ax_vid.scatter(*gaze_left['norm_pos'][0], c=eye_left_color)
            gc_h = ax_gc.imshow(gaze_centered_video[0], extent=[0, 1, 1, 0])
        else:
            gaze_rect_lh = gaze_rect([-1,-1], rect_width, rect_height, ax=ax_vid, linewidth=3)
            gaze_dot_lh = ax_vid.scatter(*[-1,-1], c='black')
            gc_h = ax_gc.imshow(np.zeros((200,200)), extent=[0, 1, 1, 0])

        if gr.exists():
            gaze_rect_rh = gaze_rect(gaze_right['norm_pos'][0], rect_width, rect_height, ax=ax_vid, linewidth=3, edgecolor=eye_right_color)
            gaze_dot_rh = ax_vid.scatter(*gaze_right['norm_pos'][0], c=eye_right_color)
            #gc_h = ax_gc.imshow(gaze_centered_video[0], extent=[0, 1, 1, 0])
        else:
            gaze_rect_rh = gaze_rect([-1,-1], rect_width, rect_height, ax=ax_vid, linewidth=3)
            gaze_dot_rh = ax_vid.scatter(*[-1,-1], c='black')
            #gc_h = ax_gc.imshow(np.zeros((200,200)), extent=[0, 1, 1, 0])

            
        if include_left_eye:
            success, eye_left_image = eye_left_vid.VideoObj.read()
            eye_left_h = ax_eye_left.imshow(eye_left_image, extent=[0, 1, 1, 0])
            
        if pl.exists():
            pupil_left = dict(np.load(pl, allow_pickle=True))
            ellipse_data_left = dict((k, np.array(v) / eye_video_size)
                                    for k, v in pupil_left['ellipse'][0].items())
            pupil_left_eh, pupil_left_dh = show_ellipse(ellipse_data_left,
                                    center_color=eye_left_color,
                                    facecolor=eye_left_color +
                                    (0.5,),
                                    ax=ax_eye_left)

        if include_right_eye:
            success, eye_right_image = eye_right_vid.VideoObj.read()
            eye_right_h = ax_eye_right.imshow(eye_right_image, extent=[0, 1, 1, 0])
            
        if pr.exists():
            pupil_right = dict(np.load(pr, allow_pickle=True))
            ellipse_data_right = dict((k, np.array(v) / eye_video_size)
                                    for k, v in pupil_right['ellipse'][0].items())
            pupil_right_eh, pupil_right_dh = show_ellipse(ellipse_data_right,
                                    center_color=eye_right_color,
                                    facecolor=eye_right_color +
                                    (0.5,),
                                    ax=ax_eye_right)
            
        ax_eye_left.axis([1, 0, 1, 0]) # [0,1, 0, 1]
        ax_eye_left.set_xticks([])
        ax_eye_left.set_yticks([])
        
        ax_eye_right.axis([0, 1, 0, 1]) #[1, 0, 1, 0]
        ax_eye_right.set_xticks([])
        ax_eye_right.set_yticks([])

        
        gaze_box_vis_degrees = si['fov'] * rect_width
        vmx_deg = gaze_box_vis_degrees / 2
        tick_labels = ['%.1f'%x for x in [-vmx_deg, 0, vmx_deg]]
        ax_gc.set_xticks([0, 0.5, 1])
        ax_gc.set_xticklabels(tick_labels)
        ax_gc.set_yticks([0, 0.5, 1])
        ax_gc.set_yticklabels(tick_labels)
        ax_gc.grid('on', linestyle=':', color=(0.95, 0.85, 0))
        #plt.close(fig)
        

        def init():
            """Reset world, gaze, eye and pupil artists to initial state."""
            to_return = [world_h]
            world_h.set_array(np.zeros_like(world[0]))
            
            if gl.exists():
                gaze_rect_lh.set_xy([0.5, 0.5] - np.array([rect_width/2, rect_height/2]))
                gaze_dot_lh.set_offsets([0.5, 0.5])
                gc_h.set_array(np.zeros_like(gaze_centered_video[0]))
                to_return.extend([gaze_rect_lh, gaze_dot_lh, gc_h])
            
            if include_left_eye:
                eye_left_h.set_array(np.zeros((400,400), dtype=np.uint8))
                to_return.append(eye_left_h)
                
            if pl.exists():
                _ = _set_ellipse(pupil_left_eh, pupil_left_dh,
                                 pupil_left['ellipse'], 0, eye_video_size=eye_video_size)
                to_return.extend([pupil_left_eh, pupil_left_dh])
                
            if include_right_eye:
                eye_right_h.set_array(np.zeros((400,400), dtype=np.uint8)) #np.zeros_like(eye_right_ds[0]))
                to_return.append(eye_right_h)
                
            if pr.exists():
                _ = _set_ellipse(pupil_right_eh, pupil_right_dh,
                                 pupil_right['ellipse'], 0, eye_video_size=eye_video_size)
                to_return.extend([pupil_right_eh, pupil_right_dh])
            
            return to_return

        def animate(i):
            """Draw world frame `i`, advancing eye videos to match its time."""
            global eye_left_frame
            global eye_right_frame
            global eye_right_image
            global eye_left_image
            #success, world_im = world_vid.VideoObj.read()
            world_h.set_data(world[i]) # world_im)
            world_time_this_frame = world_time[i]
            to_return = [world_h]
            
            
            if include_left_eye:
                while eye_left_time[eye_left_frame] < world_time_this_frame:
                    eye_left_frame += 1
                    success, eye_left_image = eye_left_vid.VideoObj.read()
                eye_left_h.set_data(eye_left_image)
                to_return.append(eye_left_h)
            if pl.exists():
                _ = _set_ellipse(pupil_left_eh, pupil_left_dh,
                                 pupil_left['ellipse'], eye_left_frame, eye_video_size=eye_video_size)
                to_return.extend([pupil_left_eh, pupil_left_dh])
            if include_right_eye:
                while eye_right_time[eye_right_frame] < world_time_this_frame:
                    eye_right_frame += 1
                    success, eye_right_image = eye_right_vid.VideoObj.read()
                eye_right_h.set_data(eye_right_image)
                to_return.append(eye_right_h)
            if pr.exists():
                _ = _set_ellipse(pupil_right_eh, pupil_right_dh,
                                 pupil_right['ellipse'], eye_right_frame, eye_video_size=eye_video_size)
                to_return.extend([pupil_right_eh, pupil_right_dh])

            if gl.exists():
                gaze_rect_lh.set_xy(gaze_left['norm_pos'][eye_left_frame] - np.array([rect_width/2, rect_height/2]))
                gaze_dot_lh.set_offsets(gaze_left['norm_pos'][eye_left_frame])
                try:
                    gc_h.set_data(gaze_centered_video[i])
                except:
                    pass
                to_return.extend([gaze_rect_lh, gaze_dot_lh, gc_h])
                
            if gr.exists():
                gaze_rect_rh.set_xy(gaze_right['norm_pos'][eye_right_frame] - np.array([rect_width/2, rect_height/2]))
                gaze_dot_rh.set_offsets(gaze_right['norm_pos'][eye_right_frame])
                to_return.extend([gaze_rect_rh, gaze_dot_rh])

            
            return to_return



        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1/fps * 1000, blit=True)
    except:
        eye_left_vid.VideoObj.release()
        eye_right_vid.VideoObj.release()
        raise
    return anim


def show_session(folder, 
                n_frames=4, 
                pct_start=0.10, 
                pct_fin=0.90, 
                sc=2, 
                axs=None):
    """Quickie visualization of frames from a session

    Shows `n_frames` world frames at evenly spaced times between
    `pct_start` and `pct_fin` of the recording duration.

    Parameters
    ----------
    folder : pathlib.Path
        session folder containing 'worldPrivate.mp4' and world timestamps
    n_frames : int, optional
        number of frames to show, by default 4
    pct_start : float, optional
        start of sampled range as a fraction of recording duration,
        by default 0.10
    pct_fin : float, optional
        end of sampled range as a fraction of recording duration,
        by default 0.90
    sc : float, optional
        per-subplot size scale (inches), by default 2
    axs : array of matplotlib axes, optional
        axes into which to plot, by default None

    Returns
    -------
    fig : matplotlib figure

    Notes
    -----
    As written this does not run: it references undefined names
    (`recording_duration` vs. assigned `recording_duratio`, `ax`, `vmt`),
    loads timestamps from a '.mp4' filename with `np.load`, and passes
    times in seconds as frame indices (`idx`) to `file_io.load_video`.
    """
    video_file = folder / 'worldPrivate.mp4'
    time_file = folder / 'world_timestamps_0start.mp4'
    world_time = np.load(time_file)
    recording_duratio = world_time[-1] - world_time[0]
    image_size = [2048, 1536]
    ar = image_size[0] / image_size[1]
    st = int(np.floor(recording_duration * pct_start))
    fin = int(np.floor(recording_duration * pct_fin))
    #print(st, fin)
    time_points = np.linspace(st, fin, n_frames)
    #print(time_points)
    nr,nc = vmt.plot_utils.find_squarish_dimensions(n_frames)
    if ax is None:
        fig, axs = plt.subplots(nc, nr, 
                                figsize=(nr*sc * ar, nc*sc))
    else:
        fig = axs.flatten()[0].figure
    for tp, ax in zip(time_points, axs.flatten()):
        try:
            wc = file_io.load_video(video_file, idx=[tp, tp+1])
            title_add = ''
        except IndexError:
            wc = np.zeros([1] + image_size + [3])
            title_add = ' (paused or missing)'
        ax.imshow(wc[0])
        ax.axis('off')
        ax.set_title('%d seconds%s'%(tp, title_add))
    plt.tight_layout()
    return fig


def background_fill_blocks(onoff, fcol=(.9, .9, .9), zorder=-1, ylim=None, ax=None):
    """Shade between each onset and offset in `onoff`

    Parameters
    ----------
    onoff : array-like
        (n, 2) [onset, offset] pairs in x-axis units
    fcol : matplotlib colorspec, optional
        fill color, by default (.9, .9, .9)
    zorder : float, optional
        z-order of the fills, by default -1 (behind other artists)
    ylim : tuple, optional
        vertical extent of the fills, by default None (current y limits)
    ax : matplotlib axis, optional
        axis into which to plot, by default None (current axis)

    Returns
    -------
    h : list of matplotlib Polygon or None
        handle from the last `ax.fill` call (None if `onoff` is empty)
    """
    if ax is None:
        ax = plt.gca()
    if ylim is None:
        ylim = ax.get_ylim()
    h = None
    for xf in onoff:
        h = ax.fill(np.array([xf[0], xf[1], xf[1], xf[0]]),
            [ylim[0], ylim[0], ylim[1], ylim[1]],
            color=fcol, edgecolor='none', zorder=zorder)
    ax.set_ylim(ylim)
    # Returns only one instance, not all, should be good for 
    # legend but not setting new properties
    return h


# Pylids video overlay
def label_eye_video(eye_video_file, eye_data, st=0, fin=None, eye_alpha=0.2,eye_color=(1, 0, 1), eyelid_data=None, eye_timestamp_file=None, figsize=(5, 5)):
    """Animate pupil ellipses overlaid on an eye video

    Assumes eye_video_file and eye_data have same timestamps

    st and fin are times in seconds starting from 0 (at start of video)

    Parameters
    ----------
    eye_video_file : str or pathlib.Path
        eye video file, opened with `file_io.load_mp4`
    eye_data : dict
        pupil detection output with an 'ellipse' field (per-frame ellipse
        dicts in pixels) and, if `eye_timestamp_file` is None, a
        'timestamps' field
    st : float, optional
        start time (seconds), by default 0
    fin : float, optional
        end time (seconds), by default None (last timestamp)
    eye_alpha : float, optional
        opacity of the ellipse fill, by default 0.2
    eye_color : tuple, optional
        RGB color of the ellipse and center dot, by default (1, 0, 1)
    eyelid_data : optional
        unused (eyelid plotting not implemented), by default None
    eye_timestamp_file : str or pathlib.Path, optional
        .npy file of eye timestamps, by default None
    figsize : tuple, optional
        figure size, by default (5, 5)

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation

    Notes
    -----
    As written this does not run: `ax`, `ellipse_data`, `n_frames` and
    `fps` are undefined. Ellipses are normalized by `eye_image.shape[:2]`,
    which is (rows, cols) rather than (width, height).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    # Load time
    if eye_timestamp_file is None:
        timestamps = eye_data['timestamps']
    else:
        timestamps = np.load(eye_timestamp_file)
    if fin is None:
        fin = timestamps[-1]
    # Get frame 
    ti = (timestamps >= st) & (timestamps < fin)
    frame_i, = np.nonzero(ti)
    st_frame = frame_i[0]
    fin_frame = frame_i[-1]
    # Get video object
    eye_vid = file_io.load_mp4(eye_video_file)
    eye_vid.set_frame(st_frame)
    success, eye_image = eye_vid.VideoObj.read()
    eye_vid.set_frame(st_frame)
    # Set up plot
    imh = ax.imshow(eye_image, extent=[0, 1, 1, 0])
    ell_h, pt_h = show_ellipse(ellipse_data,
                            center_color=eye_color + (1.0,),
                            facecolor=eye_color + (eye_alpha,),
                            ax=ax)
    def init():
        """Blank the eye image and set the ellipse to the start frame."""
        imh.set_array(np.zeros_like(eye_image))
        _ = _set_ellipse(ell_h, pt_h, eye_data['ellipse'], st_frame, eye_video_size=eye_image.shape[:2])
        to_return = [imh, ell_h, pt_h]
        # if eyelid_data is not None:
        #    # Initialize eyelid plots
        return to_return
    
    def animate(i):
        """Read the next eye frame and update the ellipse for frame `i`."""
        # Update
        success, eye_image = eye_vid.VideoObj.read()
        assert success, "Could not read eye video frame"
        imh.set_data(eye_image)
        _ = _set_ellipse(ell_h, pt_h, eye_data['ellipse'], frame_i[i], eye_video_size=eye_image.shape[:2])
        to_return = [imh, ell_h, pt_h]
        # if eyelid_data is not None:
        #    # Update eyelid plots
        return to_return
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1/fps * 1000, blit=True)
    return anim

def plot_at_times(tt, y, time_start, time_end, 
                  time_units='seconds', ax=None, **kwargs):
    """Plot `y` against time `tt` within a time window

    Parameters
    ----------
    tt : array
        timestamps (seconds)
    y : array
        values to plot, same length as `tt`
    time_start : float
        start of window, in `time_units`
    time_end : float
        end of window, in `time_units`
    time_units : str, optional
        'seconds' / 's' or 'minutes' / 'm'; the x axis is plotted in these
        units, by default 'seconds'
    ax : matplotlib axis, optional
        axis into which to plot, by default None, which creates a new figure
    **kwargs
        passed to `ax.plot`

    Notes
    -----
    Uses `vedb_store.utils.get_frame_indices`, but `vedb_store` is not
    imported in this module, so this raises a NameError as written. Any
    other `time_units` value leaves `multiplier` undefined.
    """
    if ax is None:
        _, ax = plt.subplots()
    if time_units in ('seconds', 's'):
        multiplier = 1
    elif time_units in ('minutes', 'm'):
        multiplier = 60
    st, fin = vedb_store.utils.get_frame_indices(time_start * multiplier, time_end * multiplier, tt)
    ax.plot(tt[st:fin] / multiplier, y[st:fin], **kwargs)



