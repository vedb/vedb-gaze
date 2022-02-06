import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches, colors


def gaze_hist(gaze, 
    confidence_threshold=0, 
    cmap='gray_r',
    hist_bins_x=81, 
    hist_bins_y=61, 
    field='position', 
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
        name of field within gaze dict to plot, by default 'position' 
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

    # Inputs
    if np.ndim(dot_locations) == 2:
        dot_locations = dot_locations[np.newaxis, :]
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
    if np.max(dot_locations) > 1:
        dot_locations /= np.array([x, y])
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
        print(vframes)
        n_dots_ds = len(vframes)
        print(n_dots_ds)
        dot_locations_ds = np.hstack(
            [
                np.median(dot_locations[:, t_i[vframe_i == j]], axis=1)[:, None, :]
                for j in vframes
            ]
        )
        # print(dot_locations_ds.shape)
    else:
        dot_locations_ds = dot_locations
        vframes = np.arange(n_frames)
    # Plotting args
    n_dots, n_dot_frames, _ = dot_locations_ds.shape
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
    def update_func(i, artists, dot_locations, vframes):

        artists[0].set_array(video_data[i])
        # Also needs attention if different timecourses for different dots
        for j, artist in enumerate(artists[1:]):
            # may end up being: if i in vframes[j]:
            if i in vframes:
                dot_i = vframes.tolist().index(i)
                _ = artist.set_offsets(dot_locations[j, dot_i])
            else:
                _ = artist.set_offsets([-1, -1])
        return artists

    init = partial(init_func, fig=fig, ax=ax, artists=artists)
    update = partial(
        update_func, artists=artists, dot_locations=dot_locations_ds, vframes=vframes
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
    image_cmap,
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
