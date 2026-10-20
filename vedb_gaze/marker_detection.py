"""Detection of calibration and validation markers in world-camera video.

Provides `find_concentric_circles` (calibration markers, via the Pupil Labs
circle detector) and `find_checkerboard` (validation markers, via OpenCV).
These are the functions named by ``config/marker-<tag>.yaml`` and run by the
`pipelines.marker_detection` step. Both return dicts of arrays with
'timestamp', 'location' (pixels in the full-size world video) and 'norm_pos'
(normalized 0-1 image coordinates).
"""
from .utils import dictlist_to_arraydict
from .externals.circle_detector import find_pupil_circle_marker
import numpy as np
import file_io
import cv2
import  copy

import multiprocessing
from multiprocessing import Pool, set_start_method, get_context
#set_start_method("spawn")

from itertools import repeat

def _opencv_ellipse_to_dict(ellipse_dict):
    """Convert an ellipse detection to a Pupil Labs-style dict. Unused, and apparently unfinished: builds `data` but returns None."""
    data = {}
    data["ellipse"] = {
        "center": (ellipse_dict.ellipse.center[0], ellipse_dict.ellipse.center[1]),
        "axes": (
            ellipse_dict["ellipses"].minor_radius * 2.0,
            ellipse_dict.ellipse.major_radius * 2.0,
        ),
        "angle": ellipse_dict.ellipse.angle * 180.0 / np.pi - 90.0,
    }
    data["diameter"] = max(data["ellipse"]["axes"])
    data["location"] = data["ellipse"]["center"]
    data["confidence"] = ellipse_dict.confidence


def _find_circles_frame(args):
    """Run a single frame of calibration marker (concentric circle) detection"""
    video_data, timestamp, scale, (vdim, hdim) = args
    world_circles = find_pupil_circle_marker(video_data, 1.0)
    output_dicts = []
    for iw, w in enumerate(world_circles):
        ellipses = w["ellipses"]
        this_circle = {}
        ellipse_centers = np.array([e[0] for e in ellipses]) / scale
        this_circle["location"] = ellipse_centers.mean(0).tolist()
        this_circle["norm_pos"] = this_circle["location"] / np.array(
            [hdim, vdim]
        )
        this_circle["norm_pos"] = this_circle["norm_pos"].tolist()
        ellipse_radii = np.array([e[1] for e in ellipses])
        this_circle["size"] = ellipse_radii.max(0)
        this_circle["timestamp"] = timestamp
        output_dicts.append(this_circle)
    return output_dicts


def find_concentric_circles(video_file,
        timestamp_file,
        scale=0.5,
        n_cores=12,
        start_frame=None,
        end_frame=None,
        batch_size=None,
        progress_bar=None):
    """Use PupilLabs circle detector to find concentric circles

    Frames are loaded in batches as grayscale, resized by `scale`, and
    processed frame-by-frame (in parallel if `n_cores` is not None).

    Parameters
    ----------
    video_file : str
        world video file to parse for concentric circle markers
    timestamp_file : str
        .npy timestamp file to accompany video file, with one timestamp per frame
    scale : float, optional
        factor by which to resize video frames before detection, by default 0.5.
        Output 'location' and 'norm_pos' are converted back to full-size coordinates.
    n_cores : int or None, optional
        number of processes for parallel detection (capped at the CPU count),
        by default 12; None runs serially
    start_frame : int, optional
        first frame to parse, by default None (start of video)
    end_frame : int, optional
        frame at which to stop parsing (exclusive), by default None (end of video)
    batch_size : int, optional
        number of frames to load at once, by default None, which uses a chunk
        of video that is ~4 GB in size
    progress_bar : callable, optional
        e.g. `tqdm.tqdm`, used to display progress through each batch, by
        default None (no progress bar)

    Returns
    -------
    circles : dict of arrays
        one entry per detected marker (possibly several per frame), with fields
        'location' : (n, 2) marker center in full-size pixels (x, y);
        'norm_pos' : (n, 2) marker center normalized 0-1 by video width, height;
        'size' : (n, 2) max ellipse axes across the detected concentric
        ellipses (in pixels of the resized image);
        'timestamp' : (n,) frame time.
        If no markers are found, all fields are empty arrays.
    """
    if progress_bar is None:
        def progress_bar(x, total=0):
            """Identity stand-in used when no progress bar is given."""
            return x

    timestamps = np.load(timestamp_file)

    # termination criteria: Make inputs?
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    use_multiprocessing = n_cores is not None
    n_frames_total, vdim, hdim, _ = file_io. list_array_shapes(video_file)
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = n_frames_total
    if batch_size is None:
        # This variable might be better as an input
        max_batch_bytes = 1024**3 * 4  # 4 GB
        n_bytes = (vdim * scale) * (hdim * scale)
        batch_size = int(np.floor(max_batch_bytes / n_bytes))

    n_frames = end_frame - start_frame
    n_batches = int(np.ceil(n_frames / batch_size))
    output_dicts = []

    for batch in range(n_batches):
        print("Running batch %d/%d" % (batch+1, n_batches))
        batch_start = batch * batch_size + start_frame
        batch_end = np.minimum(batch_start + batch_size, end_frame)
        print("Loading batch of %d frames..." % (batch_end-batch_start))
        video_data = file_io.load_video(
            video_file,
            frames=(batch_start, batch_end),
            size=scale,
            color='gray')
        n_frames = batch_end - batch_start
        assert video_data.shape[0] == n_frames, 'Frame number error'
        assert len(timestamps[batch_start:batch_end]
                   ) == n_frames, 'Timestamp / frame number mismatch'
        # Args must be concatenated like this for use with parallel processing & progress bar
        # Look up "istarmap" for a pool object for slightly less clunky syntax, but use of e.g.
        # () requires python 3.8+, which I'd rather not commit to yet
        zz = zip(video_data,
                 timestamps[batch_start:batch_end],
                 repeat(scale),
                 repeat((vdim, hdim)),
                 )
        if use_multiprocessing:
            if n_cores > multiprocessing.cpu_count():
                n_cores = multiprocessing.cpu_count()
            # Parallel call to framewise calculation of checkerboard position
            with Pool(n_cores) as p:  # get_context('spawn').Pool(n_cores) as p: #
                tmp_out = list(progress_bar(
                    p.imap(_find_circles_frame, zz), total=n_frames))
            # Filter output to have no empty dicts in list
            for x in tmp_out:
                if x: 
                    output_dicts.extend(x)
        else:
            # Run serially (easier to debug if anything goes wrong)
            for args in progress_bar(zz, total=n_frames):
                tmp = _find_circles_frame(args)
                if tmp:
                    output_dicts.extend(tmp)
    if len(output_dicts) == 0:
        out = dict(location=np.array([]),
                   norm_pos=np.array([]),
                   timestamp=np.array([]),
                   size=np.array([]),
                   )
    else:
        out = dictlist_to_arraydict(output_dicts)
    return out


def _find_checkerboard_frame(args):
    """Find chessboard and compute corner locations within chessboard for one frame. 
    
    Parameters
    ----------
    args: tuple
        of: (video_data, timestamp, scale, (vdim, hdim), computed_refinement_window_size, 
        checkerboard_size) - see `find_checkerboard` below for more sensible inputs.

    Notes
    -----
    This is a helper function, meant to facilitate multi-threaded calls to opencv 
    functions `cv2.findChessboardCorners()` and `cv2.cornerSubPix()`, which do not 
    seem to be natively parallel. The cumbersome input syntax is related to the 
    use of multiprocessing module in conjunction with tqdm for progress bars. 
    """
    # Find the chess board corners
    video_data, timestamp, scale, (vdim, hdim), refinement_window_size, checkerboard_size, detection_flags = args
    return find_checkerboard_frame(video_data, timestamp, 
                            scale=scale,
                            vdim=vdim,
                            hdim=hdim,
                            refinement_window_size=refinement_window_size,
                            checkerboard_size=checkerboard_size,
                            detection_flags=detection_flags)


def find_checkerboard_frame(frame, 
                            timestamp=0.0,
                            scale=0.5,
                            vdim=1536,
                            hdim=2048,
                            refinement_window_size=(11,11),
                            checkerboard_size=(3, 6),
                            detection_flags=11):
    """Find checkerboard marker in one image array.

    Image array should be grayscale (uint8, 0-255, as required by
    `cv2.cornerSubPix`).

    Parameters
    ----------
    frame : array
        2D image array, grayscale
    timestamp : float, optional
        time point for frame, just returned in output, by default 0.0
    scale : scalar, optional
        scale factor for image (i.e. by what factor this image was resized 
        from original video), by default 0.5. This scale factor is necessary,
        along with vdim and hdim, if the input to this function is less
        than original size, in order to translate checkerboard location back
        to pixel coordinates in the original image.
    vdim : scalar int, optional
        vertical size of image, by default 1536 (VEDB project standard res).
        See note in `scale` kwarg for why this is necessary.
    hdim : int, optional
        horizontal size of image, by default 2048 (VEDB project standard res).
        See note in `scale` kwarg for why this is necessary.
    refinement_window_size : tuple, optional
        window size (in full-size pixels) for sub-pixel corner refinement with
        `cv2.cornerSubPix`; multiplied by `scale` (and rounded up) before use,
        by default (11, 11)
    checkerboard_size : tuple, optional
        number of inner corners (columns, rows) of the checkerboard, as
        expected by `cv2.findChessboardCorners`, by default (3, 6) (i.e. a
        board of 4 x 7 squares)
    detection_flags : int, optional
        flags for `cv2.findChessboardCorners`, by default 11 (see
        `find_checkerboard`)

    Returns
    -------
    marker : dict
        empty dict if no checkerboard is found; otherwise a dict with
        'timestamp' : float, input timestamp;
        'location_full_checkerboard' : (n_corners, 2) corner positions in
        full-size pixels;
        'norm_pos_full_checkerboard' : (n_corners, 2) corner positions
        normalized 0-1 by `hdim`, `vdim`;
        'location' : (2,) mean of corners, full-size pixels;
        'norm_pos' : (2,) mean of corners, normalized 0-1.
    """
    found_checkerboard, corners1 = cv2.findChessboardCorners(
        frame, checkerboard_size, detection_flags)
    # If found, add object points, image points (after refining them)
    computed_refinement_window_size = tuple(
        [int(np.ceil(x * scale)) for x in refinement_window_size])
    if found_checkerboard:
        # Fixed opencv parameters - revisit?
        n_iterations = 30
        threshold = 0.001
        sub_pixel_corner_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, n_iterations, threshold)
        zero_zone = (-1, -1)
        corners2 = cv2.cornerSubPix(
            frame, corners1, computed_refinement_window_size, zero_zone, sub_pixel_corner_criteria)
        # Parse outputs (convert back to full-size pixels, and convert
        # to 0-1 image coordinates for full checkerboard and centroid)
        corners = np.squeeze(corners2) / scale
        marker_position = np.mean(corners, axis=0)
        corners_normalized = corners / np.array([hdim, vdim])
        marker_position_normalized = np.mean(
            corners_normalized, axis=0)
        # Keep outputs
        tmp = dict(
            timestamp=timestamp,
            location_full_checkerboard=corners,
            norm_pos_full_checkerboard=corners_normalized,
            location=marker_position,
            norm_pos=marker_position_normalized,)

    else:
        tmp = {}
    return tmp

def find_checkerboard(
    video_file,
    timestamp_file,
    checkerboard_size=(3, 6),
    scale=0.5,
    refinement_window_size=(11,11),
    n_cores=12,
    detection_flags=11,
    start_frame=None,
    end_frame=None,
    batch_size=None,
    progress_bar=None,
	invert_contrast=False,
    ):
    """Use opencv to detect a checkerboard pattern

    Parameters
    ----------
    video_file : string
        video file to parse for checkerboards
    timestamp_file : string
        timestamp file to accompany video file, with timestamps per frame
    checkerboard_size : tuple, optional
        number of inner corners (columns, rows) of checkerboard, as expected
        by `cv2.findChessboardCorners`, by default (3, 6)
    scale : float, optional
        scale factor for video; values less than one reduce video size, by default 0.5.
        Outputs are converted back to full-size coordinates.
    refinement_window_size : tuple, optional
        window size (full-size pixels) for sub-pixel corner refinement, by
        default (11, 11)
    n_cores : int or None, optional
        number of processes for parallel detection (capped at the CPU count),
        by default 12; None runs serially
    detection_flags : int, optional
        detection flags for openCV findChessboardCorners (see 
        https://docs.opencv.org/3.0.0/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a)
        default is 11, which is equivalent to:
        `cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE`
        note: cv2.CALIB_CB_FAST_CHECK seems to have minimal effect...
    start_frame : int, optional
        frame at which to start parsing, by default None, which starts at video start
    end_frame : int, optional
        frame at which to stop parsing, by default None, which stops at video end
    batch_size : int, optional
        numer of frames to parse at once, by default None, which sets to a chunk of video
        that is ~4 GB in size
    progress_bar : tqdm.tqdm progress bar, optional
        progress bar to use for display through video, by default None
    invert_contrast : bool, optional
        whether to invert contrast of image (useful for some checkerboards on black
        backgrounds; opencv expects white background), by default False

    Returns
    -------
    checkerboard_locations : dict of arrays
        one entry per frame in which a checkerboard was found, with fields
        'timestamp', 'location_full_checkerboard', 'norm_pos_full_checkerboard',
        'location', 'norm_pos' (see `find_checkerboard_frame`). If none are
        found, all fields are empty arrays.
    """
    if progress_bar is None:
        def progress_bar(x, total=0):
            """Identity stand-in used when no progress bar is given."""
            return x

    timestamps = np.load(timestamp_file)
    n_frames_total, vdim, hdim, _ = file_io. list_array_shapes(video_file)
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = n_frames_total
    if batch_size is None:
        # This variable might be better as an input
        max_batch_bytes = 1024**3 * 4  # 4 GB
        n_bytes = (vdim * scale) * (hdim * scale)
        batch_size = int(np.floor(max_batch_bytes / n_bytes))
    # More computed args & useful variables
    use_multiprocessing = n_cores is not None
    n_frames = end_frame - start_frame
    n_batches = int(np.ceil(n_frames / batch_size))
    output_dicts = []
    for batch in range(n_batches):
        print("Running batch %d/%d" % (batch+1, n_batches))
        batch_start = batch * batch_size + start_frame
        batch_end = np.minimum(batch_start + batch_size, end_frame)
        print('Loading...')
        video_data = file_io.load_video(
            video_file,
            frames=(batch_start, batch_end),
            size=scale,
            color='gray')
        if invert_contrast:
            video_data = 255-video_data
        n_frames = batch_end - batch_start
        assert video_data.shape[0] == n_frames, 'Frame number error'
        assert len(timestamps[batch_start:batch_end]) == n_frames, 'Timestamp / frame number mismatch'
        # Args must be concatenated like this for use with parallel processing & progress bar
        # Look up "istarmap" for a pool object for slightly less clunky syntax, but use of e.g.
        # () requires python 3.8+, which I'd rather not commit to yet
        zz = zip(video_data,
                 timestamps[batch_start:batch_end],
                 repeat(scale),
                 repeat((vdim, hdim)),
                 repeat(refinement_window_size),
                 repeat(checkerboard_size),
                 repeat(detection_flags))
        if use_multiprocessing:
            if n_cores > multiprocessing.cpu_count():
                n_cores = multiprocessing.cpu_count()
            # Parallel call to framewise calculation of checkerboard position
            with Pool(n_cores) as p: # get_context('spawn').Pool(n_cores) as p: # 
                tmp_out = list(progress_bar(p.imap(_find_checkerboard_frame, zz), total=n_frames))
            # Filter output to have no empty dicts in list
            tmp_out = [x for x in tmp_out if x]
            output_dicts.extend(tmp_out)
        else:
            # Run serially (easier to debug if anything goes wrong)
            for args in progress_bar(zz, total=n_frames):
                tmp = _find_checkerboard_frame(args)
                if tmp:
                    output_dicts.append(tmp)
    # Handle empty output
    if len(output_dicts)==0:
        out = dict(
            timestamp=np.array([]),
            location_full_checkerboard=np.array([]),
            norm_pos_full_checkerboard=np.array([]),
            location=np.array([]),
            norm_pos=np.array([]),
            )
    else:
        out = dictlist_to_arraydict(output_dicts)
    return out
