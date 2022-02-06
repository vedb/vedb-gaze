from data_analysis.utils import dictlist_to_arraydict
from pupil_recording_interface.externals.circle_detector import find_pupil_circle_marker
import numpy as np
import file_io
import cv2
import  copy

def _opencv_ellipse_to_dict(ellipse_dict):
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


def find_concentric_circles(video_file, timestamp_file, scale=1.0, start_frame=None, end_frame=None, batch_size=None, progress_bar=None):
    """Use PupilLabs circle detector to find concentric circles
    
    Assumes uint8 RGB image input

    Parameters
    ----------
    video_file : string
        video file to parse for checkerboards
    timestamp_file : string
        timestamp file to accompany video file, with timestamps per frame
    scale : float, optional
        [description], by default 1.0
    start_frame : [type], optional
        [description], by default None
    end_frame : [type], optional
        [description], by default None
    batch_size : [type], optional
        [description], by default None
    progress_bar : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """    """
    """
    if progress_bar is None:
        def progress_bar(x, total=0): return x

    timestamps = np.load(timestamp_file)

    # termination criteria: Make inputs?
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    n_frames_total, vdim, hdim, _ = file_io.var_size(video_file)
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
        video_data = file_io.load_mp4(
            video_file,
            frames=(batch_start, batch_end),
            size=scale,
            color='gray')

        for batch_frame, frame in enumerate(progress_bar(range(batch_start, batch_end))):
            world_circles = find_pupil_circle_marker(
                video_data[batch_frame], 1.0)
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
                this_circle["timestamp"] = timestamps[frame]
                output_dicts.append(this_circle)
    out = dictlist_to_arraydict(output_dicts)
    return out


def find_checkerboard(
    video_file,
    timestamp_file,
    checkerboard_size=(6, 8),
    scale=1.0,
    start_frame=None,
    end_frame=None,
    batch_size=None,
    progress_bar=None
    ):
    """Use opencv to detect a checkerboard pattern

    Parameters
    ----------
    video_file : string
        video file to parse for checkerboards
    timestamp_file : string
        timestamp file to accompany video file, with timestamps per frame
    checkerboard_size : tuple, optional
        size of checkerboard, by default (6, 8)
    scale : float, optional
        scale factor for video; values less than one reduce video size, by default 1.0
    start_frame : int, optional
        frame at which to start parsing, by default None, which starts at video start
    end_frame : int, optional
        frame at which to stop parsing, by default None, which stops at video end
    batch_size : int, optional
        numer of frames to parse at once, by default None, which sets to a chunk of video
        that is ~4 GB in size
    progress_bar : tqdm.tqdm progress bar, optional
        progress bar to use for display through video, by default None

    Returns
    -------
    checkerboard_locations
        dictionary of arrays for locations of checkerboards
    """
    if progress_bar is None:
        def progress_bar(x, total=0): return x

    timestamps = np.load(timestamp_file)

    # termination criteria: Make inputs?
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    n_frames_total, vdim, hdim, _ = file_io.var_size(video_file)
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
        video_data = file_io.load_mp4(
            video_file,
            frames=(batch_start, batch_end),
            size=scale,
            color='gray')

        for batch_frame, frame in enumerate(progress_bar(range(batch_start, batch_end))):
            # Find the chess board corners
            found_checkerboard, corners1 = cv2.findChessboardCorners(
                video_data[batch_frame], checkerboard_size, None)
            # If found, add object points, image points (after refining them)
            if found_checkerboard:
                # Fixed opencv parameters - revisit?
                winSize = (11, 11)
                zeroZone = (-1, -1)
                corners2 = cv2.cornerSubPix(
                    video_data[batch_frame], corners1, winSize, zeroZone, criteria)
                # Parse outputs (convert back to full-size pixels, and convert
                # to 0-1 image coordinates for full checkerboard and centroid)
                corners = np.squeeze(corners2) / scale
                marker_position = np.mean(corners, axis=0)
                corners_normalized = corners / np.array([hdim, vdim])
                marker_position_normalized = np.mean(
                    corners_normalized, axis=0)
                # Keep outputs
                tmp = dict(
                    timestamp=timestamps[frame],
                    location_full_checkerboard=corners,
                    norm_pos_full_checkerboard=corners_normalized,
                    location=marker_position,
                    norm_pos=marker_position_normalized,)
                output_dicts.append(tmp)

    return dictlist_to_arraydict(output_dicts)
