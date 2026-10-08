"""Pupil detection with the Pupil Labs 2D detector (`pupil_detectors`).

Provides `plabs_detect_pupil`, one of the functions that can be named by
``config/pupil-<tag>.yaml`` and run by the `pipelines.pupil_detection` step.
"""
try:
    import pupil_detectors
except:
    print("No pupil detection available; pupil_detectors library not present")
import numpy as np
import file_io
import pathlib
import time
from .utils import dictlist_to_arraydict


def plabs_detect_pupil(
    video_file, 
    timestamp_file=None, 
    start_frame=None, 
    end_frame=None, 
    batch_size=None,
    progress_bar=None,
    id=None, 
    properties=None, 
    sleep_time=0.001,
    **kwargs
    ):
    """Detect pupils in every frame of an eye video with the Pupil Labs 2D detector

    Simple wrapper allowing Pupil Labs `pupil_detectors.Detector2D` to process
    a whole eye video, loaded in batches of up to ~4 GB.

    Parameters
    ----------
    video_file : str
        eye video file in which to detect pupils. If `id` is None, the file
        name (not the folder) must start with 'eye0' or 'eye1' (Pupil Labs
        convention).
    timestamp_file : str or pathlib.Path, optional
        .npy file with one timestamp per video frame, by default None (no
        'timestamp' field in the output)
    start_frame : int, optional
        first frame to process, by default None (start of video)
    end_frame : int, optional
        frame at which to stop (exclusive), by default None (end of video)
    batch_size : int, optional
        number of frames to load at once, by default None, which uses as many
        frames as fit in ~4 GB
    progress_bar : callable or None, optional
        e.g. `tqdm.tqdm`; wraps the per-frame loop to display progress, by
        default None (no progress bar)
    id : int, optional
        eye ID stored in the 'id' field of the output: 0 for eye0 (right eye)
        or 1 for eye1 (left eye), by default None, which infers it from the
        file name of `video_file`
    properties : dict, optional
        parameters for `pupil_detectors.Detector2D` (see Notes), by default
        None (detector defaults)
    sleep_time : float, optional
        seconds to sleep after each frame, by default 0.001
    **kwargs
        ignored

    Returns
    -------
    pupil_data : dict of arrays
        one entry per frame, with the fields returned by the Pupil Labs
        detector (e.g. 'location' in pixels, 'confidence', 'ellipse',
        'diameter'; 'internal_2d_raw_data' is removed) plus:
        'norm_pos' : (n, 2) pupil position normalized 0-1 by eye video
        width and height;
        'luminance' : mean gray value of the frame;
        'timestamp' : timestamp for the frame;
        'id' : eye ID.

    Notes
    -----
    Parameters for Pupil Detector2D object, passed as a dict called
    `properties` to `pupil_detectors.Detector2D()`; fields are:
        coarse_detection = True
        coarse_filter_min = 128
        coarse_filter_max = 280
        intensity_range = 23
        blur_size = 5
        canny_treshold = 160
        canny_ration = 2
        canny_aperture = 5
        pupil_size_max = 100
        pupil_size_min = 10
        strong_perimeter_ratio_range_min = 0.6
        strong_perimeter_ratio_range_max = 1.1
        strong_area_ratio_range_min = 0.8
        strong_area_ratio_range_max = 1.1
        contour_size_min = 5
        ellipse_roundness_ratio = 0.09    # HM! Try setting this?
        initial_ellipse_fit_treshhold = 4.3
        final_perimeter_ratio_range_min = 0.5
        final_perimeter_ratio_range_max = 1.0
        ellipse_true_support_min_dist = 3.0
        support_pixel_ratio_exponent = 2.0
    """
    scale = 1.0  # hard-coded to always load full-size video
    if id is None:
        # Infer eye id from the file name only (video_file may be a str or a
        # pathlib.Path; folder names must not affect the result)
        video_name = pathlib.Path(video_file).name
        if video_name.startswith('eye0'):
            id = 0
        elif video_name.startswith('eye1'):
            id = 1
        else:
            raise ValueError(f"Can't infer the eye id from video file name {video_name!r}: "
                             "per Pupil Labs conventions it should start with 'eye0' (right "
                             "eye) or 'eye1' (left eye). Otherwise pass `id=0` or `id=1`.")
    if progress_bar is None:
        def progress_bar(x):
            """Identity stand-in used when no progress bar is given."""
            return x
    if timestamp_file is None:
        timestamps = None
    else:
        timestamps = np.load(timestamp_file)
    # Specify detection method later?
    if properties is None:
        properties = {}
    det = pupil_detectors.Detector2D(properties=properties)

    n_frames_total, vdim, hdim, _ = file_io. list_array_shapes(video_file)
    eye_dims = np.array([hdim, vdim])
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
    pupil_dicts = []
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

        #for frame in progress_bar(range(n_frames)):
        for batch_frame, frame in enumerate(progress_bar(range(batch_start, batch_end))):
            fr = video_data[batch_frame].copy()
            # Pupil needs c-ordered arrays, so switch from default load:
            fr = np.ascontiguousarray(fr)
            # Call detector & process output
            out = det.detect(fr)
            time.sleep(sleep_time)
            # Get rid of raw data as input; no need to keep
            if "internal_2d_raw_data" in out:
                _ = out.pop("internal_2d_raw_data")
            # Save average luminance of eye video for reference
            out["luminance"] = fr.mean()
            # Normalized position
            out["norm_pos"] = (np.array(out["location"]) / eye_dims).tolist()
            if timestamps is not None:
                out["timestamp"] = timestamps[frame]
            out["id"] = id
            pupil_dicts.append(out)
    out = dictlist_to_arraydict(pupil_dicts)
    return out
