try:
    import pupil_detectors
except:
    print("No pupil detection available; pupil_detectors library not present")
import numpy as np
import file_io
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
    **kwargs
    ):
    """
    This is a simple wrapper to allow Pupil Labs `pupil_detectors` code
    to process a whole video of eye data.
    Parameters
    ----------
    video_file : string
        video file to parse for checkerboards
    timestamp_file : string
        timestamp file to accompany video file, with timestamps per frame
    progress_bar : tqdm object or None
        if tqdm object is provided, a progress bar is displayed
        as the video is processed.
    id : int, 0 or 1
        ID for eye (eye0, i.e. left, or eye1, i.e. right)
    Notes
    -----
    Parameters for Pupil Detector2D object, passed as a dict called
    `properties` to `pupil_detectors.Detector2D()`; fields are:
        coarse_detection = True
        coarse_filter_min = 128
        coarse_filter_max = 280
        intensity_
        ge = 23
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
    Returns
    -------
    pupil_dicts : list of dicts
        dictionary for each detected instance of a pupil. Each
        entry has fields:
        luminance
        timestamp [if timestamps are provided, which they should be]
        norm_pos
    """
    scale = 1.0  # hard-coded to always load full-size video
    if progress_bar is None:
        def progress_bar(x): return x
    timestamps = np.load(timestamp_file)
    # Specify detection method later?
    if properties is None:
        properties = {}
    det = pupil_detectors.Detector2D(properties=properties)

    n_frames_total, vdim, hdim, _ = file_io.var_size(video_file)
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
        video_data = file_io.load_mp4(
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
            # Get rid of raw data as input; no need to keep
            if "internal_2d_raw_data" in out:
                _ = out.pop("internal_2d_raw_data")
            # Save average luminance of eye video for reference
            out["luminance"] = fr.mean()
            # Normalized position
            out["norm_pos"] = (np.array(out["location"]) / eye_dims).tolist()
            if timestamps is not None:
                out["timestamp"] = timestamps[frame]
            if id is not None:
                out["id"] = id
            pupil_dicts.append(out)
    out = dictlist_to_arraydict(pupil_dicts)
    return out
