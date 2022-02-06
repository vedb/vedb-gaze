import numpy as np
from .utils import onoff_from_binary
from pupil_recording_interface.externals.data_processing import (
    _filter_pupil_list_by_confidence,
    _extract_2d_data_monocular,
    _extract_2d_data_binocular,
    _match_data,
)
from pupil_recording_interface.externals import calibrate_2d


def get_data(pupil_list, ref_list, mode="2d", min_calibration_confidence=0.5):
    """Returns extracted data for calibration and whether there is binocular data

    Parameters
    ----------
    pupil_list :

    ref_list :
    """

    pupil_list = _filter_pupil_list_by_confidence(
        pupil_list, min_calibration_confidence
    )

    matched_data = _match_data(pupil_list, ref_list)
    (
        matched_binocular_data,
        matched_monocular_data,
        matched_pupil0_data,
        matched_pupil1_data,
    ) = matched_data

    binocular = None
    extracted_data = None
    if mode == "3d":
        if matched_binocular_data:
            binocular = True
            extracted_data = _extract_3d_data(g_pool, matched_binocular_data)
        elif matched_monocular_data:
            binocular = False
            extracted_data = _extract_3d_data(g_pool, matched_monocular_data)

    elif mode == "2d":
        if matched_binocular_data:
            binocular = True
            cal_pt_cloud_binocular = _extract_2d_data_binocular(matched_binocular_data)
            cal_pt_cloud0 = _extract_2d_data_monocular(matched_pupil0_data)
            cal_pt_cloud1 = _extract_2d_data_monocular(matched_pupil1_data)
            extracted_data = (
                cal_pt_cloud_binocular,
                cal_pt_cloud0,
                cal_pt_cloud1,
            )
        elif matched_monocular_data:
            binocular = False
            cal_pt_cloud = _extract_2d_data_monocular(matched_monocular_data)
            extracted_data = (cal_pt_cloud,)

    return binocular, extracted_data


def select_calibration_times(circle_positions, frame_times, stable_time=0.6):
    """"""
    # Find points with at least something detected
    circle_detected = np.all(~np.isnan(circle_positions), 1)
    on, off, dur = onoff_from_binary(circle_detected, return_duration=True).T

    durations = frame_times[off] - frame_times[on]
    # print(durations)
    keep = durations > stable_time
    out = np.vstack([frame_times[on[keep]], frame_times[off[keep]]]).T
    return out


def calibrate_2d_monocular(cal_pt_cloud, frame_size):
    method = "monocular polynomial regression"

    map_fn, inliers, params = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud, frame_size, binocular=False
    )
    if not inliers.any():
        return method, None

    mapper = "Monocular_Gaze_Mapper"
    args = {"params": params}
    result = {"subject": "start_plugin", "name": mapper, "args": args}
    return method, result


def calibrate_2d_binocular(cal_pt_cloud_binocular, cal_pt_cloud0,
                           cal_pt_cloud1, frame_size):
    method = "binocular polynomial regression"

    map_fn, inliers, params = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud_binocular, frame_size, binocular=True
    )
    if not inliers.any():
        return method, None

    map_fn, inliers, params_eye0 = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud0, frame_size, binocular=False
    )
    if not inliers.any():
        return method, None

    map_fn, inliers, params_eye1 = calibrate_2d.calibrate_2d_polynomial(
        cal_pt_cloud1, frame_size, binocular=False
    )
    if not inliers.any():
        return method, None

    mapper = "Binocular_Gaze_Mapper"
    args = {
        "params": params,
        "params_eye0": params_eye0,
        "params_eye1": params_eye1,
    }
    result = {"subject": "start_plugin", "name": mapper, "args": args}
    return method, result
