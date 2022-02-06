# Calibrate alternative for vedb
import numpy as np

from .utils import onoff_from_binary, time_to_index


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
