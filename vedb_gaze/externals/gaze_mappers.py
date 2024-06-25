"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from collections import deque

import numpy as np

from . import calibrate_2d


class Binocular_Gaze_Mapper:
    def __init__(self, params, params_eye0, params_eye1):
        self.params = params
        self.params_eye0 = params_eye0
        self.params_eye1 = params_eye1
        self.multivariate = True
        self.map_fn = calibrate_2d.make_map_function(*self.params)
        self.map_fn_fallback = []
        self.map_fn_fallback.append(
            calibrate_2d.make_map_function(*self.params_eye0)
        )
        self.map_fn_fallback.append(
            calibrate_2d.make_map_function(*self.params_eye1)
        )
        self.min_pupil_confidence = 0.6
        self._caches = (deque(), deque())
        self.recently_estimated_framerate = 1 / 120
        self.framerate_estimation_smoothing_factor = 1 / 50
        self.sample_cutoff = 10

    def _map_binocular(self, p0, p1):
        if self.multivariate:
            gaze_point = self.map_fn(p0["norm_pos"], p1["norm_pos"])
        else:
            gaze_point_eye0 = self.map_fn_fallback[0](p0["norm_pos"])
            gaze_point_eye1 = self.map_fn_fallback[1](p1["norm_pos"])
            gaze_point = (
                (gaze_point_eye0[0] + gaze_point_eye1[0]) / 2.0,
                (gaze_point_eye0[1] + gaze_point_eye1[1]) / 2.0,
            )
        confidence = (p0["confidence"] + p1["confidence"]) / 2.0
        ts = (p0["timestamp"] + p1["timestamp"]) / 2.0
        return {
            "topic": "gaze.2d.01.",
            "norm_pos": gaze_point,
            "confidence": confidence,
            "timestamp": ts,
            "base_data": [p0, p1],
        }

    def _map_monocular(self, p):
        gaze_point = self.map_fn_fallback[p["id"]](p["norm_pos"])
        return {
            "topic": "gaze.2d.{}.".format(p["id"]),
            "norm_pos": gaze_point,
            "confidence": p["confidence"],
            "timestamp": p["timestamp"],
            "base_data": [p],
        }

    def is_cache_valid(self, cache):
        return len(cache) >= 2

    def estimate_frame_rate_raw(self, cache):
        return np.mean(np.diff([d["timestamp"] for d in cache]))

    def estimate_framerate_smoothed(self, eye0_cache, eye1_cache):
        if self.is_cache_valid(eye0_cache) and self.is_cache_valid(eye1_cache):
            eye0_framerate_raw = self.estimate_frame_rate_raw(eye0_cache)
            eye1_framerate_raw = self.estimate_frame_rate_raw(eye1_cache)
            estimated_framerate_raw = max(
                eye0_framerate_raw, eye1_framerate_raw
            )
        elif self.is_cache_valid(eye0_cache):
            estimated_framerate_raw = self.estimate_frame_rate_raw(eye0_cache)
        elif self.is_cache_valid(eye1_cache):
            estimated_framerate_raw = self.estimate_frame_rate_raw(eye1_cache)
        else:
            return self.recently_estimated_framerate

        self.recently_estimated_framerate += (
            estimated_framerate_raw - self.recently_estimated_framerate
        ) * self.framerate_estimation_smoothing_factor
        return self.recently_estimated_framerate

    def map_batch(self, pupil_list):
        current_caches = self._caches
        self._caches = (deque(), deque())
        results = []
        for p in pupil_list:
            results.extend(self.on_pupil_datum(p))

        self._caches = current_caches
        return results

    def on_pupil_datum(self, p):
        self._caches[p["id"]].append(p)
        temporal_cutoff = 2 * self.estimate_framerate_smoothed(*self._caches)

        # map low confidence pupil data monocularly
        if (
            self._caches[0]
            and self._caches[0][0]["confidence"] < self.min_pupil_confidence
        ):
            p = self._caches[0].popleft()
            gaze_datum = self._map_monocular(p)
        elif (
            self._caches[1]
            and self._caches[1][0]["confidence"] < self.min_pupil_confidence
        ):
            p = self._caches[1].popleft()
            gaze_datum = self._map_monocular(p)
        # map high confidence data binocularly if available
        elif self._caches[0] and self._caches[1]:
            # we have binocular data
            if (
                self._caches[0][0]["timestamp"]
                < self._caches[1][0]["timestamp"]
            ):
                p0 = self._caches[0].popleft()
                p1 = self._caches[1][0]
                older_pt = p0
            else:
                p0 = self._caches[0][0]
                p1 = self._caches[1].popleft()
                older_pt = p1

            if abs(p0["timestamp"] - p1["timestamp"]) < temporal_cutoff:
                gaze_datum = self._map_binocular(p0, p1)
            else:
                gaze_datum = self._map_monocular(older_pt)

        elif len(self._caches[0]) > self.sample_cutoff:
            p = self._caches[0].popleft()
            gaze_datum = self._map_monocular(p)
        elif len(self._caches[1]) > self.sample_cutoff:
            p = self._caches[1].popleft()
            gaze_datum = self._map_monocular(p)
        else:
            gaze_datum = None

        if gaze_datum:
            return [gaze_datum]
        else:
            return []
