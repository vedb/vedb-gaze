"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import os
import sys
import getpass
import platform
import warnings
import logging

import numpy as np

logger = logging.getLogger(__name__)


def get_system_info():
    try:
        if platform.system() == "Windows":
            username = os.environ["USERNAME"]
            sysname, nodename, release, version, machine, _ = platform.uname()
        else:
            username = getpass.getuser()
            sysname, nodename, release, version, machine = os.uname()
    except Exception as e:
        logger.error(e)
        username = "unknown"
        sysname, nodename, release, version, machine = (
            sys.platform,
            "unknown",
            "unknown",
            "unknown",
            "unknown",
        )

    info_str = "User: {}, Platform: {}, Machine: {}, Release: {}, Version: {}"

    return info_str.format(username, sysname, nodename, release, version)


def gen_pattern_grid(size=(4, 11)):
    pattern_grid = []
    for i in range(size[1]):
        for j in range(size[0]):
            pattern_grid.append([(2 * j) + i % 2, i, 0])
    return np.asarray(pattern_grid, dtype="f4")


def normalize(pos, size, flip_y=False):
    """
    normalize return as float
    """
    width, height = size
    x = pos[0]
    y = pos[1]
    x /= float(width)
    y /= float(height)
    if flip_y:
        return x, 1 - y
    return x, y


def denormalize(pos, size, flip_y=False):
    """
    denormalize
    """
    width, height = size
    x = pos[0]
    y = pos[1]
    x *= width
    if flip_y:
        y = 1 - y
    y *= height
    return x, y


def dist_pts_ellipse(ellipse, points):
    """
    return unsigned euclidian distances of points to ellipse
    """
    pos, size, angle = ellipse
    ex, ey = pos
    dx, dy = size
    pts = np.float64(points)
    rx, ry = dx / 2.0, dy / 2.0
    angle = (angle / 180.0) * np.pi
    pts = pts - np.array(
        (ex, ey)
    )  # move pts to ellipse appears at origin , with this we copy data -deliberatly!

    M_rot = np.mat(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    pts = np.array(
        pts * M_rot
    )  # rotate so that ellipse axis align with coordinate system

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pts /= np.array((rx, ry))  # normalize such that ellipse radii=1
        norm_mag = np.sqrt((pts * pts).sum(axis=1))
        norm_dist = abs(
            norm_mag - 1
        )  # distance of pt to ellipse in scaled space
        ratio = (
            norm_dist
        ) / norm_mag  # scale factor to make the pts represent their dist to ellipse
        scaled_error = np.transpose(
            pts.T * ratio
        )  # per vector scalar multiplication: makeing sure that boradcasting is done right
        real_error = scaled_error * np.array((rx, ry))
        error_mag = np.sqrt((real_error * real_error).sum(axis=1))

    return error_mag
