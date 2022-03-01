# Calibrate alternative for vedb

import numpy as np
import matplotlib.pyplot as plt
import pathlib
try:
    import thinplate as tps
except:
    print(("`thinplate` library not found.\n"
           "Please download and install py-thinplate-spline\n"
           "( http://github.com/cheind/py-thin-plate-spline )\n"
           "if you wish to use thin plate splines for calibration.\b"
           ))
import os

from .utils import (
    match_time_points,
    arraydict_to_dictlist, 
    dictlist_to_arraydict,
    read_yaml,
)

from pupil_recording_interface.externals.data_processing import (
    _filter_pupil_list_by_confidence,
    _extract_2d_data_monocular,
    _extract_2d_data_binocular,
    _match_data,
)
from pupil_recording_interface.externals import calibrate_2d

from pupil_recording_interface.externals.gaze_mappers import Binocular_Gaze_Mapper

BASE_DIR, _ = os.path.split(__file__)
BASE_DIR = pathlib.Path(BASE_DIR)

def parse_plab_data(pupil_list, ref_list, mode="2d", min_calibration_confidence=0.5):
    """Parse lists of dictionaries geneated by pupil labs code in preparation for calibration

    Returns extracted data for calibration and a flag indicating whether 
    there is binocular data present

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
            cal_pt_cloud_binocular = _extract_2d_data_binocular(
                matched_binocular_data)
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


def _fit_tps_gaze(calibration_markers, pupil, lambd=0.01, min_calibration_threshold=0.6, ):

    x, y = calibration_markers['norm_pos'].T
    px, py = pupil['norm_pos'].T
    keep = pupil['confidence'] > min_calibration_threshold
    # X
    data_x = np.vstack([px[keep], py[keep], x[keep]]).T
    theta_x = tps.TPS.fit(data_x, lambd=lambd)
    # Y
    data_y = np.vstack([px[keep], py[keep], y[keep]]).T
    theta_y = tps.TPS.fit(data_y, lambd=lambd)
    return data_x, theta_x, data_y, theta_y


def _map_tps(pupil, orig_data, mapper):
    """Map pupil coordinates to video coordinates via calculated transform
    
    Note that for thin plate splines, both the X and Y pupil coordinates are used to generate
    the X component of the gaze coordinates, and then both X and Y pupil coordinates
    are used to generate the Y component of the gaze coordinates

    Parameters
    ----------
    pupil : dict
        array of dicts for pupil position. Must contain 'norm_pos'
    mapper
    """
    xp, yp = pupil['norm_pos'].T
    out = tps.TPS.z(np.vstack([xp, yp]).T, orig_data, mapper)
    return out


def get_point_grid(n_points=60,
                   n_horizontal_lines=7,
                   n_vertical_lines=10,
                   st_horizontal=0,
                   fin_horizontal=1,
                   st_vertical=0,
                   fin_vertical=1):
    """Get a list of (x,y) points that form a rectangular grid
    
    Parameters
    ----------
    n_points : int, optional
        Number of points across horizontal axis (vertical points
        are computed from aspect ratio given by 
        `n_lines_<horiz / vert>`)
    n_horizontal_lines : int, optional
        Number of grid lines across the plot horizontally
    n_vertical_lines : int, optional
        Number of grid lines across the plot vertically
    st_horizontal : int, optional
        Description
    fin_horizontal : int, optional
        Description
    st_vertical : int, optional
        Description
    fin_vertical : int, optional
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    aspect_ratio = n_vertical_lines / n_horizontal_lines
    n_points_horizontal = n_points
    n_points_vertical = int(np.ceil(n_points / aspect_ratio))
    # Horizontal lines first
    tt_sparse_h = np.linspace(st_vertical, fin_vertical, n_horizontal_lines)
    tt_dense_h = np.linspace(
        st_horizontal, fin_horizontal, n_points_horizontal)
    xga, yga = np.meshgrid(tt_dense_h, tt_sparse_h)
    # Vertical lines next
    tt_sparse_v = np.linspace(st_horizontal, fin_horizontal, n_vertical_lines)
    tt_dense_v = np.linspace(st_vertical, fin_vertical, n_points_vertical)
    xgb, ygb = np.meshgrid(tt_sparse_v, tt_dense_v)
    # Combine & return
    xg_pts = np.hstack([xga.flatten(), xgb.flatten()])
    yg_pts = np.hstack([yga.flatten(), ygb.flatten()])
    return xg_pts, yg_pts


class Calibration(object):
    def __init__(self,
                 pupil_arrays,
                 calibration_arrays,
                 video_dims,
                 min_calibration_confidence=0.6,
                 calibration_type='monocular_pl',
                 mapping_computed=False,
                 **params,
                 ):
        """Class to compute gaze mapping calibration in a few different ways.
        
        Parameters
        ----------
        pupil_arrays : dictionary of arrays or tuple of 2 
            dictionaries of arrays
        calibration_arrays : dict
            dictionary of arrays for detected calibration markers
        video_dims : tuple
            (video_vdim, video_hdim)
        min_calibration_confidence : float, optional
            Description
        calibration_type : str, optional
            Description
        
        """
        calibration_list = arraydict_to_dictlist(calibration_arrays)

        # Store for later
        self.calibration_list = calibration_list
        self.calibration_type = calibration_type
        self.min_calibration_confidence = min_calibration_confidence
        self.calibration_arrays = calibration_arrays
        self.pupil_arrays = pupil_arrays
        self.video_dims = video_dims
        self.params = params
        self.mapping_computed = mapping_computed
        # Optionally load if params are not a dict
        if 'tag' in self.params:
            fpath = BASE_DIR / 'config' / \
                ('calibration_%s.yaml' % self.params['tag'])
            cal_params = read_yaml(fpath)
            if 'min_calibration_confidence' in cal_params:
                self.min_calibration_confidence = cal_params.pop(
                    'min_calibration_confidence')
            self.params = cal_params

        if isinstance(pupil_arrays, (list, tuple)) and len(pupil_arrays) == 2:
            left_pupil, right_pupil = self.pupil_arrays
            if self.mapping_computed:
                self.pupil_left = left_pupil
                self.pupil_right = right_pupil
            else:
                self.pupil_left = match_time_points(
                    calibration_arrays, left_pupil, )
                self.pupil_right = match_time_points(
                    calibration_arrays, right_pupil, )
                self.pupil_arrays = [self.pupil_left, self.pupil_right]
            self.pupil_list = arraydict_to_dictlist(left_pupil)
            self.pupil_list.extend(arraydict_to_dictlist(right_pupil))
            # Sort both left and right pupils by time. `id` field in each dict
            # will (SHOULD!) still indicate which eye is which. 
            self.pupil_list = sorted(self.pupil_list, key=lambda x: x['timepoint'])
        else:
            if not self.mapping_computed:
                self.pupil_arrays = match_time_points(
                    calibration_arrays, pupil_arrays)
            self.pupil_list = arraydict_to_dictlist(pupil_arrays)

        # Define mapping functions
        if not hasattr(self, 'map_params'):
            self._get_map_params()
        self._get_mapper()

    def _get_map_params(self):
        # Run calibration
        if self.calibration_type == 'monocular_pl':
            # NOTE: zero index for matched_data here is because this is monocular,
            # and matched data only returns a 1-long tuple.
            is_binocular, matched_data = parse_plab_data(self.pupil_list, self.calibration_list, mode='2d',
                                                               min_calibration_confidence=self.min_calibration_confidence)
            method, result = calibrate_2d_monocular(
                matched_data[0], frame_size=self.video_dims)
            self.map_params = result["args"]["params"]
            #cx, cy, n = self.map_params
        elif self.calibration_type == 'binocular_pl':
            is_binocular, matched_data = parse_plab_data(self.pupil_list, self.calibration_list, mode='2d',
                                                               min_calibration_confidence=self.min_calibration_confidence)
            method, result = calibrate_2d_binocular(
                *matched_data, frame_size=self.video_dims)
            self.map_params = result['args']
        elif self.calibration_type == 'monocular_tps':
            if self.params is None:
                kws = {}
            else:
                kws = self.params
            self.map_params = _fit_tps_gaze(self.calibration_arrays, self.pupil_arrays,
                                            min_calibration_threshold=self.min_calibration_confidence, **kws)

    def _get_mapper(self):
        if self.calibration_type == 'monocular_pl':
            self._mapper = calibrate_2d.make_map_function(
                *self.map_params)
        elif self.calibration_type == 'binocular_pl':
            self._mapper = Binocular_Gaze_Mapper(
                self.map_params["params"],
                self.map_params["params_eye0"],
                self.map_params["params_eye1"])
        elif self.calibration_type == 'monocular_tps':
            data_x, mapper_x, data_y, mapper_y = self.map_params
            # For thin plate splines, both the X and Y pupil coordinates are used to generate
            # the X component of the gaze coordinates, and then both X and Y pupil coordinates
            # are used to generate the Y component of the gaze coordinates
            def map_xy_tps(pupil_arrays):
                xtps = _map_tps(pupil_arrays, data_x, mapper_x)
                ytps = _map_tps(pupil_arrays, data_y, mapper_y)
                gaze = np.vstack([xtps, ytps]).T
                return gaze
            self._mapper = map_xy_tps

    def map(self, pupil_arrays, return_type='arraydict'):
        # Map gaze to video coordinates
        if self.calibration_type == 'binocular_pl':
            # Create a list of dicts, sorted by timestamps
            if not isinstance(pupil_arrays, list):
                raise ValueError('For binocular calibrations, `pupil_arrays` input must be a list of two dictionaries\n of arrays for left and right eyes.')
            pupil_list = arraydict_to_dictlist(pupil_arrays[0])
            pupil_list.extend(arraydict_to_dictlist(pupil_arrays[1]))
            pupil_list = sorted(pupil_list, key=lambda x: x['timestamp'])
            # Map w/ pupil's binocular mapper
            gaze = self._mapper.map_batch(pupil_list)
            # Convert back to array of dictionaries & reformat
            gaze_array = dictlist_to_arraydict(gaze)
            gaze_array['position'] = gaze_array.pop('norm_pos')
        elif self.calibration_type == 'monocular_pl':
            # Mapper takes two inputs: normalized pupil x and y position
            x, y = pupil_arrays['norm_pos'].T
            gaze = self._mapper([x, y])
            # Transpose output so time is the first dimension
            gaze = np.vstack(gaze).T
            gaze_array = dict(timestamp=pupil_arrays['timestamp'],
                              position=gaze)
            if 'confidence' in pupil_arrays:
                gaze_array['confidence']=pupil_arrays['confidence']
        elif self.calibration_type == 'monocular_tps':
            gaze = self._mapper(pupil_arrays)
            gaze_array = dict(timestamp=pupil_arrays['timestamp'],
                              position=gaze)
            if 'confidence' in pupil_arrays:
                gaze_array['confidence'] = pupil_arrays['confidence']

        if return_type == 'arraydict':
            return gaze_array
        elif return_type == 'dictlist':
            return arraydict_to_dictlist(gaze_array)            
        elif return_type == 'array':
            return gaze_array['position']

    @property
    def calibration_data(self):
        pp = dict(
            map_params=self.map_params,
            calibration_arrays=self.calibration_arrays,
            pupil_arrays=self.pupil_arrays,
            video_dims=self.video_dims,
            calibration_type=self.calibration_type,
        )
        return pp

    def save(self, fpath):
        """Save critical data to do mappings of pupil to gaze"""
        np.savez(fpath, **self.calibration_data)

    @classmethod
    def _get_fpath(base_dir, folder, eye, calibration_tag, pupil_tag, marker_type, marker_tag, epoch):
        """Get filepath to save, given elements, assuming this is not for database storage"""
        if epoch == 'all':
            fname = f'calibration_{calibration_tag}_{eye}_{pupil_tag}_{marker_type}_epoch{epoch}_{marker_tag}.npz'
        else:
            fname = f'calibration_{calibration_tag}_{eye}_{pupil_tag}_{marker_type}_epoch{epoch:02d}_{marker_tag}.npz'
        fpath = base_dir / folder / fname
        return fpath

    @classmethod
    def load(cls, fpath):        
        ob = cls.__new__(cls)
        tmp = np.load(fpath, allow_pickle=True)
        params = {}
        for k, v in tmp.items():
            if isinstance(v, np.ndarray) and (v.dtype == np.dtype('O')):
                if v.shape == ():
                    params[k] = v.item()
                else:
                    params[k] = v
            else:
                params[k] = v
        # This next line - setting a property before calling __init__ - feels somewhat shady... 
        # in conjunction with the if clause in __init__ above, seems to do what is desired.
        ob.map_params = params.pop('map_params')
        ob.__init__(**params)
        #ob._get_mapper()
        return ob

    def show_calibration(self,
                         im=None,
                         eye='left',
                         n_horizontal_lines=12,
                         n_vertical_lines=None,
                         n_points=40,
                         point_size=3,
                         sc=0.1,
                         color=((1.0, 0.9, 0),),
                         ax_world='new',
                         ax_eye=None,
                         ):
        """Plot visualization of calibration manifold on eye and 
        
        Parameters
        ----------
        im : None, optional
            Description
        eye : str, optional
            Description
        n_horizontal_lines : int, optional
            Description
        n_vertical_lines : int or None, optional
            if None, computed from n_horizontal_lines and aspect ratio
            of pupil points.
        n_lines_vert : int, optional
            Description
        """

        if ax_eye is not None:
            if isinstance(ax_eye, (list, tuple)):
                ax_left, ax_right = ax_eye
                # recursive call to plot l, r eye grids on l, r axes
        if self.calibration_type == 'binocular_pl':
            grid_pts_array_l = self._get_grid_points(
                'left', n_points=n_points, sc=sc, n_horizontal_lines=n_horizontal_lines, return_type='arraydict')
            grid_pts_array_l['id'] = np.ones_like(
                grid_pts_array_l['timestamp'], dtype=np.int64)
            grid_pts_array_l['confidence'] = np.ones_like(
                grid_pts_array_l['timestamp'], dtype=np.float32)
            grid_pts_array_r = self._get_grid_points(
                'right', n_points=n_points, sc=sc, n_horizontal_lines=n_horizontal_lines, return_type='arraydict')
            grid_pts_array_r['id'] = np.zeros_like(
                grid_pts_array_r['timestamp'], dtype=np.int64)
            grid_pts_array_r['confidence'] = np.ones_like(
                grid_pts_array_r['timestamp'], dtype=np.float32)
            if (grid_pts_array_l is None) or (grid_pts_array_r is None):
                grid_array = None
            else:
                gaze_binocular = self.map(stack_arraydicts(
                    grid_pts_array_l, grid_pts_array_r), return_type='dictlist')
                # Separate left and right eyes
                gaze['left'] = [
                    g for g in gaze_binocular if g['base_data'][0]['id'] == 0]
                gaze['left'] = gaze_utils.dictlist_to_arraydict(gaze_left)
                gaze['right'] = [
                    g for g in gaze_binocular if g['base_data'][0]['id'] == 1]
                gaze['right'] = gaze_utils.dictlist_to_arraydict(gaze_right)
                grid_array = gaze[eye]['position']
        elif self.calibration_type in ('monocular_pl', 'monocular_tps'):
            grid_pts_list = self._get_grid_points(
                eye, n_points=n_points, sc=sc, n_horizontal_lines=n_horizontal_lines, return_type='arraydict')
            if grid_pts_list is None:
                grid_array = None
            else:
                grid_array = self.map(grid_pts_list, return_type='array')
        if ax_world is not None:
            if ax_world == 'new':
                fig, ax_world = plt.subplots()
        if im is not None:
            ax_world.imshow(im, extent=[0, 1, 1, 0])
        if grid_array is not None:
            ax_world.scatter(*grid_array.T, s=point_size, c=color)
        ax_world.axis([0, 1, 1, 0])
        if ax_eye is not None:
            eye_array = self._get_grid_points(
                eye, n_points=n_points, sc=sc, n_horizontal_lines=n_horizontal_lines, return_type='array')
            if eye_array is not None:
                ax_eye.scatter(*eye_array.T, s=point_size, c=color)
            #ax_eye.axis([0, 1, 1, 0])

    def _get_grid_points(self, eye=None, sc=0.1, n_horizontal_lines=10, n_vertical_lines=None, n_points=40, return_type='dictlist', min_possible_eye_points=10):
        """Get grid points for displaying calibration
        
        Parameters
        ----------
        eye : TYPE
            Description
        """
        if 'binocular' not in self.calibration_type: 
            pupils = self.pupil_arrays
        else:
            pupils = getattr(self, 'pupil_%s' % eye)
        ci = pupils['confidence'] > self.min_calibration_confidence
        if ci.sum() < min_possible_eye_points:
            return None
        x, y = pupils['norm_pos'][ci].T
        vmin, vmax = np.min(y), np.max(y)
        hmin, hmax = np.min(x), np.max(x)
        height = np.ptp(y)
        width = np.ptp(x)
        vmin += height * sc
        vmax -= height * sc
        hmin += width * sc
        hmax -= width * sc
        aspect_ratio = width / height
        pts = get_point_grid(
            n_horizontal_lines=n_horizontal_lines,
            n_vertical_lines=np.int(
                n_horizontal_lines * aspect_ratio) if n_vertical_lines is None else n_vertical_lines,
            st_horizontal=hmin,
            fin_horizontal=hmax,
            st_vertical=vmin,
            fin_vertical=vmax,
            n_points=n_points,
        )
        pts = np.array(pts).T
        grid_pts = dict(norm_pos=pts, timestamp=np.arange(len(pts))/120.)
        grid_pts_list = arraydict_to_dictlist(grid_pts)
        if return_type == 'array':
            return pts
        elif return_type == 'arraydict':
            return grid_pts
        elif return_type == 'dictlist':
            return grid_pts_list
