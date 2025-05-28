# Full pipelines for gaze estimation
import numpy as np
import tqdm
import tqdm.notebook
import file_io
import os
import pathlib

from . import utils
from .options import config
from .calibration import Calibration


BASE_DIR = pathlib.Path(config.get('paths', 'base_dir')).expanduser()
PYDRA_OUTPUT_DIR = pathlib.Path(config.get('paths', 'pydra_cache') ).expanduser()
BASE_OUTPUT_DIR = pathlib.Path(config.get('paths', 'output_dir') ).expanduser()
PARAM_DIR = pathlib.Path(os.path.split(__file__)[0]) / 'config'

timestamp_template = 'timestamps_0start'
world_template ='Private' # or ''
eye_template = '' # or '_blur'

### --- Utilities --- ###
def is_notebook():
    """Test whether code is being run in a jupyter notebook or not"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def get_default_kwargs(fn):
    """Get keyword arguments and default values for a function

    Uses `inspect` module; this is a thin convenience wrapper

    Parameters
    ----------
    fn : function
        function for which to get kws
    """
    import inspect
    kws = inspect.getargspec(fn)
    defaults = dict(zip(kws.args[-len(kws.defaults):], kws.defaults))
    return defaults


### --- Pipeline steps --- ###
def pupil_detection(eye_video_file,
        eye_time_file, 
        param_tag, 
        output_dir,
        eye='left', 
        is_verbose=False):
    """Run a pupil detection function as a pipeline step
    
    Parameters
    ----------
    eye_vide_file : str
        file to load in which to detect pupils
    eye_time_file : str
        file to load with timestamps for eye video (if separate)
    output_dir : pathlib.Path or str
        directory for 

    """
    if is_verbose:
        print("\n=== Finding pupil locations ===\n")
    if is_notebook():
        progress_bar = tqdm.notebook.tqdm
    else:
        progress_bar = tqdm.tqdm
    # assure `output_dir` is a pathlib object
    output_dir = pathlib.Path(output_dir)
    # Check for extant file / failed run
    fpath = output_dir / f'pupil_detection-{eye}-{param_tag}.npz'
    fpath_fail = output_dir / (fpath.name.replace('.npz', 'failed'))
    if fpath.exists():
        return fpath
    if fpath_fail.exists():
        return fpath_fail
    # Load parameters from stored yaml files
    param_fpath = PARAM_DIR / f'pupil-{param_tag}.yaml'
    kwargs = utils.read_yaml(param_fpath)
    fn = kwargs.pop('fn')
    # convert `fn` string to callable function
    func = utils.get_function(fn)
    # Optionally add progress bar, if supported
    default_kw = get_default_kwargs(func)
    if 'progress_bar' in default_kw:
        kwargs['progress_bar'] = progress_bar
    # Call function
    data = func(eye_video_file, timestamp_file=eye_time_file, **kwargs,)
    # Detect failure
    failed = len(data['norm_pos']) == 0
    # Outputs
    if failed:
        # if failed, save empty text file
        fpath_fail.open(mode='w')
        return fpath_fail
    else:
        np.savez(fpath, **data)
        return fpath


def marker_detection(video_file,
                     time_file,
                     param_tag,
                     output_dir,
                     start_frame=None,
                     end_frame=None,
                     epoch=None,
                     is_verbose=False):
    """Run a pupil detection function as a pipeline step
    
    Parameters
    ----------
    session folder : str
        folder for eye video session to run

    """
    # For detections, check for extant output file:
    if epoch is None:
        epoch_str = 'epochall'
    else:
        epoch_str = f'epoch{epoch:02d}'
    # assure `output_dir` is a pathlib object
    output_dir = pathlib.Path(output_dir)
    # Check for extant file / failed run
    fpath = output_dir / f'markers-{param_tag}-{epoch_str}.npz'
    fpath_fail = (fpath.parent / fpath.name).replace('.npz','.failed')
    if fpath.exists():
        return fpath
    elif fpath_fail.exists():
        return fpath_fail

    if is_verbose:
        print(f"\n=== Finding all marker locations w/ {param_tag} ===\n")
    if is_notebook():
        progress_bar = tqdm.notebook.tqdm
    else:
        progress_bar = tqdm.tqdm
    # Load parameters from stored yaml files
    param_fpath = PARAM_DIR / f'marker-{param_tag}.yaml'
    kwargs = utils.read_yaml(param_fpath)
    fn = kwargs.pop('fn')
    # convert `fn` string to callable function
    func = utils.get_function(fn)
    # Optionally add progress bar, if supported
    default_kw = get_default_kwargs(func)
    if 'progress_bar' in default_kw:
        kwargs['progress_bar'] = progress_bar
    # Optionally (if present), use manual labels for marker times
    if start_frame is not None:
        if 'start_frame' in kwargs:
            if kwargs['start_frame'] is None:
                _ = kwargs.pop('start_frame')
            else:
                raise ValueError("Multiple manual start / end times provided.")
    if end_frame is not None:
        if 'end_frame' in kwargs:
            if kwargs['end_frame'] is None:
                _ = kwargs.pop('end_frame')
            else:
                raise ValueError(
                    "Multiple manual start / end times provided.")
    # Run function
    data = func(video_file, time_file, 
        start_frame=start_frame,
        end_frame=end_frame,
        **kwargs,)
    # Check for failure
    failed = len(data['norm_pos']) == 0
    if failed:
        # if failed, save empty text file
        fpath_fail.open(mode='w')
    else:
        np.savez(fpath, **data)                    
    return fpath

def marker_splitting(marker_file,
        time_file,
        param_tag,
        output_dir,
        is_verbose=False):
    """"""
    if is_verbose:
        print(f"\n=== Finding marker epochs ({param_tag}) ===\n")
    if is_notebook:
        progress_bar = tqdm.notebook.tqdm
    else:
        progress_bar = tqdm.tqdm

    # assure `output_dir` is a pathlib object
    output_dir = pathlib.Path(output_dir)
    # Check for failed input
    if 'failed' in marker_file:
        return 'previous_step.failed'
    # Check for extant file / previous failed run of this step
    _, orig_tag, epoch_str = marker_file.name.split('-')
    fname_test = f'marker-{orig_tag}-{param_tag}-epoch*'
    # Potentially multi-file check
    files_out, _ = check_files(output_dir, fname_test)
    if len(files_out) > 0:
        return files_out

    # Get marker file
    marker_data = dict(np.load(marker_file))
    # Get time
    all_timestamps = np.load(time_file)
    # Load parameters from stored yaml files
    param_fpath = PARAM_DIR / f'marker_parsing-{param_tag}.yaml'
    kwargs = utils.read_yaml(param_fpath)
    fn = kwargs.pop('fn')
    # convert `fn` string to callable function
    func = utils.get_function(fn)
    # Optionally add progress bar, if supported
    default_kw = get_default_kwargs(func)
    if 'progress_bar' in default_kw:
        kwargs['progress_bar'] = progress_bar
    # Run function; inputs must be marker data, all_timestamps
    data = func(marker_data, all_timestamps, **kwargs,)
    # Detect failure
    failed = len(data) == 0
    # Manage epochs
    if failed:
        fname = f'marker-{orig_tag}-{param_tag}-{epoch_str}.failed'
        (output_dir / fname).open()
        fnames = [fname]

    elif (len(data) >= 1):
        # Save each data instance as an epoch if 'epochall' provided    
        fnames = []
        for e, dd in enumerate(data):
            fpath = output_dir / f'marker-{orig_tag}-{param_tag}-epoch{e:02d}.npz'
            np.savez(fpath, **dd)
            fnames.append(fpath)

    return fnames

def marker_clustering(marker_file,
        time_file,
        param_tag,
        output_dir,
        is_verbose=False):
    """"""
    if is_verbose:
        print(f"\n=== Finding marker epochs ({param_tag}) ===\n")
    if is_notebook:
        progress_bar = tqdm.notebook.tqdm
    else:
        progress_bar = tqdm.tqdm
    
    # assure `output_dir` is a pathlib object
    output_dir = pathlib.Path(output_dir)
    # Check for failed input
    if 'failed' in marker_file:
        return output_dir / 'previous_step.failed'
    # Check for extant file / previous failed run of this step
    _, orig_tag, epoch_str = marker_file.name.split('-')
    fname_test = f'marker-{orig_tag}-{param_tag}-epoch*'
    # Potentially multi-file check
    files_out, _ = check_files(output_dir, fname_test)
    if len(files_out) > 0:
        return files_out
    
    # Get marker file
    marker_data = dict(np.load(marker_file))
    # Get time
    all_timestamps = np.load(time_file)
    # Load parameters from stored yaml files
    param_fpath = PARAM_DIR / f'marker_parsing-{param_tag}.yaml'
    kwargs = utils.read_yaml(param_fpath)
    fn = kwargs.pop('fn')
    # convert `fn` string to callable function
    func = utils.get_function(fn)
    # Optionally add progress bar, if supported
    default_kw = get_default_kwargs(func)
    if 'progress_bar' in default_kw:
        kwargs['progress_bar'] = progress_bar
    # Run function; inputs must be marker data, all_timestamps
    data = func(marker_data, all_timestamps, **kwargs,)
    # Detect failure
    failed = data is None #len(data) == 0
    # Manage epochs
    if failed:
        fpath = output_dir / f'marker-{orig_tag}-{param_tag}-{epoch_str}.failed'
        fpath.open(mode='w')
    else:
        fpath = output_dir / f'marker-{orig_tag}-{param_tag}-{epoch_str}.npz'
        np.savez(fpath, **data)
    return fpath


def compute_calibration(marker_file,
                        pupil_files,
                        input_hash,
                        param_tag,
                        video_dimensions,
                        output_dir,
                        eye=None,
                        is_verbose=False):
    """
    pupil_files can be list of [left, right]. This should be a dict to be more explicit.
    param_tag should be informative;
    params it points to specify:
        calibration_type
        min_confidence
        (lambda)
        (etc)
    
    """
    # Handle inputs
    if not isinstance(pupil_files, (list, tuple)):
        pupil_files = [pupil_files]
    fname = f'calibration-{eye}-{param_tag}-{input_hash}.npz'
    # assure `output_dir` is a pathlib object
    output_dir = pathlib.Path(output_dir)
    # Check for failed input
    if ('failed' in str(marker_file)) or any(['failed' in str(pf) for pf in pupil_files]):
        return output_dir / 'previous_step.failed'
    # Check for extant file / previous failed run of this step
    fpath = output_dir / fname
    if fpath.exists():
        return fpath
    fpath_fail = output_dir / (fname.replace('.npz', 'failed'))
    if fpath_fail.exists():
        return fpath_fail

    if is_verbose:
        print("\n=== Computing calibration ===\n")

    # Get marker file
    marker_data = dict(np.load(marker_file))
    pupil_data = [dict(np.load(fp)) for fp in pupil_files]
    # Load parameters from stored yaml files
    param_fpath = PARAM_DIR / f'calibration-{param_tag}.yaml'
    kwargs = utils.read_yaml(param_fpath)
    fn = kwargs.pop('fn')
    
    if len(pupil_data) == 1:
        # only one eye
        pupil_data = pupil_data[0]
    # convert `fn` string to callable class
    cal_class = utils.get_function(fn) # calibration_class)
    print("Computing calibration...")
    try:
        cal = cal_class(pupil_data, marker_data, video_dimensions, **kwargs,)
        failed = False
    except:
        failed = True

    # Manage ouptut file
    if failed:
        fpath = output_dir / fname.replace('.npz','.failed')
        fpath.open()
    else:
        fpath = output_dir / fname
        cal.save(fpath)
    return fpath


def map_gaze(pupil_files,
             calibration_file,
             param_tag,
             output_dir,
             is_verbose=False):
    """Estimate gaze from calibration & pupil positions"""
    # Handle inputs
    if not isinstance(pupil_files, (list, tuple)):
        pupil_files = [pupil_files]

    fname = f'gaze_{eye}-{param_tag}-{calibration_tag}-{input_hash}.npz'
    # assure `output_dir` is a pathlib object
    output_dir = pathlib.Path(output_dir)
    # Check for failed input
    if ('failed' in str(calibration_file)) or any(['failed' in str(pf) for pf in pupil_files]):
        return output_dir / 'previous_step.failed'
    # Check for extant file / previous failed run of this step
    fpath = output_dir / fname
    if fpath.exists():
        return fpath
    fpath_fail = output_dir / (fname.replace('.npz', 'failed'))
    if fpath_fail.exists():
        return fpath_fail

    if is_verbose:
        print("\n=== Computing gaze ===\n")

    # Get marker file
    pupil_data = [dict(np.load(fp)) for fp in pupil_files]
    calibration = Calibration.load(calibration_file)
    _, eye, calibration_tag, input_hash = calibration_file.name.split('-')
    # Load parameters from stored yaml files
    param_fpath = PARAM_DIR / f'gaze-{param_tag}.yaml'
    kwargs = utils.read_yaml(param_fpath)
    fn = kwargs.pop("fn")
    func = utils.get_function(fn)
    
    if len(pupil_data) == 1:
        # only one eye
        pupil_data = pupil_data[0]
    
    print("Computing gaze...")
    data = func(calibration, pupil_data, **kwargs)

    # Detect failure?
    failed = len(data) == 0
    # Manage ouptut file
    if failed:
        fpath_fail.open(mode='w')
    else:
        np.savez(fpath, **data)
    return fpath


def compute_error(gaze_file,
             marker_file,
             param_tag,
             output_dir,
             input_hash,
             eye=None,
             is_verbose=False):
    if is_verbose:
        print("\n=== Computing error ===\n")
    try:
        # Get marker file
        _, orig_tag, cluster_tag, epoch_str = marker_file.name.split('-')
        fname = f'error-{eye}-{param_tag}-{input_hash}-{epoch_str}.npz'
        # assure `output_dir` is a pathlib object
        output_dir = pathlib.Path(output_dir)
        # Check for failed input
        if ('failed' in str(marker_file)) or ('failed' in str(gaze_file)):
            return output_dir / 'previous_step.failed'
        # Check for extant file / previous failed run of this step
        fpath = output_dir / fname
        if fpath.exists():
            return fpath
        fpath_fail = output_dir / (fname.replace('.npz', 'failed'))
        if fpath_fail.exists():
            return fpath_fail

        gaze_data = dict(np.load(gaze_file))
        marker_data = dict(np.load(marker_file))
        
        # Load parameters from stored yaml files
        param_fpath = PARAM_DIR / f'error-{param_tag}.yaml'
        kwargs = utils.read_yaml(param_fpath)
        fn = kwargs.pop('fn')
        eye = gaze_file.name.split('-')[1]
        func = utils.get_function(fn)
        print("Computing error...")
        error = func(marker_data, gaze_data, **kwargs)

        # Detect failure in some more subtle way?
        failed = False
    except:
        # Print something?
        failed = True

    # Manage ouptut file
    if failed:
        fpath_fail.open(mode='w')
        return fpath
    else:
        np.savez(fpath, **error)
        return fpath


# correct pupil slippage
# TO COME


def split_time(marker_time_file, marker_type):
    """Takes as input marker_times.yaml and a key ('calibration_frames' or 'validation_frames'
    (note these are intentionally vedb specific), returns indices for epochs

    marker_type : str
        either 'calibration_frames' or 'validation_frames'
    """
    marker_times = utils.read_yaml(marker_time_file)
    epoch_frames = marker_times[marker_type]
    # Add epoch number as third parameter
    epochs = list(range(len(epoch_frames)))
    start_frames = [x[0] for x in epoch_frames]
    end_frames = [x[1] for x in epoch_frames]
    
    return epochs, start_frames, end_frames


def check_files(output_dir, template, key_list=None):
    if key_list is None:
        files_out = sorted(list(output_dir.glob(template)))
        failed_files = ['failed' in x.name for x in files_out]
    else:
        files_out = {}
        failed_files = {}
        for k in key_list:
            files_out[k] = sorted(list(output_dir.glob(template%k)))
            failed_files[k] = ['failed' in x.name for x in files_out[k]]
    return files_out, failed_files



### --- Workflows --- ###
def pipeline_vedb(session,
                  pupil_tag='pylids_pupils_eyelids_v2',
                  pupil_detrend_tag=None,
                  calibration_marker_tag='circles_halfres',
                  calibration_split_tag=None,
                  calibration_cluster_tag='cluster_circles',
                  validation_marker_tag='checkerboard_halfres_4x7squares',
                  validation_split_tag=None, 
                  validation_cluster_tag='cluster_checkerboards',
                  calibration_tag='calibration-monocular_tps_cv_cluster_median_conf75_cut3std',
                  gaze_tag='default_mapper',
                  error_tag='error-smooth_tps_cv_clust_med_outlier4std_conf75', 
                  calibration_epoch=0,
                  input_base=BASE_DIR,
                  output_base=PYDRA_OUTPUT_DIR,
                  is_verbose=False,
                  ):
    """Build a gaze pipeline based on a VEDB folder and tags for each step of gaze processing.
    
    This contains assumptions for file structures related to VEDB, i.e. that world video
    and eye video files will have certain names and be in certain locations.

    It splits epochs based on EITHER a marker_times.yaml file (if it exists) or by 
    epoch splitting based on assumptions (see marker_parsing.py)

    Parameters
    ----------
    session is a string identifier for a vedb session, e.g. '2021_02_27_10_12_44'

    'eyes' input is used for calibration, gaze, & error estimation, when we may use average
    or binocular estimates of gaze. Calibration is handled separately from this parameter, 
    because gaze may be simply an average of two separte 
    """
    # Input folder
    input_dir = input_base.expanduser() / session
    # Output folder - For now, assume output to separate folder structure from input 
    output_dir = output_base.expanduser() / session
    # Hashes of inputs for steps with too many inputs for a_b_c type filename construction
    calibration_args = [x for x in [calibration_marker_tag, calibration_split_tag, \
                                       calibration_cluster_tag, f'epoch{calibration_epoch:02d}', \
                                       pupil_tag, pupil_detrend_tag] if not x is None]
    calibration_input_hash = hash('-'.join(calibration_args))
    error_args = [x for x in [calibration_marker_tag, calibration_split_tag, \
                                       calibration_cluster_tag, f'epoch{calibration_epoch:02d}', \
                                       pupil_tag, pupil_detrend_tag, \
                                       calibration_tag, gaze_tag,
                                       validation_marker_tag, validation_split_tag, validation_cluster_tag, \
                                       ] if not x is None]
    error_input_hash = hash('-'.join(error_args))

    # # asterisk to allow for failed attempts and subsequent error handling. All outputs should be npz files if successful.
    # output_file_names = dict(
    #     pupil=f'pupil_detection-%s-{pupil_tag}.*',
    #     pupil_detrend=f'pupil-%s-{pupil_tag}-{pupil_detrend_tag}.*', # blank for eye
    #     # by epoch (and/or not): - fixed calibration epoch for now
    #     calibration_marker=f'marker-{calibration_marker_tag}_*.*',
    #     # by epoch (and/or not): - fixed calibration epoch for now
    #     calibration_cluster=f'marker-{calibration_marker_tag}-{calibration_cluster_tag}-*.*', 
    #     # Choice for division into epochs 
    #     # a. check marker_times.yaml file for n epochs?
    #     # b. have n_<cal or val>_epochs as an input? 
    #     # c. file names are templates for anything with epochs. making a template flexible to 'all' or '00' may be annoying.
    #     # by epoch (and/or not):
    #     # Going with C, seems simplest at this level, probably later
    #     validation_marker=f'marker-{validation_marker_tag}-*.*', # blank for epoch
    #     # by epoch (and/or not):
    #     validation_cluster=f'marker-{validation_marker_tag}-{validation_cluster_tag}-*.*',
    #     calibration=f'calibration-%s-{calibration_tag}-{calibration_input_hash}.*', # blank for eye
    #     gaze=f'gaze-%s-{gaze_tag}_{calibration_tag}-{calibration_input_hash}.*', # blank for eye
    #     error=f'error-%s-{error_tag}_{error_input_hash}-*.npz', # blank for eye, epoch string
    # )
    
    eye_dict = dict(left=1, right=0)
    input_file_names = dict(
        eye_video_file=dict((eye, input_dir / f'eye{eye_dict[eye]:d}{eye_template}.mp4') for eye in ['left','right']),
        eye_time_file=dict((eye, input_dir / f'eye{eye_dict[eye]:d}_{timestamp_template}.npy') for eye in ['left','right']),
        world_video_file=input_dir / f'world{world_template}.mp4',
        world_time_file=input_dir / f'world_{timestamp_template}.npy',
        marker_time_file=input_dir / 'marker_times.yaml',
    )

    # For calibration
    video_dimensions = file_io.list_array_shapes(input_file_names['world_video_file'])[2:0:-1]

    if calibration_marker_tag is None:
        calibration_markers = []    # Split into calibration / validation epochs
    else:
        if input_file_names['marker_time_file'].exists():
            ## Calibration
            # Do manual split, select epoch, detect, cluster
            calibration_epochs, calibration_start_frames, calibration_end_frames = split_time(
                    marker_file=input_file_names['marker_time_file'],
                    marker_type='calibration_frames',
                    )
            # Run calibration on that epoch only
            calibration_markers = marker_detection(
                video_file=input_file_names['world_time_file'],
                time_file=input_file_names['world_time_file'],
                param_tag=calibration_marker_tag,
                output_dir=output_dir,
                epoch=calibration_epoch,
                start_frame=calibration_start_frames[calibration_epoch],
                end_frame=calibration_end_frames[calibration_epoch],
                is_verbose=is_verbose,)
        else:
            raise NotImplementedError('Must have marker_times.yaml file for now')
            # Detect markers
            calibration_markers = marker_detection(
                name='calibration_detection',
                video_file=input_file_names['world_time_file'],
                time_file=input_file_names['world_time_file'],
                param_tag=calibration_marker_tag,
                output_dir=output_dir,
                is_verbose=is_verbose,)
            # Split into epochs and select epoch
            if calibration_split_tag is None:
                pass
            else:
                # FIX ME. Split only.
                calibration_markers = marker_filtering(
                    marker_fname=calibration_markers,
                    session_folder=session.folder,  # gaze_pipeline.lzin.folder,
                    param_tag=cal_marker_split_tag,
                    marker_type='concentric_circle',
                    is_verbose=is_verbose,
                    )
    ## Validation
    validation_markers = []
    if validation_marker_tag is not None:
        if input_file_names['marker_time_file'].exists():
            # Do manual split
            validation_epochs, validation_start_frames, validation_end_frames = split_time(
                marker_file=input_file_names['marker_time_file'],
                marker_type='validation_frames',
                )
            # Run detection
            validation_markers = []
            for this_epoch, this_start, this_end in zip(validation_epochs, validation_start_frames, validation_end_frames):
                tmp_vm = marker_detection(
                    name='validation_detection',
                    video_file=input_file_names['world_time_file'],
                    time_file=input_file_names['world_time_file'],
                    param_tag=validation_marker_tag,
                    output_dir=output_dir,
                    epoch=this_epoch,
                    start_frame=this_start,
                    end_frame=this_end,
                    is_verbose=is_verbose,)
                validation_markers.append(tmp_vm)
        else:
            ## No yaml file with marker times
            raise NotImplementedError('Must have marker_times.yaml file for now')


    # Cluster calibration marker output
    if calibration_cluster_tag is None:
        calibration_markers_clustered = []
    else:
        calibration_markers_clustered = marker_clustering(
            marker_file=calibration_markers,
            time_file=input_file_names['world_time_file'],
            param_tag=calibration_cluster_tag,
            output_dir=output_dir,
            is_verbose=False
        )

    # Cluster already-split validation marker outputs
    validation_markers_clustered = []
    if validation_cluster_tag is not None:
        # Run clustering 
        validation_markers_clustered = []
        for vm in validation_markers:
            tmp_vm = marker_clustering(
                marker_file=vm, 
                time_file=input_file_names['world_time_file'],
                param_tag=validation_cluster_tag,
                output_dir=output_dir,
                is_verbose=is_verbose,
            )
            validation_markers_clustered.append(tmp_vm)

    ## Pupil detection
    if pupil_tag is None:
        pupils = {}
    else:
        p_left = pupil_detection(
                name='pupil_left',
                eye_video_file=input_file_names['eye_video_file']['left'],
                eye_time_file=input_file_names['eye_time_file']['left'],
                param_tag=pupil_tag,
                eye='left',
                output_dir=output_dir,
                is_verbose=is_verbose,
                )
        p_right = pupil_detection(
                name='pupil_right',
                eye_video_file=input_file_names['eye_video_file']['right'],
                eye_time_file=input_file_names['eye_time_file']['right'],
                param_tag=pupil_tag,
                eye='right',
                output_dir=output_dir,
                is_verbose=is_verbose,
                )
        pupils = {'left':p_left, 'right':p_right}

    # Computing calibration 
    calibration_out = {}
    if calibration_tag is not None:        
        if any([x is None for x in [pupil_tag, calibration_marker_tag, calibration_cluster_tag]]):
                raise ValueError(
                    "Must specify tags for all previous steps to compute calibration!")
        if 'binocular' in calibration_tag:
            # Run 
            calibration_out['both'] = compute_calibration(
                marker_file=calibration_markers_clustered,
                pupil_files=[pupils['left'], pupils['right']],
                input_hash=calibration_input_hash,
                param_tag=calibration_tag,
                video_dimensions=video_dimensions,
                output_dir=output_dir,
                eye='both',
                is_verbose=is_verbose,
                )
        elif 'monocular' in calibration_tag:
            for eye in ['left','right']:
                calibration_out[eye] = compute_calibration(
                    marker_file=calibration_markers_clustered,
                    pupil_files=pupils[eye],
                    input_hash=calibration_input_hash,
                    param_tag=calibration_tag,
                    video_dimensions=video_dimensions,
                    output_dir=output_dir,
                    eye=eye,
                    is_verbose=is_verbose,
                )
        else:
            raise ValueError(f"Unknown calbration tag {calibration_tag}")
            
    # Mapping gaze    
    gaze = {}
    if gaze_tag is not None:
        # Check for presence in pl_elements
        if 'monocular' in calibration_tag:
            eyes = ['left', 'right']
        else:
            eyes = ['both']
        for eye in eyes:
            if eye == 'both':
                pupil_files = [pupils['left'],pupils['right']]
            else:
                pupil_files = pupils[eye]
            gaze[eye] = map_gaze(
                pupil_files=pupil_files,
                calibration_file=calibration_out[eye],
                calibration_epoch=calibration_epoch,
                param_tag=gaze_tag,
                eye=eye,
                is_verbose=is_verbose)
    
    error = {}
    if error_tag is not None:
        if any([x is None for x in [pupil_tag, 
                                    calibration_marker_tag, 
                                    calibration_cluster_tag,
                                    validation_marker_tag,
                                    validation_cluster_tag,
                                    calibration_tag,
                                    gaze_tag,
                                    ]]):
            raise ValueError("To compute gaze error, tags for all other steps must be specified")
        # 'eyes' defined above in calibration computation, and all steps must run
        # to compute error, so it will be defined.
        for eye in eyes:
            error[eye] = []
            for vm in validation_markers_clustered:
                tmp_e = compute_error(
                    gaze_file=gaze[eye],
                    marker_file=vm,
                    param_tag=error_tag,
                    output_dir=output_dir,
                    eye=eye,
                    input_hash=error_input_hash,
                    is_verbose=is_verbose,)
                error[eye].append(tmp_e)

    return dict(pupils=pupils, calibration_markers=calibration_markers, calibration_markers_clustered=calibration_markers_clustered,
                validation_markers=validation_markers, validation_markers_clustered=validation_markers_clustered, calibration=calibration_out, gaze=gaze, error=error)
"""
# Inputs: 
session folder
list of params: [pupil_detection, marker_detection, marker_filtering, calibration, gaze_estimation, ]

"""
