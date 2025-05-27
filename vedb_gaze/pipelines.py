# Full pipelines for gaze estimation
import numpy as np
import tqdm
import tqdm.notebook
import file_io
import typing
import copy
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
@pydra.mark.task
def select(x_list, index):
    """Utility to choose one of multiple outputs from a step for input to the next step"""
    return x_list[index]

@pydra.mark.task
@pydra.mark.annotate({'return': {'pupil_locations': typing.Any}})
def pupil_detection(eye_video_file,
        eye_time_file, 
        param_tag, 
        output_dir,
        eye='left', 
        is_verbose=False):
    """Run a pupil detection function as a pipeline step
    
    Parameters
    ----------
    session folder : str
        folder for eye video session to run

    """
    if is_verbose:
        print("\n=== Finding pupil locations ===\n")
    if is_notebook():
        progress_bar = tqdm.notebook.tqdm
    else:
        progress_bar = tqdm.tqdm

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


@pydra.mark.task
@pydra.mark.annotate({'return': {'marker_locations': typing.Any}})
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

@pydra.mark.task
@pydra.mark.annotate({'return': {'marker_locations': typing.List}})
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

    # Get marker file
    marker_data = dict(np.load(marker_file))
    _, orig_tag, epoch_str = marker_file.name.split('-')
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

@pydra.mark.task
@pydra.mark.annotate({'return': {'marker_locations': typing.Any}})
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

    # Get marker file
    marker_data = dict(np.load(marker_file))
    _, orig_tag, epoch_str = marker_file.name.split('-')
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


@pydra.mark.task
@pydra.mark.annotate({'return': {'calibration_file': typing.Any}})
def compute_calibration(marker_file,
                        pupil_files,
                        input_hash,
                        param_tag,
                        video_dimensions,
                        output_dir,
                        eye=None,
                        is_verbose=False):
    """
    param_tag should be informative;
    params it points to specify:
        calibration_type
        min_confidence
        (lambda)
        (etc)
    
    """
    if is_verbose:
        print("\n=== Computing calibration ===\n")
    # Handle inputs
    if not isinstance(pupil_files, (list, tuple)):
        pupil_files = [pupil_files]

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
    fname = f'calibration-{eye}-{param_tag}-{input_hash}.npz'
    if failed:
        fpath = output_dir / fname.replace('.npz','.failed')
        fpath.open()
    else:
        fpath = output_dir / fname
        cal.save(fpath)
    return fpath


@pydra.mark.task
@pydra.mark.annotate({'return': {'gaze_locations': typing.Any}})
def map_gaze(pupil_files,
             calibration_file,
             param_tag,
             output_dir,
             is_verbose=False):
    if is_verbose:
        print("\n=== Computing gaze ===\n")
    # Handle inputs
    if not isinstance(pupil_files, (list, tuple)):
        pupil_files = [pupil_files]

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
    fname = f'gaze_{eye}-{param_tag}-{calibration_tag}-{input_hash}.npz'
    fpath = os.path.join(output_dir, fname)
    if failed:
        fpath = output_dir / fname.replace('.npz','.failed')
        fpath.open(mode='w')
    else:
        fpath = output_dir / fname
        np.savez(fpath, **data)

    return fpath


@pydra.mark.task
@pydra.mark.annotate({'return': {'error': typing.Any}})
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
        gaze_data = dict(np.load(gaze_file))
        marker_data = dict(np.load(marker_file))
        _, orig_tag, cluster_tag, epoch_str = marker_file.name.split('-')
        # Load parameters from stored yaml files
        param_fpath = PARAM_DIR / f'error-{param_tag}.yaml'
        kwargs = utils.read_yaml(param_fpath)

        fn = kwargs.pop('fn')
        eye = gaze_file.name.split('-')[1]
        func = utils.get_function(fn)
        print("Computing error...")
        error = func(marker_data, gaze_data, **kwargs)

        # Detect failure?
        failed = False  # len(data) == 0
    except:
        # Print something?
        failed = True

    # Manage ouptut file
    fname = f'error-{eye}-{param_tag}-{input_hash}-{epoch_str}.npz'
    if failed:
        fpath = output_dir / fname.replace('.npz','.failed')
        fpath.open(mode='w')
        return fpath
    else:
        fpath = output_dir / fname
        np.savez(fpath, **error)
        return fpath


# correct pupil slippage



@pydra.mark.task
@pydra.mark.annotate({'return': {'epochs': typing.List,
                                 'start_frames':typing.List,
                                 'end_frames':typing.List}})
#@python.define(outputs=["epochs","start_frames", "end_frames"])
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
def make_pipeline_vedb(session,
                  pupil_tag='pylids_pupils_eyelids_v2',
                  pupil_detrend_tag=None,
                  calibration_marker_tag='circles_halfres',
                  calibration_split_tag=None,
                  calibration_cluster_tag='cluster_circles',
                  validation_marker_tag='checkerboard_halfres_4x7squares',
                  validation_split_tag=None, # ??
                  validation_cluster_tag='cluster_checkerboards',
                  calibration_tag='monocular_tps_xxx-xxx-blah', # ??
                  gaze_tag='default_mapper', # ??
                  error_tag='smooth_tps_default', # ??
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

    # asterisk to allow for failed attempts and subsequent error handling. All outputs should be npz files if successful.
    output_file_names = dict(
        pupil=f'pupil_detection-%s-{pupil_tag}.*',
        pupil_detrend=f'pupil-%s-{pupil_tag}-{pupil_detrend_tag}.*', # blank for eye
        # by epoch (and/or not): - fixed calibration epoch for now
        calibration_marker=f'marker-{calibration_marker_tag}_*.*',
        # by epoch (and/or not): - fixed calibration epoch for now
        calibration_cluster=f'marker-{calibration_marker_tag}-{calibration_cluster_tag}-*.*', 
        # Choice for division into epochs 
        # a. check marker_times.yaml file for n epochs?
        # b. have n_<cal or val>_epochs as an input? 
        # c. file names are templates for anything with epochs. making a template flexible to 'all' or '00' may be annoying.
        # by epoch (and/or not):
        # Going with C, seems simplest at this level, probably later
        validation_marker=f'marker-{validation_marker_tag}-*.*', # blank for epoch
        # by epoch (and/or not):
        validation_cluster=f'marker-{validation_marker_tag}-{validation_cluster_tag}-*.*',
        calibration=f'calibration-%s-{calibration_tag}-{calibration_input_hash}.*', # blank for eye
        gaze=f'gaze-%s-{gaze_tag}_{calibration_tag}-{calibration_input_hash}.*', # blank for eye
        error=f'error-%s-{error_tag}_{error_input_hash}-*.npz', # blank for eye, epoch string
    )
    
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


    # Create workflow
    gaze_pipeline = pydra.Workflow(name='full_gaze_error_vedb', 
                                   input_spec=['folder', ],
                                   folder=session,
                                   )
    # Split into calibration / validation epochs
    if input_file_names['marker_time_file'].exists():
        ## Calibration
        # do manual split, select epoch, detect, cluster
        gaze_pipeline.add(split_time(
                name='calibration_epoch_split',
                marker_file=input_file_names['marker_time_file'],
                marker_type='calibration_frames',
                ))
        # Run detections on manual split times
        if calibration_marker_tag is None:
            calibration_marker_out = []
        else:
            calibration_all_files, calibration_marker_failed = check_files(output_dir, output_file_names['calibration_marker'], )
            calibration_all_files = [x for x in calibration_all_files if not 'epochall' in x]
            if len(calibration_all_files) > 0:
                calibration_marker_out = []
                calibration_marker_inpt = calibration_all_files[calibration_epoch]
                if calibration_marker_failed[calibration_epoch]:
                    raise Exception('Calibration marker detection failed, not possible to continue')
            else:
                # Select ONE calibration epoch to run
                gaze_pipeline.add(select(name='calibration_epoch_choice',
                            x_list=gaze_pipeline.calibration_epoch_split.lzout.epochs,
                            index=calibration_epoch))
                gaze_pipeline.add(select(name='calibration_start_choice',
                            x_list=gaze_pipeline.calibration_epoch_split.lzout.start_frames,
                            index=calibration_epoch))
                gaze_pipeline.add(select(name='calibration_end_choice',
                            x_list=gaze_pipeline.calibration_epoch_split.lzout.end_frames,
                            index=calibration_epoch))
                # Run calibration on that epoch only
                gaze_pipeline.add(marker_detection(
                    name='calibration_detection',
                    video_file=input_file_names['world_time_file'],
                    time_file=input_file_names['world_time_file'],
                    param_tag=calibration_marker_tag,
                    output_dir=output_dir,
                    epoch=calibration_epoch,
                    start_frame=gaze_pipeline.calibration_start_choice.lzout.out,
                    end_frame=gaze_pipeline.calibration_end_choice.lzout.out,
                    is_verbose=is_verbose,) )
                calibration_marker_out = [("calibration_marker",
                                        gaze_pipeline.calibration_detection.lzout.marker_locations), ]
                calibration_marker_inpt = gaze_pipeline.calibration_detection.lzout.marker_locations
        
        ## Validation
        gaze_pipeline.add(split_time(
            name='validation_epoch_split',
            marker_file=input_file_names['marker_time_file'],
            marker_type='validation_frames',
            ))
        # run detections on split manual times
        if validation_marker_tag is None:
            validation_marker_out = []
        else:
            validation_all_files, validation_failed = check_files(output_dir, output_file_names['validation_marker'], )
            validation_all_files = [x for x in validation_all_files if (not 'epochall' in x)]
            if len(validation_all_files) > 0:
                # Test for all failure
                if all(validation_failed):
                    validation_marker_out = []
                    validation_marker_inpt = []
                else:
                    validation_marker_out = []
                    validation_marker_inpt = validation_all_files

            else:
                # Run detection
                gaze_pipeline.add(marker_detection(
                    name='validation_detection',
                    video_file=input_file_names['world_time_file'],
                    time_file=input_file_names['world_time_file'],
                    param_tag=validation_marker_tag,
                    output_dir=output_dir,
                    # Lists for these... should loop
                    epoch=gaze_pipeline.validation_epoch_split.lzout.epochs,
                    start_frame=gaze_pipeline.validation_epoch_split.lzout.start_frames,
                    end_frame=gaze_pipeline.validation_epoch_split.lzout.end_frames,
                    is_verbose=is_verbose,) )

                validation_marker_out = [('validation_marker', 
                    gaze_pipeline.validation_detection.lzout.marker_locations),]
                validation_marker_split_inpt = gaze_pipeline.validation_detection.lzout.marker_locations

    else:
        ## No yaml file with marker times
        raise Exception("Gotta have yaml. (TO DO: Fix me)")
        # Do full detect, split, select, cluster 
        # Calibration marker detection
        if calibration_marker_tag is None:
            calibration_marker_out = []
        else:
            calibration_all_file = list(output_dir.glob(output_file_names['calibration_marker']%'epochall'))
            if len(calibration_all_file)==1: 
                # Suppress output for completed steps.
                # See commented out part and above for alternative.
                calibration_marker_out = [] #("calibration_marker",
                                            #pl_elements['calibration_marker_all'].fname), ]
                calibration_marker_inpt = calibration_all_file[0]
            else:
                gaze_pipeline.add(marker_detection(
                    name='calibration_detection',
                    video_file=input_file_names['world_time_file'],
                    time_file=input_file_names['world_time_file'],
                    param_tag=calibration_marker_tag,
                    output_dir=output_dir,
                    is_verbose=is_verbose,)
                )
                calibration_marker_out = [("calibration_marker_all",
                                        gaze_pipeline.calibration_detection.lzout.marker_locations), ]
                calibration_marker_inpt = gaze_pipeline.calibration_detection.lzout.marker_locations
        # Calibration split
        if calibration_split_tag is None:
            pass
        
        # Calibration marker filter & split
        if calibration_cluster_tag is None:
            calibration_cluster_out = []
        else:
            if 'calibration_marker_filtered' in pl_elements:
                # Make a list long enough to contain an index for the calibration marker epoch in question
                # Note: this is shady, sorry future me for bugs this will prob generate
                cal_epoch_fnames = [''] * (calibration_epoch + 1)
                cal_epoch_fnames[calibration_epoch] = pl_elements['calibration_marker_filtered'].fname
                # TEMP: Suppress output for completed steps.
                # May want to reconcile this later somehow. Or not.
                calibration_cluster_out = []#("calibration_epochs",
                                # cal_epoch_fnames), ]
                calibrations_filtered = cal_epoch_fnames
            else:
                if calibration_marker_tag is None:
                    raise ValueError("Must specify calibration_marker_tag to conduct calibration marker filtering!")
                gaze_pipeline.add(marker_filtering(
                    name='calibration_marker_filtering',
                    marker_fname=calibration_marker_inpt,
                    session_folder=session.folder,  # gaze_pipeline.lzin.folder,
                    fn=cal_marker_filter_params.fn,
                    param_tag=cal_marker_filter_param_tag,
                    db_tag='-'.join([calibration_marker_tag, cal_marker_filter_param_tag]),
                    marker_type='concentric_circle',
                    db_name=db_name,
                    is_verbose=is_verbose,
                    )
                )
                calibration_cluster_out = [("calibration_epochs",
                                gaze_pipeline.calibration_marker_filtering.lzout.marker_locations),]
                calibrations_filtered = gaze_pipeline.calibration_marker_filtering.lzout.marker_locations
    # Cluster calibration marker output
    if calibration_cluster_tag is None:
        calibration_cluster_out = []
    else:
        # calibration_clustered_file, calibration_cluster_failed = check_files(output_dir, output_file_names['calibration_cluster'], )
        
        # if (len(calibration_clustered_file) > 0) and calibration_clustered_file[calibration_epoch].exists():
        #     if calibration_cluster_failed[calibration_epoch]:
        #         raise Exception('Calibration clustering failed, cannot continue')
        #     else:
        #         calibration_cluster_out = []
        #         calibration_cluster_inpt = calibration_clustered_file[calibration_epoch]
        # else:
        gaze_pipeline.add(marker_clustering(
            name='calibration_clustering',
            marker_file=calibration_marker_inpt,
            time_file=input_file_names['world_time_file'],
            param_tag=calibration_cluster_tag,
            output_dir=output_dir,
            is_verbose=False
        ))
        calibration_cluster_out = [('calibration_cluster',
                                    gaze_pipeline.calibration_clustering.lzout.marker_locations), ]
        calibration_cluster_inpt = gaze_pipeline.calibration_clustering.lzout.marker_locations

    # Cluster already-split validation marker outputs
    if validation_cluster_tag is None:
        validation_cluster_out = []
    else:
        # # One-off, keep non-failed ones, it's fine. Epoch will be manifest in the file name.
        # validation_clustered_files = sorted(list(output_dir.glob(output_file_names['validation_cluster']%'epoch*')))
        # if len(validation_clustered_files) > 0:
        #     # Return extant files
        #     validation_clustered_files = [x for x in validation_clustered_files if not 'failed' in x]
        #     if len(validation_clustered_files) > 0:
        #         validation_cluster_out = []
        #         validation_cluster_inpt = validation_clustered_files
        #     else:
        #         pass # OOPS all failed # FIX ME

        # else:
        # Run clustering 
        if validation_marker_tag is None:
            raise ValueError(
                "Must specify validation_marker_tag to conduct validation marker filtering!")
        gaze_pipeline.add(marker_clustering(
            name='validation_marker_clustering',
            marker_file=validation_marker_split_inpt, # a list
            time_file=input_file_names['world_time_file'],
            param_tag=validation_cluster_tag,
            output_dir=output_dir,
            is_verbose=is_verbose,
        )
        )
        validation_cluster_out = [("validation_cluster",
                            gaze_pipeline.validation_marker_clustering.lzout.marker_locations), ]
        validation_cluster_inpt = gaze_pipeline.validation_marker_clustering.lzout.marker_locations


    # Pupil detection
    if pupil_tag is None:
        pupil_out = []
    else:
        # Gotta account for failed
        pupil_files, pupil_files_failed = check_files(output_dir, output_file_names['pupil'], key_list=('left','right') )
        if pupil_files['left'][0].exists() and pupil_files['right'][0].exists():
            if any(pupil_files_failed['left']) or any(pupil_files_failed['right']):
                raise Exception('Pupils not detected, cannot continue')
            # Suppress output for completed steps. Limits verbosity of ouput, 
            # also possibly leads to inconsistent outputs of pipeline. Alternative
            # to this suppression is commented out below.
            pupil_out = []
                #("pupil_left",
                # pl_path),
                #("pupil_right",
                # pr_path),
            #]
            assert (len(pupil_files['left']) == 1) and (len(pupil_files['right']) == 1), 'Multiple pupil detection files found for one eye!'
            pupil_inpt = [pupil_files['left'][0], pupil_files['right'][0]]
        else:
            gaze_pipeline.add(pupil_detection(
                name='pupil_left',
                eye_video_file=input_file_names['eye_video_file']['left'],
                eye_time_file=input_file_names['eye_time_file']['left'],
                param_tag=pupil_tag,
                eye='left',
                output_dir=output_dir,
                is_verbose=is_verbose,
                )
            )
            gaze_pipeline.add(pupil_detection(
                name='pupil_right',
                eye_video_file=input_file_names['eye_video_file']['right'],
                eye_time_file=input_file_names['eye_time_file']['right'],
                param_tag=pupil_tag,
                eye='right',
                output_dir=output_dir,
                is_verbose=is_verbose,
                )
            )
            pupil_out = [
                ("pupil_left", 
                gaze_pipeline.pupil_left.lzout.pupil_locations),
                ("pupil_right",
                gaze_pipeline.pupil_right.lzout.pupil_locations),
                ]
            pupil_inpt = [x[1] for x in pupil_out]

    # Computing calibration 
    if calibration_tag is None:
        calibration_out = []
    else:
        if any([x is None for x in [pupil_tag, calibration_marker_tag, calibration_cluster_tag]]):
                raise ValueError(
                    "Must specify tags for all previous steps to compute calibration!")
        if 'binocular' in calibration_tag:
            # calibration_files, calibration_failed = check_files(output_dir, output_file_names['calibration']%'both')
            # if len(calibration_files) == 1:
            #     calibration_out = []
            #         #('calibration_both', pl_elements['calibration']['both'].fname)]
            #     calibration_inpt = calibration_files
            # elif len(calibration_files) > 1:
            #     raise Exception("More than one file for specified calibration found!")
            # else:
            # Run 
            gaze_pipeline.add(compute_calibration(
                name='calibration_both',
                marker_file=calibration_cluster_out,
                pupil_files=pupil_inpt,
                input_hash=calibration_input_hash,
                param_tag=calibration_tag,
                video_dimensions=video_dimensions,
                output_dir=output_dir,
                eye='both',
                is_verbose=is_verbose,
                )
            )
            calibration_out = [('calibration_both', gaze_pipeline.calibration_both.lzout.calibration_file)]
        elif 'monocular' in calibration_tag:
            # calibration_files, calibration_failed = check_files(output_dir, output_file_names['calibration'], key_list=('left','right'))
            # if (len(calibration_files['left'])==1) and (len(calibration_files['right'])==1):
            #     if calibration_failed['left'][0] or calibration_failed['right'][0]:
            #         calibration_out = []
            #         calibration_inpt = []
            #     else:
            #         calibration_out = [] #('calibration_left', pl_elements['calibration']['left'].fname),
            #                              #('calibration_right', pl_elements['calibration']['right'].fname, )]
            #         calibration_inpt = [calibration_files['left'][0], 
            #                             calibration_files['right'][0]]
            # elif len(calibration_files['left']==0) and (len(calibration_files['right'])==0):
            # There may be a more compact way to do this with a pydra splitter, 
            # but I can't see how w/ the extra kwarg for `eye`
            gaze_pipeline.add(compute_calibration(
                name='calibration_left',
                marker_file=gaze_pipeline.calibration_epoch_choice.lzout.out,
                pupil_files=pupil_inpt[0],
                input_hash=calibration_input_hash,
                param_tag=calibration_tag,
                video_dimensions=video_dimensions,
                output_dir=output_dir,
                eye='left',
                is_verbose=is_verbose,
            )
            )
            gaze_pipeline.add(compute_calibration(
                name='calibration_right',
                marker_file=gaze_pipeline.calibration_epoch_choice.lzout.out,
                pupil_files=pupil_inpt[0],
                input_hash=calibration_input_hash,
                param_tag=calibration_tag,
                video_dimensions=video_dimensions,
                output_dir=output_dir,
                eye='right',
                is_verbose=is_verbose,
            )
            )
            calibration_out = [('calibration_left',gaze_pipeline.calibration_left.lzout.calibration_file), 
                                ('calibration_right',gaze_pipeline.calibration_right.lzout.calibration_file, )]
            # else:
            #     # Weird circumstance, one succeeded, one failed
            #     raise Exception("Odd circumstances with calibration, either too many or one failed")
            calibration_inpt = [x[1] for x in calibration_out]
        else:
            raise ValueError(f"Unknown calbration tag {calibration_tag}")
            
    # Mapping gaze
    if gaze_tag is None:
        gaze_out = []
    else:
        # Check for presence in pl_elements
        if 'monocular' in calibration_tag:
            eyes = ['left', 'right']
            pupil_files =  pupil_inpt
        else:
            eyes = ['both']
            pupil_files = [pupil_inpt]
        # This is a mess, needs re-doing if pydra caching doesn't work well
        gaze_out = []
        # # Check for extant files
        # gaze_files, gaze_failed = check_files(output_dir, output_file_names['gaze'], key_list=eyes)
        # for e, eye in enumerate(eyes):
        #     if len(gaze_files[eye]) == 1:
        #         gaze_inpt = gaze_files[0]
        #if 'gaze' in pl_elements:
        #    gaze_inpt = [pl_elements['gaze'][eye].fname for eye in eyes]
        #else:
        for e, eye in enumerate(eyes):
            gaze_name = 'gaze_%s'%eye
            gaze_pipeline.add(map_gaze(
                name=gaze_name,
                pupil_files=pupil_files[e],
                calibration_file=calibration_inpt[e],
                calibration_epoch=calibration_epoch,
                param_tag=gaze_tag,
                eye=eye,
                is_verbose=is_verbose)
                )
            gaze_out.append((gaze_name, getattr(gaze_pipeline, gaze_name).lzout.gaze_locations))
        gaze_inpt = [x[1] for x in gaze_out]

    
    error_out = []
    if error_tag is not None:
        if any([x is None for x in [pupil_tag, 
                                    calibration_marker_tag, 
                                    calibration_cluster_tag,
                                    validation_marker_tag,
                                    validation_cluster_tag,
                                    calibration_tag,
                                    gaze_tag,
                                    ]]):
            raise ValueError("To compute gaze error, all other steps must be specified")
        # 'eyes' defined above in calibration computation, and all steps must run
        # to compute error, so it will be defined.
        for e, eye in enumerate(eyes):
            error_name = f'compute_error_{eye}'
            gaze_pipeline.add(compute_error(
                name=error_name,
                gaze_file=gaze_inpt[e],
                marker_file=validation_cluster_inpt,
                param_tag=error_tag,
                output_dir=output_dir,
                eye=eye,
                is_verbose=is_verbose,
            ).split('marker_file')
            )
            error_out.append(
                (error_name, getattr(gaze_pipeline, error_name).lzout.error))
    gaze_pipeline.set_output(pupil_out + calibration_marker_out + calibration_cluster_out + \
        validation_marker_out + validation_cluster_out + calibration_out + gaze_out + error_out)

    return gaze_pipeline
"""
# Inputs: 
session folder
list of params: [pupil_detection, marker_detection, marker_filtering, calibration, gaze_estimation, ]

"""
