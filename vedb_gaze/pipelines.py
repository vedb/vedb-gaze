# Calibration script
# import vm_preproc as vmp
# import vedb_store
import numpy as np
import tqdm
import tqdm.notebook
import vedb_store
import pydra
import typing
import os

from . import pupil_detection_pl, calibrate_pl, marker_detection, utils
from .options import config




BASE_DIR = config.get('paths', 'base_dir')
PYDRA_OUTPUT_DIR = config.get('paths', 'pydra_cache') # '/pomcloud0/vedb/pydra_cache/'
BASE_OUTPUT_DIR = config.get('paths', 'output_dir') # '/pomcloud0/vedb/processed/'
PARAM_DIR = os.path.join(os.path.split(__file__)[0], 'config/')


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


def get_function(function_name):
    """Load a function to a variable by name

    Parameters
    ----------
    function_name : str
        string name for function (including module)
    """
    import importlib
    fn_path = function_name.split('.')
    module_name = '.'.join(fn_path[:-1])
    fn_name = fn_path[-1]
    module = importlib.import_module(module_name)
    func = getattr(module, fn_name)
    return func



# @pydra.mark.task
# @pydra.mark.annotate({'return':{'pupil_locations': ty.Dict}})
# def calibration_step(session_folder, params, eye='left', db_name=None):
#     pass

@pydra.mark.task
@pydra.mark.annotate({'return': {'pupil_locations': typing.Any}})
def pupil_detection_step(session_folder, fn, param_tag, eye='left', db_tag=None, db_name=None):
    """Run a pupil detection function as a pipeline step
    
    Parameters
    ----------
    session folder : str
        folder for eye video session to run

    """
    if is_notebook():
        progress_bar = tqdm.notebook.tqdm
    else:
        progress_bar = tqdm.tqdm

    # Attempt loading from database if provided
    if db_name is not None:
        # This relies on proper configuration of vedb_store, to
        # know what hostname it should look for
        dbi = vedb_store.docdb.getclient(dbname=db_name)
        dbi.is_verbose = False
        # Find session to process
        session = dbi.query(1, type='Session', folder=session_folder)
        # Find input arguments for processing
        try:
            param_dict = dbi.query(1, type='ParamDictionary', fn=fn, tag=param_tag)
        except:
            print(f"Unknown parameter dictionary!\nfn={fn}\ntag={param_tag}")
            raise
        # Test for completed analysis stored in database
        try:
            pupil_data = dbi.query(1, type='PupilDetection',
                                   tag=db_tag,
                                   session=session._id,
                                   params=param_dict._id)
            return pupil_data.fpath
        except:
            # Not yet run
            pass
        # Prepare inputs
        eye_time_file, eye_video_file = session.paths["eye_%s" % eye]
        kwargs = param_dict.params

    else:
        eye_dict = dict(left=0, right=1)
        eye_time_file = os.path.join(
            session_folder, 'eye%d_timestamps.npy' % eye_dict[eye])
        eye_video_file = os.path.join(
            session_folder, 'eye%d.mp4' % eye_dict[eye])
        # Load parameters from stored yaml files
        fn_name = fn.split('.')[-1]
        param_fname = f'{fn_name}-{param_tag}.yaml'
        param_fpath = os.path.join(PARAM_DIR, param_fname)
        kwargs = utils.read_yaml(param_fpath)

    # convert `fn` string to callable function
    func = get_function(fn)
    # Optionally add progress bar, if supported
    default_kw = get_default_kwargs(func)
    if 'progress_bar' in default_kw:
        kwargs['progress_bar'] = progress_bar
    # Call function
    data = func(eye_video_file, eye_time_file, **kwargs,)
    # Manage ouptut file
    session_date = os.path.split(session_folder)[-1]
    if db_name is None:
        sdir = os.path.join(PYDRA_OUTPUT_DIR, session_date)
        fname = f'pupil_detection_{eye}_{fn_name}_{param_tag}.npz'
    else:
        sdir = os.path.join(BASE_OUTPUT_DIR, session_date)
        pup_doc = vedb_store.PupilDetection(
            data=data,
            session=session,
            eye=eye,
            params=param_dict,
            tag=param_dict.tag + '_%s' % eye,
            dbi=dbi,
            _id=dbi.get_uuid())
        fname = pup_doc.fname
    # Detect failure
    if len(data['norm_pos']) == 0:
        fn_name = fn.split('.')[-1]
        ff, _ = os.path.splitext(fname)
        fname = f'{ff}_fail.txt'
        with open(os.path.join(sdir, fname), mode='w') as fid:
            fid.write("No pupils detected")
        fpath = os.path.join(sdir, fname)
        return fpath
    else:
        if db_name is None:
            fpath = os.path.join(sdir, fname)
            # Alternatively, rely on pydra to save... for now, that would seem
            # to complicate things.
            np.savez(fpath)
        else:
            pup_doc.save()
            fpath = pup_doc.fpath
    return fpath


@pydra.mark.task
@pydra.mark.annotate({'return': {'marker_locations': typing.Any}})
def marker_detection_step(session_folder,
                          fn,
                          param_tag,
                          db_tag=None,
                          marker_type=None,
                          db_name=None):
    """Run a pupil detection function as a pipeline step
    
    Parameters
    ----------
    session folder : str
        folder for eye video session to run

    """
    if is_notebook():
        progress_bar = tqdm.notebook.tqdm
    else:
        progress_bar = tqdm.tqdm
    # Get default database if not provided
    if db_name is not None:
        # This relies on proper configuration of vedb_store, to
        # know what hostname it should look for
        dbi = vedb_store.docdb.getclient(dbname=db_name)
        # Find session to process
        session = dbi.query(1, type='Session', folder=session_folder)
        # Find input arguments for processing
        param_dict = dbi.query(1, type='ParamDictionary', fn=fn, tag=param_tag)
        # Test for completed analysis stored in database
        try:
            marker_data = dbi.query(
                1, type='MarkerDetection',
                session=session._id,
                detection_params=param_dict._id,
                tag=db_tag,
                epoch_bytype='all')
            return marker_data.fpath
        except:
            # Not yet run
            pass
        # Prepare inputs
        world_time_file, world_video_file = session.paths["world_camera"]
        kwargs = param_dict.params

    else:
        world_time_file = os.path.join(session_folder, 'world_timestamps.npy')
        world_video_file = os.path.join(session_folder, 'world.mp4')
        # Load parameters from stored yaml files
        fn_name = fn.split('.')[-1]
        param_fname = f'{fn_name}-{param_tag}.yaml'
        param_fpath = os.path.join(PARAM_DIR, param_fname)
        kwargs = utils.read_yaml(param_fpath)

    # convert `fn` string to callable function
    func = get_function(fn)
    # Optionally add progress bar, if supported
    default_kw = get_default_kwargs(func)
    if 'progress_bar' in default_kw:
        kwargs['progress_bar'] = progress_bar
    data = func(world_video_file, world_time_file, **kwargs,)

    # Manage ouptut file
    session_date = os.path.split(session_folder)[-1]
    if db_name is None:
        sdir = os.path.join(PYDRA_OUTPUT_DIR, session_date)
        fname = f'marker_detection_{fn_name}_{param_tag}.npz'
    else:
        sdir = os.path.join(BASE_OUTPUT_DIR, session_date)
        mk_doc = vedb_store.MarkerDetection(
            data=data,
            session=session,
            marker_type=marker_type,
            detection_params=param_dict,
            epoch_params=None,
            epoch_bytype='all',
            tag=db_tag,
            dbi=dbi,
            _id=dbi.get_uuid())
        fname = mk_doc.fname
    # Detect failure
    if len(data['norm_pos']) == 0:
        fn_name = fn.split('.')[-1]
        ff, _ = os.path.splitext(fname)
        fname = f'{ff}_fail.txt'
        with open(os.path.join(sdir, fname), mode='w') as fid:
            fid.write("No markers detected")
        fpath = os.path.join(sdir, fname)
        return fpath
    else:
        if db_name is None:
            fpath = os.path.join(sdir, fname)
            # Alternatively, rely on pydra to save... for now, that would seem
            # to complicate things.
            np.savez(fpath)
        else:
            mk_doc.save()
            fpath = mk_doc.fpath
    return fpath


@pydra.mark.task
@pydra.mark.annotate({'return': {'marker_locations': typing.Dict}})
def marker_filtering_step(marker_fname,
                          session_folder,
                          fn,
                          param_tag,
                          db_tag=None,
                          marker_type=None,
                          db_name=None):
    """"""
    if is_notebook:
        progress_bar = tqdm.notebook.tqdm
    else:
        progress_bar = tqdm.tqdm
    # Get default database if not provided
    if db_name is not None:
        # This relies on proper configuration of vedb_store, to
        # know what hostname it should look for
        dbi = vedb_store.docdb.getclient(dbname=db_name)
        # Find session to process
        session = dbi.query(1, type='Session', folder=session_folder)
        # Find input arguments for processing
        orig_detections = dbi.query(1, type='MarkerDetection', 
                                    fname=marker_fname)
        detection_param_dict = orig_detections.detection_params
        epoch_param_dict = dbi.query(1, type='ParamDictionary', 
                                    fn=fn, tag=param_tag)
        # Test for completed analysis stored in database
        marker_docs = dbi.query(
            type='MarkerLocation',
            session=session._id,
            detection_params=detection_param_dict._id,
            epoch_params=epoch_param_dict._id,
        )
        if len(marker_docs) > 0:
            # Has been run, need to sort by epochs and return something
            marker_docs = sorted(marker_docs, key=lambda x: x.epoch_bytype)
            return [md.fpath for md in marker_docs]

        marker_data = orig_detections.data
        # Load raw time
        all_timestamps = np.load(session.paths['world_camera'][0])
        kwargs = epoch_param_dict.params

    else:
        # Get marker file
        marker_fpath = os.path.join(
            PYDRA_OUTPUT_DIR, session_folder, marker_fname)
        marker_data = dict(np.load(marker_fpath))
        # TODO: remove 'staging' here and leave it at BASE_PATH for non-db calls
        time_fpath = os.path.join(
            BASE_DIR, 'staging', session_folder, 'world_timestamps.npy')
        all_timestamps = np.load(time_fpath)
        # Load parameters from stored yaml files
        fn_name = fn.split('.')[-1]
        param_fname = f'{fn_name}-{param_tag}.yaml'
        param_fpath = os.path.join(PARAM_DIR, param_fname)
        kwargs = utils.read_yaml(param_fpath)

    # convert `fn` string to callable function
    func = get_function(fn)
    # Optionally add progress bar, if supported
    default_kw = get_default_kwargs(func)
    if 'progress_bar' in default_kw:
        kwargs['progress_bar'] = progress_bar
    # Inelegant but functional to use bespoke inputs here;
    # this won't generalize well if we replace this step
    # with another function, but that seems unlikely
    data = func(marker_data, all_timestamps, **kwargs,)

    # Manage ouptut file
    session_date = os.path.split(session_folder)[-1]
    if db_name is None:
        sdir = os.path.join(PYDRA_OUTPUT_DIR, session_date)
        fnames = [
            f'marker_detection_{fn_name}_{param_tag}_{e}.npz' for e in range(len(data))]
    else:
        sdir = os.path.join(BASE_OUTPUT_DIR, session_date)
        fnames = []
        for ie, epoch_data in enumerate(data):
            mk_doc = vedb_store.MarkerDetection(
                data=epoch_data,
                session=session,
                marker_type=marker_type,
                detection_params=detection_param_dict,
                epoch_params=epoch_param_dict,
                epoch_bytype=ie,
                tag=db_tag,
                dbi=dbi,
                _id=dbi.get_uuid())
            fnames.append(mk_doc.fname)
    # Detect failure
    if len(data) == 0:
        fn_name = fn.split('.')[-1]
        ff, _ = os.path.splitext(fname)
        fname = f'{ff}_fail.txt'
        with open(os.path.join(sdir, fname), mode='w') as fid:
            fid.write("No markers detected")
        fpath = os.path.join(sdir, fname)
        return fpath
    else:
        if db_name is None:
            fpath = os.path.join(sdir, fname)
            # Alternatively, rely on pydra to save... for now, that would seem
            # to complicate things.
            np.savez(fpath)
        else:
            mk_doc.save()
            fpath = mk_doc.fpath
    return fpath


# Workflows
"""
# Inputs: 
session folder
list of params: [pupil_detection, marker_detection, marker_filtering, calibration, gaze_estimation, ]

"""
