# Full pipelines for gaze estimation
import numpy as np
import tqdm
import tqdm.notebook
import vedb_store
import file_io
import pydra
import typing
import copy
import os

from . import utils
from .options import config
from .calibration import Calibration


BASE_DIR = config.get('paths', 'base_dir')
PYDRA_OUTPUT_DIR = config.get('paths', 'pydra_cache') # '/pomcloud0/vedb/pydra_cache/'
BASE_OUTPUT_DIR = config.get('paths', 'output_dir') # '/pomcloud0/vedb/processed/'
PARAM_DIR = os.path.join(os.path.split(__file__)[0], 'config/')

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
def pupil_detection(session_folder, 
        fn, 
        param_tag, 
        eye='left', 
        db_tag=None, 
        db_name=None,
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

    # Attempt loading from database if provided
    if db_name is not None:
        # This relies on proper configuration of vedb_store, to
        # know what hostname it should look for
        dbi = vedb_store.docdb.getclient(dbname=db_name, is_verbose=False)
        # Find session to process
        session = dbi.query(1, type='Session', folder=session_folder)
        dbi.is_verbose = is_verbose > 1
        # Find input arguments for processing
        try:
            param_dict = dbi.query(1, type='ParamDictionary', fn=fn, tag=param_tag)
        except:
            print(f"Unknown parameter dictionary!\nfn={fn}\ntag={param_tag}")
            raise
        # Test for completed analysis stored in database
        try:
            if is_verbose:
                print("> Searching for %s pupil locations..."%eye)
            pupil_data = dbi.query(1, type='PupilDetection',
                                   tag=db_tag,
                                   eye=eye,
                                   session=session._id,
                                   params=param_dict._id)
            if is_verbose:
                print("FOUND DETECTED PUPILS.")
            return pupil_data.fname
        except:
            # Not yet run
            pass
        # Prepare inputs
        eye_time_file, eye_video_file = session.paths["eye_%s" % eye]
        kwargs = param_dict.params

    else:
        eye_dict = dict(left=1, right=0)
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
    func = utils.get_function(fn)
    # Optionally add progress bar, if supported
    default_kw = get_default_kwargs(func)
    if 'progress_bar' in default_kw:
        kwargs['progress_bar'] = progress_bar
    # Call function
    data = func(eye_video_file, eye_time_file, **kwargs,)
    #move_intermediate_files = 'intermediate_files' in data
    # Manage ouptut file
    session_date = os.path.split(session_folder)[-1]
    # Detect failure
    failed = len(data['norm_pos']) == 0
    if db_name is None:
        sdir = os.path.join(PYDRA_OUTPUT_DIR, session_date)
        fname = f'pupil_detection_{eye}_{fn_name}_{param_tag}.npz'
    else:
        sdir = os.path.join(BASE_OUTPUT_DIR, session_date)
        pup_doc = vedb_store.PupilDetection(
            data=data,
            session=session,
            eye=eye,
            failed=failed,
            params=param_dict,
            tag=param_dict.tag,
            dbi=dbi,
            _id=dbi.get_uuid())
        fname = pup_doc.fname
        #if move_intermediate_files:
        #    os.rename(data['intermediate_files'], db_location)
    if db_name is None:
        fpath = os.path.join(sdir, fname)
        # Alternatively, rely on pydra to save... for now, that would seem
        # to complicate things.
        np.savez(fpath)
    else:
        pup_doc.save()
        fname = pup_doc.fname
    return fname


@pydra.mark.task
@pydra.mark.annotate({'return': {'marker_locations': typing.Any}})
def marker_detection(session_folder,
                     fn,
                     param_tag,
                     db_tag=None,
                     marker_type=None,
                     db_name=None,
                     is_verbose=False):
    """Run a pupil detection function as a pipeline step
    
    Parameters
    ----------
    session folder : str
        folder for eye video session to run

    """
    if is_verbose:
        print(f"\n=== Finding all {marker_type} marker locations ({param_tag}) ===\n")
    if is_notebook():
        progress_bar = tqdm.notebook.tqdm
    else:
        progress_bar = tqdm.tqdm
    # Get default database if not provided
    if db_name is not None:
        # This relies on proper configuration of vedb_store, to
        # know what hostname it should look for
        dbi = vedb_store.docdb.getclient(dbname=db_name, is_verbose=False)
        # Find session to process
        session = dbi.query(1, type='Session', folder=session_folder)
        dbi.is_verbose = is_verbose > 1
        # Find input arguments for processing
        if is_verbose:
            print("> Searching for marker detection params...")
        param_dict = dbi.query(1, type='ParamDictionary', fn=fn, tag=param_tag)
        # Test for completed analysis stored in database
        try:
            if is_verbose:
                print("> Searching for detected markers...")
            marker_data = dbi.query(
                1, type='MarkerDetection',
                session=session._id,
                detection_params=param_dict._id,
                tag=db_tag,
                epoch='all')
            if is_verbose:
                print("FOUND DETECTED MARKERS.")
            return marker_data.fname
        except:
            # Not yet run
            pass
        # Prepare inputs
        world_time_file, world_video_file = session.paths["world_camera"]
        # Optional, potentially present manual labels for marker times
        marker_time_file = os.path.join(session.path, 'marker_times.yaml')
        kwargs = param_dict.params

    else:
        world_time_file = os.path.join(session_folder, 'world_timestamps.npy')
        world_video_file = os.path.join(session_folder, 'world.mp4')
        marker_time_file = os.path.join(session_folder, 'marker_times.yaml')
        # Load parameters from stored yaml files
        fn_name = fn.split('.')[-1]
        param_fname = f'{fn_name}-{param_tag}.yaml'
        param_fpath = os.path.join(PARAM_DIR, param_fname)
        kwargs = utils.read_yaml(param_fpath)
    # convert `fn` string to callable function
    func = utils.get_function(fn)
    # Optionally add progress bar, if supported
    default_kw = get_default_kwargs(func)
    if 'progress_bar' in default_kw:
        kwargs['progress_bar'] = progress_bar
    # Optionally (if present), use manual labels for marker times
    if os.path.exists(marker_time_file):
        marker_times = utils.read_yaml(marker_time_file)
        if marker_type == 'concentric_circle':
            frames = marker_times['calibration_frames']
        elif marker_type == 'checkerboard':
            frames = marker_times['validation_frames']
        # Loop over list
        data = []
        print('Using manually labeled times.')
        for j, (start_frame, end_frame) in enumerate(frames):
            print('Searching epoch %d / %d'%(j + 1, len(frames)))
            tmp = func(world_video_file, world_time_file, 
                start_frame=start_frame,
                end_frame=end_frame,
                **kwargs,)
            data.append(tmp)
        # Concatenate results
        data = utils.stack_arraydicts(*data)
    else:
        data = func(world_video_file, world_time_file, **kwargs,)
    failed = len(data['norm_pos']) == 0
    # Manage ouptut file
    if db_name is None:
        session_date = os.path.split(session_folder)[-1]
        sdir = os.path.join(PYDRA_OUTPUT_DIR, session_date)
        fname = f'marker_detection_{fn_name}_{param_tag}.npz'
        fpath = os.path.join(sdir, fname)
        # Alternatively, rely on pydra to save... for now, that would seem
        # to complicate things.
        np.savez(fpath)
    else:
        mk_doc = vedb_store.MarkerDetection(
            data=data,
            session=session,
            marker_type=marker_type,
            detection_params=param_dict,
            failed=failed,
            epoch_params=None,
            epoch='all',
            tag=db_tag,
            dbi=dbi,
            _id=dbi.get_uuid())
        mk_doc.save()
        fname = mk_doc.fname
    return fname


@pydra.mark.task
@pydra.mark.annotate({'return': {'marker_locations': typing.List}})
def marker_filtering(marker_fname,
                          session_folder,
                          fn,
                          param_tag,
                          db_tag=None,
                          marker_type=None,
                          db_name=None,
                          is_verbose=False):
    """"""
    if is_verbose:
        print(f"\n=== Finding {marker_type} marker epochs ({param_tag}) ===\n")
    if is_notebook:
        progress_bar = tqdm.notebook.tqdm
    else:
        progress_bar = tqdm.tqdm
    # Get default database if not provided
    if db_name is not None:
        # This relies on proper configuration of vedb_store, to
        # know what hostname it should look for
        dbi = vedb_store.docdb.getclient(dbname=db_name, is_verbose=False)
        dbi.is_verbose = is_verbose > 1
        # Find components
        try:
            # Find session to process
            session = dbi.query(1, type='Session', folder=session_folder)
            session.db_load()
            # Find input arguments for processing
            if is_verbose:
                print("> Searching for markers to filter...")
            orig_detections = dbi.query(1, type='MarkerDetection', 
                                        fname=marker_fname)
            orig_detections.db_load()
            detection_param_dict = orig_detections.detection_params
            marker_data = orig_detections.data
            if is_verbose:
                print("> Searching for filtering parameters...")
            epoch_param_dict = dbi.query(1, type='ParamDictionary', 
                                        fn=fn, tag=param_tag)
            epoch_param_dict.db_load()
            # Load raw time
            all_timestamps = np.load(session.paths['world_camera'][0])
            kwargs = epoch_param_dict.params
        except Exception as err:
            raise
            #print(err)
            #raise ValueError('At least one input does not exist!')
        # Test for completed analysis stored in database
        if is_verbose:
            print('> Searching for finished filtering...')
        marker_docs = dbi.query(
            type='MarkerDetection',
            session=session._id,
            detection_params=detection_param_dict._id,
            epoch_params=epoch_param_dict._id,
        )
        if len(marker_docs) > 0:
            if is_verbose:
                print("FOUND FILTERED MARKERS.")
            # Has been run, need to sort by epochs and return something
            marker_docs = sorted(marker_docs, key=lambda x: x.epoch)
            return [md.fname for md in marker_docs]
        else:
            if is_verbose:
                print("NOT FOUND RUNNING ANEW...")
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
    func = utils.get_function(fn)
    # Optionally add progress bar, if supported
    default_kw = get_default_kwargs(func)
    if 'progress_bar' in default_kw:
        kwargs['progress_bar'] = progress_bar
    # Inelegant but functional to use bespoke inputs here;
    # this won't generalize well if we replace this step
    # with another function, but that seems unlikely
    data = func(marker_data, all_timestamps, **kwargs,)
    # Detect failure
    failed = len(data) == 0
    # Manage ouptut file
    session_date = os.path.split(session_folder)[-1]
    if db_name is None:
        sdir = os.path.join(PYDRA_OUTPUT_DIR, session_date)
        fnames = [
            f'marker_detection_{fn_name}_{db_tag}_{e}.npz' for e in range(len(data))]
        for ff, dd in zip(fnames, data):
            np.savez(os.path.join(sdir, ff), **dd)
    else:
        fnames = []
        if failed:
            # Save mk_doc
            if marker_type=='checkerboard':
                epoch_data = dict(
                    timestamp=np.array([]),
                    location_full_checkerboard=np.array([]),
                    norm_pos_full_checkerboard=np.array([]),
                    location=np.array([]),
                    norm_pos=np.array([]),)
            elif marker_type=='concentric_circle':
                epoch_data = dict(
                    location=np.array([]),
                    norm_pos=np.array([]),
                    timestamp=np.array([]),
                    size=np.array([]),)
            mk_doc = vedb_store.MarkerDetection(
                data=epoch_data,
                session=session,
                marker_type=marker_type,
                detection_params=detection_param_dict,
                epoch_params=epoch_param_dict,
                epoch=0,
                tag=db_tag,
                failed=failed,
                dbi=dbi,
                _id=dbi.get_uuid())
            mk_doc.save()
            fnames = [mk_doc.fname]
        else:
            for ie, epoch_data in enumerate(data):
                mk_doc = vedb_store.MarkerDetection(
                    data=epoch_data,
                    session=session,
                    marker_type=marker_type,
                    detection_params=detection_param_dict,
                    epoch_params=epoch_param_dict,
                    epoch=ie,
                    tag=db_tag,
                    failed=failed,
                    dbi=dbi,
                    _id=dbi.get_uuid())
                mk_doc.save()
                fnames.append(mk_doc.fname)
    return fnames


@pydra.mark.task
@pydra.mark.annotate({'return': {'calibration_file': typing.Any}})
def compute_calibration(marker_fname,
                        pupil_fnames,
                        session_folder,
                        calibration_class,
                        param_tag,
                        eye=None,
                        db_tag=None,
                        db_name=None,
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
    if not isinstance(pupil_fnames, (list, tuple)):
        pupil_fnames = [pupil_fnames]

    # Get default database if not provided
    if db_name is not None:
        # This relies on proper configuration of vedb_store, to
        # know what hostname it should look for
        dbi = vedb_store.docdb.getclient(dbname=db_name, is_verbose=False)
        # Find session to process
        session = dbi.query(1, type='Session', folder=session_folder)
        dbi.is_verbose = is_verbose > 1
        session.db_load()
        # Find input arguments for processing
        if is_verbose:
            print("> Searching for marker epochs...")
        markers = dbi.query(1, type='MarkerDetection',
                                    fname=marker_fname)
        markers.db_load()
        if is_verbose:
            print("> Searching for pupil locations...")
        pupil_docs = [dbi.query(1, type='PupilDetection', fname=pf) for pf in pupil_fnames]
        for pup in pupil_docs:
            pup.db_load()
        # Get parameters
        if is_verbose:
            print("> Searching for calibration parameters...")
        calibration_params = dbi.query(1, type='ParamDictionary', tag=param_tag)
        # Test for completed analysis stored in database
        try:
            if is_verbose:
                print("> Searching for computed calibration...")
            calib_doc = dbi.query(1,
                type='Calibration',
                calibration_class=calibration_class,
                session=session._id,
                pupil_detection=[pup._id for pup in pupil_docs],
                marker_detection=markers._id,
                params=calibration_params._id,
                eye=eye,
            )
            print("FOUND CALIBRATION.")
            return calib_doc.fname
        except:
            marker_data = markers.data
            pupil_data = [p.data for p in pupil_docs]
            kwargs = calibration_params.params
            vdims = session.recording_system.world_camera.resolution
    else:
        # Get marker file
        if session_folder.endswith(os.sep):
            session_folder = session_folder[:-1]
        fbase, folder = os.path.split(session_folder)
        marker_fpath = os.path.join(
            PYDRA_OUTPUT_DIR, folder, marker_fname)
        marker_data = dict(np.load(marker_fpath))
        pupil_fpaths = [os.path.join(
            PYDRA_OUTPUT_DIR, folder, pupil_fname
            ) for pupil_fname in pupil_fnames]
        pupil_data = [dict(np.load(fp)) for fp in pupil_fpaths]
        # Load parameters from stored yaml files
        fn_name = calibration_class.split('.')[-1]
        param_fname = f'{fn_name}-{param_tag}.yaml'
        param_fpath = os.path.join(PARAM_DIR, param_fname)
        kwargs = utils.read_yaml(param_fpath)
        vdims = file_io.var_size(os.path.join(
            session_folder, 'world.mp4'))[2:0:-1]
    if len(pupil_data) == 1:
        # only one eye
        pupil_data = pupil_data[0]
    # convert `calibration_class` string to callable class
    cal_class = utils.get_function(calibration_class)
    print("Computing calibration...")
    cal = cal_class(pupil_data, marker_data, vdims, **kwargs,)

    # Detect failure?
    failed = False # len(data) == 0
    # Manage ouptut file
    if db_name is None:
        # won't get here, because I don't know what to do about paths...
        sdir = os.path.join(PYDRA_OUTPUT_DIR, folder)
        ctype = kwargs['calibration_type']
        fname = f'calibration_{ctype}_{db_tag}.npz'
        cal.save(os.path.join(sdir, fname))
    else:
        if failed:
            # Save mk_doc
            cal_doc = []
            cal_doc.save()
            fname = [cal_doc.fname]
        else:
            cal_doc = vedb_store.Calibration(
                calibration_class = calibration_class,
                session=session,
                pupil_detection=pupil_docs,
                marker_detection=markers,
                eye=eye,
                epoch=markers.epoch,
                failed=failed,
                params=calibration_params,
                data=cal.calibration_data,
                tag=db_tag,
                dbi=dbi,
                _id=dbi.get_uuid())
            cal_doc.save()
    return cal_doc.fname


@pydra.mark.task
@pydra.mark.annotate({'return': {'gaze_locations': typing.Any}})
def map_gaze(session_folder,
                    fn,
                    pupil_fnames,
                    calibration_fname,
                    calibration_epoch,
                    param_tag,
                    eye=None,
                    db_tag=None,
                    db_name=None,
                    is_verbose=False):
    if is_verbose:
        print("\n=== Computing gaze ===\n")
    # Handle inputs
    if not isinstance(pupil_fnames, (list, tuple)):
        pupil_fnames = [pupil_fnames]

    # Get default database if not provided
    if db_name is not None:
        # This relies on proper configuration of vedb_store, to
        # know what hostname it should look for
        dbi = vedb_store.docdb.getclient(dbname=db_name, is_verbose=False)
        # Find session to process
        session = dbi.query(1, type='Session', folder=session_folder)
        dbi.is_verbose = is_verbose > 1
        # Find input arguments for processing
        if is_verbose:
            print("> Searching for pupil locations...")
        pupil_docs = [dbi.query(1, type='PupilDetection', fname=pf)
                  for pf in pupil_fnames]
        for pup in pupil_docs:
            pup.db_load()
        # Get Calibration & mapping parameters
        if is_verbose:
            print("> Searching for computed calibration...")
        calib_doc = dbi.query(1, type='Calibration', fname=calibration_fname)
        if is_verbose:
            print("> Searching for gaze mapping parameters...")
        mapping_params = dbi.query(1, type='ParamDictionary', tag=param_tag)
        # Search for computed gaze
        if is_verbose:
            print("> Searching for computed gaze...")
        try:
            gaze_doc = dbi.query(1, 
                type='Gaze', 
                pupil_detection=[p._id for p in pupil_docs],
                calibration=calib_doc._id,
                calibration_epoch=calibration_epoch,
                eye=eye,
                params=mapping_params._id,
                tag=db_tag)
            print("FOUND GAZE.")
            return gaze_doc.fname
        except:
            session.db_load()
            # vdims? (video dimensions for gaze mapping?)
            pupil_data = [p.data for p in pupil_docs]
            calib_doc.load()
            calibration = calib_doc.calibration
            kwargs = mapping_params.params

    else:
        # Get marker file
        if session_folder.endswith(os.sep):
            session_folder = session_folder[:-1]
        _, folder = os.path.split(session_folder)
        pupil_fpaths = [os.path.join(
            PYDRA_OUTPUT_DIR, folder, pupil_fname
             ) for pupil_fname in pupil_fnames]
        pupil_data = [dict(np.load(fp)) for fp in pupil_fpaths]
        calibration = Calibration.load(calibration_fname)
        # Load parameters from stored yaml files
        param_fname = f'gaze_mapper-{param_tag}.yaml'
        param_fpath = os.path.join(PARAM_DIR, param_fname)
        kwargs = utils.read_yaml(param_fpath)
        #vdims = file_io.var_size(os.path.join(
        #    session_folder, 'world.mp4'))[2:0:-1]
    
    if len(pupil_data) == 1:
        # only one eye
        pupil_data = pupil_data[0]
    
    func = utils.get_function(fn)
    print("Computing gaze...")
    gaze = func(calibration, pupil_data, **kwargs)

    # Detect failure?
    failed = False  # len(data) == 0
    # Manage ouptut file
    if db_name is None:
        # FIX ME.
        # won't get here, because I don't know what to do about paths...
        sdir = os.path.join(PYDRA_OUTPUT_DIR, folder)
        ctype = kwargs['calibration_type']
        fname = f'gaze_{ctype}_{db_tag}.npz'
        np.save(gaze, os.path.join(sdir, fname))
    else:
        if failed:
            gaze = {}
        gaze_doc = vedb_store.Gaze(
            data=gaze,
            session=session,
            pupil_detection=pupil_docs,
            calibration=calib_doc,
            calibration_epoch=calibration_epoch,
            eye=eye,
            failed=failed,
            params=mapping_params,
            tag=db_tag,
            dbi=dbi,
            _id=dbi.get_uuid())
        gaze_doc.save()
    return gaze_doc.fname


@pydra.mark.task
@pydra.mark.annotate({'return': {'error': typing.Any}})
def compute_error(session_folder,
             fn,
             gaze_fname,
             marker_fname,
             param_tag,
             eye=None,
             epoch=None,
             db_tag=None,
             db_name=None,
             is_verbose=False):
    if is_verbose:
        print("\n=== Computing error ===\n")
    # Get default database if not provided
    if db_name is not None:
        # This relies on proper configuration of vedb_store, to
        # know what hostname it should look for
        dbi = vedb_store.docdb.getclient(dbname=db_name, is_verbose=False)
        # Find session to process
        session = dbi.query(1, type='Session', folder=session_folder)
        dbi.is_verbose = is_verbose > 1
        # Find input arguments for processing
        if is_verbose:
            print("> Searching for gaze estimate...")
        gaze_doc = dbi.query(1, type='Gaze', fname=gaze_fname)
        # Get Calibration & mapping parameters
        if is_verbose:
            print("> Searching for detected (validation) markers...")
        marker_doc = dbi.query(1, type='MarkerDetection', fname=marker_fname)
        if is_verbose:
            print("> Searching for error computation parameters...")
        error_params = dbi.query(1, type='ParamDictionary', tag=param_tag)
        # Search for computed gaze
        if is_verbose:
            print("> Searching for computed error...")
        try:
            error_doc = dbi.query(1,
                                 type='GazeError',
                                 gaze=gaze_doc._id,
                                 marker_detection=marker_doc._id,
                                 eye=eye,
                                 #epoch=epoch, marker_doc will specify epoch...
                                 params=error_params._id,
                                 tag=db_tag)
            print("FOUND ERROR ESTIMATE.")
            return error_doc.fname
        except:
            session.db_load()
            # vdims? (video dimensions for gaze mapping?)
            marker_data = marker_doc.data
            gaze_data = gaze_doc.data
            kwargs = error_params.params

    else:
        # Get marker file
        if session_folder.endswith(os.sep):
            session_folder = session_folder[:-1]
        _, folder = os.path.split(session_folder)
        gaze_fpath = os.path.join(PYDRA_OUTPUT_DIR, folder, gaze_fname)
        gaze_data = dict(np.load(gaze_fpath))
        marker_fpath = os.path.join(PYDRA_OUTPUT_DIR, folder, marker_fname)
        marker_data = dict(np.load(marker_fpath))
        # Load parameters from stored yaml files
        param_fname = f'compute_error-{param_tag}.yaml'
        param_fpath = os.path.join(PARAM_DIR, param_fname)
        kwargs = utils.read_yaml(param_fpath)
        #vdims = file_io.var_size(os.path.join(
        #    session_folder, 'world.mp4'))[2:0:-1]

    func = utils.get_function(fn)
    print("Computing error...")
    error = func(marker_data, gaze_data, **kwargs)

    # Detect failure?
    failed = False  # len(data) == 0
    # Manage ouptut file
    if db_name is None:        
        sdir = os.path.join(PYDRA_OUTPUT_DIR, folder)
        fname = f'error_{db_tag}_epoch{epoch}.npz'
        np.save(error, os.path.join(sdir, fname))
    else:
        if failed:
            error = {}
        error_doc = vedb_store.GazeError(
                data=error,
                session=session,
                gaze=gaze_doc,
                marker_detection=marker_doc,
                eye=eye,
                epoch=marker_doc.epoch,
                failed=failed,
                params=error_params,
                tag=db_tag,
                dbi=dbi,
                _id=dbi.get_uuid()
                )
        error_doc.save()
    return error_doc.fname


# correct pupil slippage


### --- Workflows --- ###
def make_pipeline(session,
                  pupil_param_tag='plab_default',
                  pupil_drift_param_tag=None,
                  cal_marker_param_tag='circles_halfres',
                  val_marker_param_tag='checkerboard_halfres',
                  cal_marker_filter_param_tag='cluster_default',
                  val_marker_filter_param_tag='basic_split',
                  calib_param_tag='monocular_pl_default',
                  mapping_param_tag='default_mapper',
                  error_param_tag='smooth_tps_default',
                  calibration_epoch=0,
                  db_name='vedb_internal',
                  is_verbose=False,
                  ):
    """Build a gaze pipeline based on tags for each step"""
    dbi = vedb_store.docdb.getclient(dbname=db_name, is_verbose=False)
    # Initial query for pipeline results:
    pl_elements = utils.load_pipeline_elements(session,
        pupil_param_tag=pupil_param_tag,
        pupil_drift_param_tag=pupil_drift_param_tag,
        cal_marker_param_tag=cal_marker_param_tag,
        val_marker_param_tag=val_marker_param_tag,
        cal_marker_filter_param_tag=cal_marker_filter_param_tag,
        val_marker_filter_param_tag=val_marker_filter_param_tag,
        calib_param_tag=calib_param_tag,
        mapping_param_tag=mapping_param_tag,
        error_param_tag=error_param_tag,
        calibration_epoch=calibration_epoch,
        dbi=dbi,
        is_verbose=False,
        )
    # Get param dicts
    # Abusing database interface for speed
    dbi.is_verbose = False
    pds_all = dbi.query(type='ParamDictionary')
    dbi.is_verbose = True
    if pupil_param_tag is not None:
        pupil_params = [x for x in pds_all if x.tag==pupil_param_tag][0]
    if cal_marker_param_tag is not None:
        cal_marker_params = [x for x in pds_all if x.tag == cal_marker_param_tag][0]
    if val_marker_param_tag is not None:
        val_marker_params = [x for x in pds_all if x.tag==val_marker_param_tag][0]
    if cal_marker_filter_param_tag is not None:
        cal_marker_filter_params = [x for x in pds_all if x.tag==cal_marker_filter_param_tag][0]
    if val_marker_filter_param_tag is not None:
        val_marker_filter_params = [x for x in pds_all if x.tag==val_marker_filter_param_tag][0]
    if calib_param_tag is not None:
        calib_params = [x for x in pds_all if x.tag==calib_param_tag][0]
    if mapping_param_tag is not None:
        mapping_params = [x for x in pds_all if x.tag==mapping_param_tag][0]
    if error_param_tag is not None:
        error_params = [x for x in pds_all if x.tag==error_param_tag][0]
    # Create workflow
    gaze_pipeline = pydra.Workflow(name='gaze_default', 
                                   input_spec=['folder', 'db_name'],
                                   folder=session.folder,
                                   db_name=db_name,
                                   )
    # Pupil detection
    if pupil_param_tag is None:
        pupil_out = []
    else:
        if 'pupil' in pl_elements and (('left' in pl_elements['pupil']) and ('right' in pl_elements['pupil'])):
            pl_path = pl_elements['pupil']['left'].fname
            pr_path = pl_elements['pupil']['right'].fname
            # TEMP: Suppress output for completed steps.
            # May want to reconcile this later somehow. Or not.
            pupil_out = []
                #("pupil_left",
                # pl_path),
                #("pupil_right",
                # pr_path),
            #]
            pupil_inpt = [pl_path, pr_path]
        else:
            gaze_pipeline.add(pupil_detection(
                name='pupil_left',
                session_folder=session.folder,  # gaze_pipeline.lzin.folder,
                fn=pupil_params.fn,
                param_tag=pupil_param_tag,
                eye='left',
                db_tag=pupil_param_tag,
                db_name=db_name,  # gaze_pipeline.lzin.db_name,
                is_verbose=is_verbose,
                )
            )
            gaze_pipeline.add(pupil_detection(
                name='pupil_right',
                session_folder=session.folder,  # gaze_pipeline.lzin.folder,
                fn=pupil_params.fn,
                param_tag=pupil_param_tag,
                eye='right',
                db_tag=pupil_param_tag,
                db_name=db_name,  # gaze_pipeline.lzin.db_name,
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
    # Calibration marker detection
    if cal_marker_param_tag is not None:
        if 'calibration_marker_all' in pl_elements:
            # TEMP: Suppress output for completed steps.
            # May want to reconcile this later somehow. Or not.
            calibration_marker_out = [] #("calibration_marker",
                                        #pl_elements['calibration_marker_all'].fname), ]
            calibration_marker_inpt = pl_elements['calibration_marker_all'].fname
        else:
            gaze_pipeline.add(marker_detection(
                name='calibration_detection',
                session_folder=session.folder,  # gaze_pipeline.lzin.folder,
                fn=cal_marker_params.fn,
                param_tag=cal_marker_param_tag,
                db_tag=cal_marker_param_tag,
                marker_type='concentric_circle',
                db_name=db_name,  # gaze_pipeline.lzin.db_name,
                is_verbose=is_verbose,)
            )
            calibration_marker_out = [("calibration_marker",
                                    gaze_pipeline.calibration_detection.lzout.marker_locations), ]
            calibration_marker_inpt = gaze_pipeline.calibration_detection.lzout.marker_locations
    else:
        calibration_marker_out = []
    
    # Filtering out spurious calibration marker detections
    if cal_marker_filter_param_tag is None:
        cal_filter_out = []
    else:
        if 'calibration_marker_filtered' in pl_elements:
            # Make a list long enough to contain an index for the calibration marker epoch in question
            # Note: this is shady, sorry future me for bugs this will prob generate
            cal_epoch_fnames = [''] * (calibration_epoch + 1)
            cal_epoch_fnames[calibration_epoch] = pl_elements['calibration_marker_filtered'].fname
            # TEMP: Suppress output for completed steps.
            # May want to reconcile this later somehow. Or not.
            cal_filter_out = []#("calibration_epochs",
                               # cal_epoch_fnames), ]
            calibrations_filtered = cal_epoch_fnames
        else:
            if cal_marker_param_tag is None:
                raise ValueError("Must specify cal_marker_param_tag to conduct calibration marker filtering!")
            gaze_pipeline.add(marker_filtering(
                name='calibration_marker_filtering',
                marker_fname=calibration_marker_inpt,
                session_folder=session.folder,  # gaze_pipeline.lzin.folder,
                fn=cal_marker_filter_params.fn,
                param_tag=cal_marker_filter_param_tag,
                db_tag='-'.join([cal_marker_param_tag, cal_marker_filter_param_tag]),
                marker_type='concentric_circle',
                db_name=db_name,
                is_verbose=is_verbose,
                )
            )
            cal_filter_out = [("calibration_epochs",
                            gaze_pipeline.calibration_marker_filtering.lzout.marker_locations),]
            calibrations_filtered = gaze_pipeline.calibration_marker_filtering.lzout.marker_locations
    # Computing calibration 
    if calib_param_tag is None:
        calibration_out = []
    else:
        if any([x is None for x in [pupil_param_tag, cal_marker_param_tag, cal_marker_filter_param_tag]]):
                raise ValueError(
                    "Must specify tags for all previous steps to compute calibration!")
        # Choose epoch
        gaze_pipeline.add(select(name='cal_epoch_choice',
                                 x_list=calibrations_filtered,
                                    index=calibration_epoch))
        if 'calibration' in pl_elements:
            if 'binocular' in calib_param_tag:
                calibration_out = []
                    #('calibration_both', pl_elements['calibration']['both'].fname)]
                calibration_inpt = [pl_elements['calibration']['both'].fname]
            elif 'monocular' in calib_param_tag:
                calibration_out = []#('calibration_left', pl_elements['calibration']['left'].fname),
                                    #('calibration_right', pl_elements['calibration']['right'].fname, )]
                calibration_inpt = [pl_elements['calibration']['left'].fname, 
                                    pl_elements['calibration']['right'].fname]
        else:
            if 'binocular' in calib_param_tag:
                gaze_pipeline.add(compute_calibration(
                    name='calibration_both',
                    marker_fname=gaze_pipeline.cal_epoch_choice.lzout.out,
                    pupil_fnames=pupil_inpt,
                    session_folder=session.folder,
                    calibration_class=calib_params.fn,
                    param_tag=calib_param_tag,
                    eye='both',
                    db_tag='-'.join([pupil_param_tag,
                                    cal_marker_param_tag,
                                    cal_marker_filter_param_tag,
                                    calib_param_tag,]),
                    db_name=db_name,
                    is_verbose=is_verbose,
                    )
                )
                calibration_out = [('calibration_both', gaze_pipeline.calibration_both.lzout.calibration_file)]

            elif 'monocular' in calib_param_tag:
                # There may be a more compact way to do this with a pydra splitter, 
                # but I can't see how w/ the extra kwarg for `eye`
                gaze_pipeline.add(compute_calibration(
                    name='calibration_left',
                    marker_fname=gaze_pipeline.cal_epoch_choice.lzout.out,
                    pupil_fnames=pupil_inpt[0],
                    session_folder=session.folder,
                    calibration_class=calib_params.fn,
                    param_tag=calib_param_tag,
                    eye='left',
                    db_tag='-'.join([pupil_param_tag,
                                    cal_marker_param_tag,
                                    cal_marker_filter_param_tag,
                                    calib_param_tag, ]),
                    db_name=db_name,
                    is_verbose=is_verbose,
                )
                )
                gaze_pipeline.add(compute_calibration(
                    name='calibration_right',
                    marker_fname=gaze_pipeline.cal_epoch_choice.lzout.out,
                    pupil_fnames=pupil_inpt[1],
                    session_folder=session.folder,
                    calibration_class=calib_params.fn,
                    param_tag=calib_param_tag,
                    eye='right',
                    db_tag='-'.join([pupil_param_tag,
                                    cal_marker_param_tag,
                                    cal_marker_filter_param_tag,
                                    calib_param_tag, ]),
                    db_name=db_name,
                    is_verbose=is_verbose,
                )
                )
                calibration_out = [('calibration_left',gaze_pipeline.calibration_left.lzout.calibration_file), 
                                   ('calibration_right',gaze_pipeline.calibration_right.lzout.calibration_file, )]
            else:
                raise ValueError(f"known calbration tag {calib_param_tag}")
            calibration_inpt = [x[1] for x in calibration_out]
    # Mapping gaze
    if mapping_param_tag is None:
        gaze_out = []
    else:
        # Check for presence in pl_elements
        gaze_tags = [x for x in [pupil_param_tag,
                                pupil_drift_param_tag,
                                cal_marker_param_tag,
                                cal_marker_filter_param_tag,
                                calib_param_tag,
                                mapping_param_tag] if x is not None]
        gaze_db_tag = '-'.join(gaze_tags)
        if 'monocular' in calib_param_tag:
            eyes = ['left', 'right']
            pupil_fnames =  pupil_inpt
        else:
            eyes = ['both']
            pupil_fnames = [pupil_inpt]
        gaze_out = []
        if 'gaze' in pl_elements:
            gaze_inpt = [pl_elements['gaze'][eye].fname for eye in eyes]
        else:
            for e, eye in enumerate(eyes):
                gaze_name = 'gaze_%s'%eye
                gaze_pipeline.add(map_gaze(
                    name=gaze_name,
                    session_folder=session.folder,
                    fn=mapping_params.fn,
                    pupil_fnames=pupil_fnames[e],
                    calibration_fname=calibration_inpt[e],
                    calibration_epoch=calibration_epoch,
                    param_tag=mapping_param_tag,
                    eye=eye,
                    db_tag=gaze_db_tag,
                    db_name=db_name,
                    is_verbose=is_verbose)
                    )
                gaze_out.append((gaze_name, getattr(gaze_pipeline, gaze_name).lzout.gaze_locations))
            gaze_inpt = [x[1] for x in gaze_out]
    # Detect validation markers
    if val_marker_param_tag is None:
        validation_marker_out = []
    else:
        if 'validation_marker_all' in pl_elements:
            validation_marker_out = []
            val_marker_inpt = pl_elements['validation_marker_all'].fname
        else:
            gaze_pipeline.add(marker_detection(
                name='validation_detection',
                session_folder=session.folder,
                fn=val_marker_params.fn,
                param_tag=val_marker_param_tag,
                db_tag=val_marker_param_tag,
                marker_type='checkerboard',
                db_name=db_name,)
            )
            validation_marker_out = [('validation_marker', 
                gaze_pipeline.validation_detection.lzout.marker_locations),]
            val_marker_inpt = gaze_pipeline.validation_detection.lzout.marker_locations

    # Filter validation markers
    if val_marker_filter_param_tag is None:
        val_filter_out = []
    else:
        if 'validation_marker_filtered' in pl_elements:
            val_filter_out = []
            val_filter_inpt = [x.fname for x in pl_elements['validation_marker_filtered']]
        else:
            if val_marker_param_tag is None:
                raise ValueError(
                    "Must specify val_marker_param_tag to conduct validation marker filtering!")
            gaze_pipeline.add(marker_filtering(
                name='validation_marker_filtering',
                marker_fname=val_marker_inpt,
                session_folder=session.folder,  # gaze_pipeline.lzin.folder,
                fn=val_marker_filter_params.fn,
                param_tag=val_marker_filter_param_tag,
                db_tag='-'.join([val_marker_param_tag,
                                val_marker_filter_param_tag]),
                marker_type='checkerboard',
                db_name=db_name,
                is_verbose=is_verbose,
            )
            )
            val_filter_out = [("validation_epochs",
                                gaze_pipeline.validation_marker_filtering.lzout.marker_locations), ]
            val_filter_inpt = gaze_pipeline.validation_marker_filtering.lzout.marker_locations
    
    error_out = []
    if error_param_tag is not None:
        if any([x is None for x in [pupil_param_tag, 
                                    cal_marker_param_tag, 
                                    cal_marker_filter_param_tag,
                                    val_marker_param_tag,
                                    val_marker_filter_param_tag,
                                    calib_param_tag,
                                    mapping_param_tag,
                                    ]]):
            raise ValueError("To compute gaze error, all other steps must be specified")
        # 'eyes' defined above in calibration computation, and all steps must run
        # to compute error, so it will be defined.
        for e, eye in enumerate(eyes):
            error_name = f'compute_error_{eye}'
            print("Attempting to find: ", gaze_inpt[e])
            gaze_pipeline.add(compute_error(
                name=error_name,
                session_folder=session.folder,  # gaze_pipeline.lzin.folder,
                marker_fname=val_filter_inpt,
                gaze_fname=gaze_inpt[e],
                eye=eye,
                # epoch=epoch, # Tricky to do this right, haven't found a way yet.
                fn=error_params.fn,
                param_tag=error_param_tag,
                db_tag='-'.join([gaze_db_tag, 
                                val_marker_param_tag,
                                val_marker_filter_param_tag, 
                                error_param_tag]),
                db_name=db_name,
                is_verbose=is_verbose,
            ).split('marker_fname')
            )
            error_out.append(
                (error_name, getattr(gaze_pipeline, error_name).lzout.error))
    else:
        print("DAFAK SOMEHOW TAG IS NONE.")
    gaze_pipeline.set_output(pupil_out + calibration_marker_out + cal_filter_out + \
        validation_marker_out + val_filter_out + calibration_out + gaze_out + error_out)

    return gaze_pipeline
"""
# Inputs: 
session folder
list of params: [pupil_detection, marker_detection, marker_filtering, calibration, gaze_estimation, ]

"""
