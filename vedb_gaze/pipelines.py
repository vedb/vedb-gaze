# Calibration script
# import vm_preproc as vmp
# import vedb_store
import numpy as np
import tqdm
import tqdm.notebook
import vedb_store
import file_io
import pydra
import typing
import os

from . import utils
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
def pupil_detection_step(session_folder, 
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
    func = get_function(fn)
    # Optionally add progress bar, if supported
    default_kw = get_default_kwargs(func)
    if 'progress_bar' in default_kw:
        kwargs['progress_bar'] = progress_bar
    # Call function
    data = func(eye_video_file, eye_time_file, **kwargs,)
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
            tag=param_dict.tag + '_%s' % eye,
            dbi=dbi,
            _id=dbi.get_uuid())
        fname = pup_doc.fname
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
def marker_detection_step(session_folder,
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
        print("\n=== Finding all marker locations ===\n")
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
            return marker_data.fname
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
def marker_filtering_step(marker_fname,
                          session_folder,
                          fn,
                          param_tag,
                          db_tag=None,
                          marker_type=None,
                          db_name=None,
                          is_verbose=False):
    """"""
    if is_verbose:
        print("\n=== Finding marker epochs ===\n")
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
            if is_verbose:
                print("> Searching for session...")
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
    func = get_function(fn)
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
        pupils = [dbi.query(1, type='PupilDetection', fname=pf) for pf in pupil_fnames]
        for pup in pupils:
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
                pupil_detection=[pup._id for pup in pupils],
                marker_detection=markers._id,
                params=calibration_params._id,
                eye=eye,
            )
            print("FOUND CALIBRATION.")
            return calib_doc.fname
        except:
            marker_data = markers.data
            pupil_data = [p.data for p in pupils]
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
    cal_class = get_function(calibration_class)
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
                pupil_detection=pupils,
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

# map_gaze

# correct pupil slippage

# Workflows


def make_pipeline(session,
                  pupil_param_tag='plab_default',
                  cal_marker_param_tag='circles_halfres',
                  val_marker_param_tag=None,
                  cal_marker_filter_param_tag='cluster_default',
                  calib_param_tag='monocular_pl_default',
                  mapping_param_tag=None,
                  db_name='vedb_test',
                  is_verbose=False,
                  ):
    """Build a gaze pipeline based on tags for each step"""
    dbi = vedb_store.docdb.getclient(dbname=db_name, is_verbose=False)
    # Get param dicts
    if pupil_param_tag is not None:
        pupil_params = dbi.query(
            1, type='ParamDictionary', tag=pupil_param_tag)
    if cal_marker_param_tag is not None:
        cal_marker_params = dbi.query(
            1, type='ParamDictionary', tag=cal_marker_param_tag)
    if val_marker_param_tag is not None:
        val_marker_params = dbi.query(
            1, type='ParamDictionary', tag=val_marker_param_tag)
    if cal_marker_filter_param_tag is not None:
        cal_marker_filter_params = dbi.query(
            1, type='ParamDictionary', tag=cal_marker_filter_param_tag)
    if calib_param_tag is not None:
        calib_params = dbi.query(
            1, type='ParamDictionary', tag=calib_param_tag)
    if mapping_param_tag is not None:
        mapping_params = dbi.query(
            1, type='ParamDictionary', tag=mapping_param_tag)
    # Create workflow
    gaze_pipeline = pydra.Workflow(name='gaze_default', 
                                   input_spec=['folder', 'db_name'],
                                   folder=session.folder,
                                   db_name=db_name,
                                   )
    if pupil_param_tag is not None:
        # Left and right pupil detection
        gaze_pipeline.add(pupil_detection_step(
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
        gaze_pipeline.add(pupil_detection_step(
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
    if cal_marker_param_tag is not None:
        # Calibration marker detection
        gaze_pipeline.add(marker_detection_step(
            name='calibration_detection',
            session_folder=session.folder,  # gaze_pipeline.lzin.folder,
            fn=cal_marker_params.fn,
            param_tag=cal_marker_param_tag,
            db_tag=cal_marker_param_tag,
            marker_type='concentric_circle',
            db_name=db_name,  # gaze_pipeline.lzin.db_name,
            is_verbose=is_verbose,)
        )
    if cal_marker_filter_param_tag is None:
        cal_filter_out = []
    else:
        if cal_marker_param_tag is None:
            raise ValueError("Must specify cal_marker_param_tag to conduct calibration marker filtering!")
        gaze_pipeline.add(marker_filtering_step(
            name='calibration_marker_filtering',
            marker_fname=gaze_pipeline.calibration_detection.lzout.marker_locations,
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
    if calib_param_tag is None:
        calibration_out = []
    else:
        if any([x is None for x in [pupil_param_tag, cal_marker_param_tag, cal_marker_filter_param_tag]]):
            raise ValueError("Must specify tags for all previous steps to compute calibration!")
        if 'binocular' in calib_param_tag:
            gaze_pipeline.add(compute_calibration(
                name='calibration',
                marker_fname=gaze_pipeline.calibration_marker_filtering.lzout.marker_locations,
                pupil_fnames=[gaze_pipeline.pupil_left.lzout.pupil_locations,
                              gaze_pipeline.pupil_right.lzout.pupil_locations, ],
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
                ).split('marker_fname')
            )
            calibration_out = [('calibration', gaze_pipeline.calibration.lzout.calibration_file)]

        elif 'monocular' in calib_param_tag:
            # There may be a more compact way to do this with a pydra splitter, 
            # but I can't see how w/ the extra kwarg for `eye`
            gaze_pipeline.add(compute_calibration(
                name='calibration_left',
                marker_fname=gaze_pipeline.calibration_marker_filtering.lzout.marker_locations,
                pupil_fnames=gaze_pipeline.pupil_left.lzout.pupil_locations,
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
            ).split('marker_fname')
            )
            gaze_pipeline.add(compute_calibration(
                name='calibration_right',
                marker_fname=gaze_pipeline.calibration_marker_filtering.lzout.marker_locations,
                pupil_fnames=gaze_pipeline.pupil_right.lzout.pupil_locations,
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
            ).split('marker_fname')
            )
            calibration_out = [('calibration_left',gaze_pipeline.calibration_left.lzout.calibration_file), 
                               ('calibration_right',gaze_pipeline.calibration_right.lzout.calibration_file, )]

    if val_marker_param_tag is not None:
        gaze_pipeline.add(marker_detection_step(
            name='validation_detection',
            session_folder=session.folder,
            fn=val_marker_params.fn,
            param_tag=val_marker_param_tag,
            db_tag=val_marker_param_tag,
            marker_type='checkerboard',
            db_name=db_name,)
        )

    gaze_pipeline.set_output([("pupil_left", 
                               gaze_pipeline.pupil_left.lzout.pupil_locations),
                              ("pupil_right",
                               gaze_pipeline.pupil_right.lzout.pupil_locations),
                              ("calibration_marker",
                               gaze_pipeline.calibration_detection.lzout.marker_locations),
                              ] + cal_filter_out + calibration_out)

    return gaze_pipeline
"""
# Inputs: 
session folder
list of params: [pupil_detection, marker_detection, marker_filtering, calibration, gaze_estimation, ]

"""
