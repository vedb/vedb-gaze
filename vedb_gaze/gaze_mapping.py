
def gaze_mapper(calibration, pupil_data, mapping_type='default_mapper', **kwargs):
    """Map gaze given a calibration object (or objects) and pupil data

    Currently (2022.03.01) this is a very thin wrapper on the .map() method of
    the calibration object provided as input. This function is here in case we 
    want to do fancier things (e.g. a mapping that combines multiple 
    calibrations) in the future. 

    Parameters
    ----------
    calibration : vedb_gaze.calibration.Calibration or list 
        calibration that has been computed
    pupil_data : _type_
        _description_
    mapping_type : str, optional
        type of mapping from pupil to gaze, by default 'default_mapper', 
        which just implements the `map()` method of the calibration object

    Returns
    -------
    _type_
        _description_
    """
    if mapping_type == 'default_mapper':
        gaze = calibration.map(pupil_data, return_type='arraydict')
        return gaze
    