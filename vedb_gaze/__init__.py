"""vedb_gaze: gaze estimation for the Visual Experience Database (VEDB).

Tools to estimate gaze from head-mounted eye tracking recorded during
natural behavior: pupil detection in eye videos, detection of calibration
(concentric circle) and validation (checkerboard) markers in world video,
calibration of pupil-to-world-camera mappings, gaze mapping, and estimation
of gaze error.

Submodules
----------
calibration
    `Calibration` class (thin-plate spline and Pupil Labs polynomial
    calibrations) mapping pupil positions to world-camera coordinates.
error_computation
    Gaze error (in degrees) at validation markers, interpolated over the image.
gaze_mapping
    Thin wrapper applying a calibration to pupil data.
labeling
    Eye-movement labeling (saccades, blinks).
marker_detection
    Detection of concentric-circle and checkerboard markers in world video.
marker_parsing
    Filtering, splitting into epochs, and clustering of marker detections.
options
    User configuration (data paths, default tags).
pipelines
    Pipeline steps with cached .npz outputs, and full workflows.
pupil_detection_pl
    Wrapper around Pupil Labs 2D pupil detector.
utils
    Data-format conversion, time matching, file loading helpers.
visualization
    Plotting helpers.
externals
    Vendored third-party code (Pupil Labs, REMoDNaV).

Main entry points
-----------------
- `pipelines.pipeline_vedb` : full gaze pipeline for a mobile VEDB session.
- `pipelines.pipeline_mri` : full gaze pipeline for fMRI eye tracking data.
- `utils.load_pipeline_elements` : load outputs of a pipeline run.
- `labeling.find_saccades_remodnav` : saccade detection with REMoDNaV.
- `labeling.detect_blinks` : blink detection.
"""
from . import (
    calibration,
    utils,
    error_computation,
    gaze_mapping,
    labeling,
    marker_detection,
    marker_parsing,
    options,
    pipelines,
    pupil_detection_pl,
    visualization,
)
