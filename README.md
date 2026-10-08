# vedb-gaze

Gaze estimation, analysis and visualization for the
[Visual Experience Database](https://github.com/vedb) (VEDB): head-mounted eye
tracking recorded during natural behavior.

vedb-gaze takes raw eye and world (scene) camera videos and produces:

- pupil positions in each eye camera
- calibration and validation marker positions in the world camera
- a pupil-to-gaze calibration per eye (thin-plate spline or polynomial)
- gaze position in normalized world-camera coordinates
- gaze error in degrees of visual angle, estimated at validation markers
- eye-movement events (saccades, fixations, pursuits) and blinks

It is meant to be used from Python sessions; there is no command-line interface.

## Installation

```bash
git clone https://github.com/vedb/vedb-gaze.git
cd vedb-gaze
conda env create -f environment.yaml   # creates the `vedb_gaze` environment
conda activate vedb_gaze               # vedb-gaze itself is installed (editable) by the env file
```

To install into an existing environment instead, install the dependencies
below, then run `pip install -e .`.

### Dependencies

Required:

- numpy, scipy, scikit-learn, pandas, matplotlib, pyyaml, tqdm, appdirs,
  msgpack, opencv (`cv2`)
- [file_io](https://github.com/piecesofmindlab/file_io) (video and array
  loading) and [plot_utils](https://github.com/piecesofmindlab/plot_utils)
  (plotting helpers), from GitHub. file_io also imports six, h5py, pillow,
  imageio and IPython, which its setup.py doesn't install.

Optional, needed for specific steps:

| Package | Needed for |
| --- | --- |
| [thinplate](https://github.com/cheind/py-thin-plate-spline) | thin-plate-spline calibration (`monocular_tps*`) and error estimation; used by the default settings. Installs PyTorch. |
| pylids | pupil and eyelid detection (`pupil-pylids_*` configs, the default) and eyelid-based blink detection |
| pupil-detectors | Pupil Labs pupil detector (`pupil-plab_*` configs) |
| [vedb-store](https://github.com/vedb/vedb-store) | a few visualization functions that index into session videos |

Missing optional packages print a message on import; the steps that need them
fail only when called.

## Configuration

Paths to data are read from a config file. On first import, vedb-gaze copies
the package defaults (`vedb_gaze/defaults.cfg`) to a user config file at
`~/.config/vedb-gaze/options.cfg` (Linux; the location comes from
`appdirs.user_config_dir`). Edit that file to set:

```ini
[paths]
base_dir = /path/to/raw/vedb/sessions        # one folder per session
proc_dir = /path/to/processed/gaze/outputs   # outputs written to proc_dir/<session>/
```

The `[defaults]` section sets the default processing tags used by
`utils.load_pipeline_elements` and `utils.make_file_strings`. Note that
`pipelines.pipeline_vedb` has its own keyword defaults, which currently differ
(e.g. `conf75` vs `conf40` calibration). Pass tags explicitly so that
processing and loading use the same ones.

## How the pipeline works

Each processing step is a function in `vedb_gaze/pipelines.py`:

| Step | Function | Config files |
| --- | --- | --- |
| Pupil detection (per eye) | `pupil_detection` | `config/pupil-<tag>.yaml` |
| Marker detection (world video) | `marker_detection` | `config/marker-<tag>.yaml` |
| Marker clustering into fixation points | `marker_clustering` | `config/marker_parsing-<tag>.yaml` |
| Calibration (pupil → gaze mapping) | `compute_calibration` | `config/calibration-<tag>.yaml` |
| Gaze mapping | `map_gaze` | `config/gaze-<tag>.yaml` |
| Error at validation markers | `compute_error` | `config/error-<tag>.yaml` |
| Pupil drift correction | `detrend_pupil` | **placeholder**, see below |

A step is selected by a *tag*. The tag names a YAML file in `vedb_gaze/config/`.
That file's `fn` key gives the function (or class) to call, as an importable
dotted name (e.g. `vedb_gaze.calibration.Calibration`, `pylids.analyze_video`),
and its other keys are passed as keyword arguments. To try new parameters, add
a new YAML file with a new tag.

The workflow functions chain the steps:

- `pipeline_vedb(session, ...)` processes a VEDB session folder. It needs
  `marker_times.yaml` in the session folder, giving frame ranges for
  calibration and validation epochs:
  ```yaml
  calibration_frames: [[start, end], ...]
  validation_frames: [[start, end], ...]
  ```
  It also expects `eye0.mp4` (right eye), `eye1.mp4` (left eye),
  `worldPrivate.mp4`, and `*_timestamps_0start.npy` timestamp files.
- `pipeline_mri(base_dir, subject_id, task, session, ...)` processes eye
  tracking from fMRI sessions in a BIDS-like folder layout (see its docstring).

### Outputs, caching and failures

Each step saves an `.npz` file in the output folder, named from its tags,
e.g. `pupil_detection-left-<pupil_tag>.npz` or
`gaze-left-<gaze_tag>-<calibration_tag>-<hash>.npz`. Calibration, gaze and
error file names include a short hash of all the upstream tags, so outputs
from different settings never overwrite each other.

- If a step's output file already exists, the step returns its path without
  recomputing, so re-running a pipeline only computes what is missing.
- If a step fails, it writes an empty `<name>.failed` file and returns that
  path. Downstream steps given a failed input skip computation and return
  `<output_dir>/previous_step.failed`. Delete `.failed` files to retry.

### Pupil drift correction (placeholder)

`pipelines.detrend_pupil` and the `pupil_detrend_tag` argument are
placeholders for future code that removes gradual drift in pupil estimates
caused by the eye tracker slipping on the head. The step is off by default
(`pupil_detrend_tag=None`) and is not functional yet; see the
`detrend_pupil` docstring for what remains to be done.

## Usage

Process a session and load the results:

```python
from vedb_gaze import pipelines, utils, options
import pathlib

session = '2021_02_27_10_12_44'
tags = dict(
    pupil_tag='pylids_pytorch_pupils_v1',
    eyelid_tag=None,
    calibration_tag='monocular_tps_cv_cluster_median_conf75_cut3std',
    error_tag='smooth_tps_cv_clust_med_outlier4std_conf75_fov101',
)
files = pipelines.pipeline_vedb(session, **tags)   # dict of output file paths

# Load everything back as dicts of arrays (gaze, pupils, markers, error, ...)
proc_dir = pathlib.Path(options.config.get('paths', 'proc_dir')).expanduser()
el = utils.load_pipeline_elements(
    proc_dir / session,
    pupil=tags['pupil_tag'],
    eyelid=None,
    calibration=tags['calibration_tag'],
    error='smooth_tps_cv_clust_med_outlier4std_conf75',
    fov_str='fov101',
)
gaze_left = el['gaze']['left']   # dict with 'timestamp', 'norm_pos', 'confidence'
```

Label eye movements and blinks:

```python
from vedb_gaze import labeling

# Saccades, fixations, pursuits and post-saccadic oscillations (REMoDNaV)
ev = labeling.find_saccades_remodnav(gaze_left)
ev['saccades_onoff']           # (n, 3): onset, offset, duration (s)
ev['events']['label']          # 'SACC', 'ISAC', 'FIXA', 'PURS', 'HPSO', ...

# Blinks from eyelid distance (needs pylids eyelid keypoints in the pupil file)
blinks = labeling.detect_blinks(el['pupil']['left'])
blinks['blinks_onoff']         # (onset, offset, duration) rows
```

`find_saccades_remodnav` is derived from
[REMoDNaV](https://github.com/psychoinformatics-de/remodnav) (MIT license).
If you use it, please cite:

> Dar, A. H., Wagner, A. S., & Hanke, M. (2021). REMoDNaV: robust eye-movement
> classification for dynamic stimulation. *Behavior Research Methods, 53*(1),
> 399–414. https://doi.org/10.3758/s13428-020-01428-x

`vedb_gaze.visualization` has plotting functions for each stage, including
`plot_session_qc` for a one-page quality-control summary of a session.

## Package layout

| Module | Contents |
| --- | --- |
| `pipelines` | pipeline steps and the `pipeline_vedb` / `pipeline_mri` workflows |
| `pupil_detection_pl` | wrapper for the Pupil Labs pupil detector |
| `marker_detection` | concentric-circle and checkerboard detection in world video |
| `marker_parsing` | splitting marker detections into epochs and clusters |
| `calibration` | `Calibration` class (TPS and polynomial pupil→gaze mappings) |
| `gaze_mapping` | applies a calibration to pupil data |
| `error_computation` | gaze error at validation markers and interpolated over the visual field |
| `labeling` | blink detection, saccade detection, related utilities |
| `utils` | file naming and loading, time matching, resampling, misc helpers |
| `visualization` | plots and animations |
| `odometry` | loading and plotting head-tracking (odometry) data |
| `externals` | vendored third-party code: Pupil Labs (LGPL v3) and REMoDNaV (MIT) |

## License

See [LICENSE](LICENSE). Code under `vedb_gaze/externals/` keeps its original
license; see each file's header.
