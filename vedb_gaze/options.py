"""Configuration options for vedb_gaze.

On import, reads the package's ``defaults.cfg`` and overlays a per-user
``options.cfg`` from the appdirs user config directory (e.g.
``~/.config/vedb-gaze/options.cfg`` on Linux). If the user file does not
exist, it is created as a copy of the defaults, to be edited by the user.

The resulting `configparser.ConfigParser` is exposed as `config`. Sections:

- ``[paths]`` : ``base_dir`` (raw session folders) and ``proc_dir``
  (processed outputs); used by `pipelines` as `BASE_DIR` / `PROC_DIR`.
- ``[defaults]`` : default parameter tags for each pipeline step.
"""
import os
import appdirs
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

cwd = os.path.dirname(__file__)
userdir = appdirs.user_config_dir("vedb-gaze", appauthor="MarkLescroart")
usercfg = os.path.join(userdir, "options.cfg")

config = configparser.ConfigParser()
config.read(os.path.join(cwd, 'defaults.cfg'))

# Update defaults with user-sepecifed values in user config
files_successfully_read = config.read(usercfg)

# If user config doesn't exist, create it
if len(files_successfully_read) == 0:
    os.makedirs(userdir, exist_ok=True)
    with open(usercfg, 'w') as fp:
        config.write(fp)
