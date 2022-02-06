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
