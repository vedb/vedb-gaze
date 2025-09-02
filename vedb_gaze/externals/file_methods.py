"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

# TODO simplify pldata writer and remove unnecessary methods and classes

import collections
import logging
import os
import pickle
import traceback as tb
from glob import iglob

import msgpack
import numpy as np


logger = logging.getLogger(__name__)
UnpicklingError = pickle.UnpicklingError

PLData = collections.namedtuple("PLData", ["data", "timestamps", "topics"])


class Persistent_Dict(dict):
    """a dict class that uses pickle to save itself to file"""

    def __init__(self, file_path, *args, **kwargs):
        super(Persistent_Dict, self).__init__(*args, **kwargs)
        self.file_path = os.path.expanduser(file_path)
        try:
            self.update(**load_object(self.file_path, allow_legacy=False))
        except IOError:
            logger.debug(
                "Session settings file '{}' not found. "
                "Will make new one on exit.".format(self.file_path)
            )
        except (KeyError, EOFError):  # KeyError, EOFError
            logger.warning(
                "Session settings file '{}'could not be read. "
                "Will overwrite on exit.".format(self.file_path)
            )
            logger.debug(tb.format_exc())

    def save(self):
        d = {}
        d.update(self)
        save_object(d, self.file_path)

    def close(self):
        self.save()


def _load_object_legacy(file_path):
    file_path = os.path.expanduser(file_path)
    with open(file_path, "rb") as fh:
        data = pickle.load(fh, encoding="bytes")
    return data


def load_object(file_path, allow_legacy=True):
    import gc

    file_path = os.path.expanduser(file_path)
    with open(file_path, "rb") as fh:
        try:
            gc.disable()  # speeds deserialization up.
            data = msgpack.unpack(fh, raw=False)
        except Exception as e:
            if not allow_legacy:
                raise e
            else:
                logger.info(
                    "{} has a deprecated format: "
                    "Will be updated on save".format(file_path)
                )
                data = _load_object_legacy(file_path)
        finally:
            gc.enable()
    return data


def save_object(object_, file_path):
    def ndarrray_to_list(
        o, _warned=[False]
    ):  # Use a mutlable default arg to hold a fn interal temp var.
        if isinstance(o, np.ndarray):
            if not _warned[0]:
                logger.warning(
                    "numpy array will be serialized as list. Invoked at:\n"
                    + "".join(tb.format_stack())
                )
                _warned[0] = True
            return o.tolist()
        return o

    file_path = os.path.expanduser(file_path)
    with open(file_path, "wb") as fh:
        msgpack.pack(object_, fh, use_bin_type=True, default=ndarrray_to_list)


class Incremental_Legacy_Pupil_Data_Loader(object):
    def __init__(self, directory=""):
        self.file_loc = os.path.join(directory, "pupil_data")

    def __enter__(self):
        self.file_handle = open(self.file_loc, "rb")
        self.unpacker = msgpack.Unpacker(
            self.file_handle, raw=False, use_list=False
        )
        self.num_key_value_pairs = self.unpacker.read_map_header()
        self._skipped = True
        return self

    def __exit__(self, *exc):
        self.file_handle.close()

    def topic_values_pairs(self):
        for _ in range(self.num_key_value_pairs):
            yield self.unpacker.unpack(), self._next_values()

    def _next_values(self):
        for _ in range(self.unpacker.read_array_header()):
            yield self.unpacker.unpack()


def load_pldata_file(directory, topic):
    ts_file = os.path.join(directory, topic + "_timestamps.npy")
    msgpack_file = os.path.join(directory, topic + ".pldata")
    try:
        data = collections.deque()
        topics = collections.deque()
        data_ts = np.load(ts_file)
        with open(msgpack_file, "rb") as fh:
            for topic, payload in msgpack.Unpacker(
                fh, raw=False, use_list=False
            ):
                data.append(Serialized_Dict(msgpack_bytes=payload))
                topics.append(topic)
    except IOError:
        data = []
        data_ts = []
        topics = []

    return PLData(data, data_ts, topics)


class PLData_Writer(object):
    """docstring for PLData_Writer"""

    def __init__(self, directory, name):
        super(PLData_Writer, self).__init__()
        self.directory = directory
        self.name = name
        self.ts_queue = collections.deque()
        file_name = name + ".pldata"
        self.file_handle = open(os.path.join(directory, file_name), "wb")

    def append(self, datum):
        datum_serialized = msgpack.packb(datum, use_bin_type=True)
        self.append_serialized(
            datum["timestamp"], datum["topic"], datum_serialized
        )

    def append_serialized(self, timestamp, topic, datum_serialized):
        self.ts_queue.append(timestamp)
        pair = msgpack.packb((topic, datum_serialized), use_bin_type=True)
        self.file_handle.write(pair)

    def extend(self, data):
        for datum in data:
            self.append(datum)

    def close(self):
        self.file_handle.close()
        self.file_handle = None

        ts_file = self.name + "_timestamps.npy"
        ts_path = os.path.join(self.directory, ts_file)
        np.save(ts_path, self.ts_queue)
        self.ts_queue = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def next_export_sub_dir(root_export_dir):
    # match any sub directories or files a three digit pattern
    pattern = os.path.join(root_export_dir, "[0-9][0-9][0-9]")
    existing_subs = sorted(iglob(pattern))
    try:
        latest = os.path.split(existing_subs[-1])[-1]
        next_sub_dir = "{:03d}".format(int(latest) + 1)
    except IndexError:
        next_sub_dir = "000"

    return os.path.join(root_export_dir, next_sub_dir)


class _Empty(object):
    def purge_cache(self):
        pass


class Serialized_Dict(object):
    __slots__ = ["_ser_data", "_data"]
    cache_len = 100
    _cache_ref = [_Empty()] * cache_len
    MSGPACK_EXT_CODE = 13

    def __init__(self, python_dict=None, msgpack_bytes=None):
        if type(python_dict) is dict:
            self._ser_data = msgpack.packb(
                python_dict, use_bin_type=True, default=self.packing_hook
            )
        elif type(msgpack_bytes) is bytes:
            self._ser_data = msgpack_bytes
        else:
            raise ValueError(
                "You did not supply mapping or payload to Serialized_Dict."
            )
        self._data = None

    def _deser(self):
        if not self._data:
            self._data = msgpack.unpackb(
                self._ser_data,
                raw=False,
                use_list=False,
                object_hook=self.unpacking_object_hook,
                ext_hook=self.unpacking_ext_hook,
            )
            self._cache_ref.pop(0).purge_cache()
            self._cache_ref.append(self)

    def __getstate__(self):
        return self._ser_data

    def __setstate__(self, msgpack_bytes):
        self._ser_data = msgpack_bytes
        self._data = None

    @classmethod
    def unpacking_object_hook(self, obj):
        if type(obj) is dict:
            return obj

    @classmethod
    def packing_hook(self, obj):
        if isinstance(obj, self):
            return msgpack.ExtType(self.MSGPACK_EXT_CODE, obj.serialized)
        raise TypeError("can't serialize {}({})".format(type(obj), repr(obj)))

    @classmethod
    def unpacking_ext_hook(self, code, data):
        if code == self.MSGPACK_EXT_CODE:
            return self(msgpack_bytes=data)
        return msgpack.ExtType(code, data)

    def purge_cache(self):
        self._data = None

    @property
    def serialized(self):
        return self._ser_data

    def __setitem__(self, key, item):
        raise NotImplementedError()

    def __getitem__(self, key):
        self._deser()
        return self._data[key]

    def __repr__(self):
        self._deser()
        return "Serialized_Dict({})".format(repr(self._data))

    @property
    def len(self):
        """Replacement implementation for __len__

        If __len__ is defined numpy will recognize this as nested structure and
        start deserializing everything instead of using this object as it is.
        """
        self._deser()
        return len(self._data)

    def __delitem__(self, key):
        raise NotImplementedError()

    def get(self, key, default):
        try:
            return self[key]
        except KeyError:
            return default

    def clear(self):
        raise NotImplementedError()

    def copy(self):
        self._deser()
        return self._data.copy()

    def has_key(self, k):
        self._deser()
        return k in self._data

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def keys(self):
        self._deser()
        return self._data.keys()

    def values(self):
        self._deser()
        return self._data.values()

    def items(self):
        self._deser()
        return self._data.items()

    def pop(self, *args):
        raise NotImplementedError()

    def __cmp__(self, dict_):
        self._deser()
        return self._data.__cmp__(dict_)

    def __contains__(self, item):
        self._deser()
        return item in self._data

    def __iter__(self):
        self._deser()
        return iter(self._data)


def save_intrinsics(directory, cam_name, resolution, intrinsics):
    """
    Saves camera intrinsics calibration to a file. For each unique camera name we maintain a single file containing all calibrations associated with this camera name.
    :param directory: Directory to which the intrinsics file will be written
    :param cam_name: Name of the camera, e.g. 'Pupil Cam 1 ID2'
    :param resolution: Camera resolution given as a tuple. This needs to match the resolution the calibration has been computed with.
    :param intrinsics: The camera intrinsics dictionary.
    :return:
    """
    # Try to load previous camera calibrations
    save_path = os.path.join(
        directory, "{}.intrinsics".format(cam_name.replace(" ", "_"))
    )
    try:
        calib_dict = load_object(save_path, allow_legacy=False)
    except Exception:
        calib_dict = {}

    calib_dict["version"] = 1
    calib_dict[str(resolution)] = intrinsics

    save_object(calib_dict, save_path)
    logger.info(
        "Calibration for camera {} at resolution {} saved to {}".format(
            cam_name, resolution, save_path
        )
    )


def save_extrinsics(directory, cam_name, resolution, extrinsics):
    """
    Saves camera intrinsics calibration to a file. For each unique camera name we maintain a single file containing all calibrations associated with this camera name.
    :param directory: Directory to which the intrinsics file will be written
    :param cam_name: Name of the camera, e.g. 'Pupil Cam 1 ID2'
    :param resolution: Camera resolution given as a tuple. This needs to match the resolution the calibration has been computed with.
    :param extrinsics: The camera extrinsics dictionary.
    :return:
    """
    # Try to load previous camera calibrations
    save_path = os.path.join(
        directory, "{}.extrinsics".format(cam_name.replace(" ", "_"))
    )
    try:
        calib_dict = load_object(save_path, allow_legacy=False)
    except Exception:
        calib_dict = {}

    calib_dict["version"] = 1
    if str(resolution) in calib_dict:
        calib_dict[str(resolution)].update(extrinsics)
    else:
        calib_dict[str(resolution)] = extrinsics

    save_object(calib_dict, save_path)
    logger.info(
        "Extrinsics for camera {} at resolution {} saved to {}".format(
            cam_name, resolution, save_path
        )
    )
