# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/cuda_wrapper.py

"""This file is a pure Python wrapper for the mtml library.
It avoids the need to compile a separate shared library, and is
convenient for use when we just need to call a few functions.
"""

import ctypes
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sglang.srt.utils import is_musa

_is_musa = is_musa()

logger = logging.getLogger(__name__)

# === export types and functions from mtml to Python ===
# for the original cudart definition, please check
# https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

MtmlReturn = ctypes.c_int
MtmlDeviceP2PCaps = ctypes.c_int
MtmlDeviceP2PStatus = ctypes.c_int


class MtmlLibrary(ctypes.Structure):
    _fields_ = []


class MtmlDevice(ctypes.Structure):
    _fields_ = []


class MtmlMtLinkSpec(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("bandWidth", ctypes.c_uint),
        ("linkNum", ctypes.c_uint),
        ("rsvd", ctypes.c_uint * 4),
    ]


# MTML ERROR CODES
MTML_SUCCESS = 0


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


class MTMLLibrary:
    # This is added for pynvml， not for mtml
    NVML_P2P_CAPS_INDEX_NVLINK = 0

    # Recommended buffer size in bytes that is guaranteed to be enough to hold the UUID string of a device (including a null terminator).
    DEVICE_UUID_BUFFER_SIZE = 48

    # NVML P2P Status
    NVML_P2P_STATUS_OK = 0
    NVML_P2P_STATUS_NOT_OK = 1

    # MtmlMtLinkState
    MTML_MTLINK_STATE_UP = 1

    mtml_funcs = [
        "mtmlLibraryInit",
        "mtmlLibraryShutDown",
        "mtmlLibraryInitDeviceByIndex",
        "mtmlDeviceGetMtLinkSpec",
        "mtmlDeviceGetMtLinkState",
        "mtmlDeviceGetMtLinkRemoteDevice",
        "mtmlDeviceGetUUID",
    ]

    class NVMLError(Exception):
        def __init__(self, code, msg="NVML Error"):
            self.code = code
            self.msg = msg
            super().__init__(f"{msg}, code={code}")

        def __str__(self):
            return f"NVMLError(code={self.code}, msg={self.msg})"

        def __repr__(self):
            return f"<NVMLError code={self.code} msg={self.msg}>"

    exported_functions = [
        # MtmlReturn MTML_API mtmlLibraryInit(MtmlLibrary **lib);
        Function(
            "mtmlLibraryInit",
            MtmlReturn,
            [ctypes.POINTER(ctypes.POINTER(MtmlLibrary))],
        ),
        # MtmlReturn MTML_API mtmlLibraryShutDown(MtmlLibrary *lib);
        Function("mtmlLibraryShutDown", MtmlReturn, [ctypes.POINTER(MtmlLibrary)]),
        # MtmlReturn MTML_API mtmlLibraryInitDeviceByIndex(const MtmlLibrary *lib, unsigned int index, MtmlDevice **dev);
        Function(
            "mtmlLibraryInitDeviceByIndex",
            MtmlReturn,
            [
                ctypes.POINTER(MtmlLibrary),
                ctypes.c_uint,
                ctypes.POINTER(ctypes.POINTER(MtmlDevice)),
            ],
        ),
        # MtmlReturn MTML_API mtmlDeviceGetMtLinkSpec(const MtmlDevice* device, MtmlMtLinkSpec* spec);
        Function(
            "mtmlDeviceGetMtLinkSpec",
            MtmlReturn,
            [ctypes.POINTER(MtmlDevice), ctypes.POINTER(MtmlMtLinkSpec)],
        ),
        # MtmlReturn MTML_API mtmlDeviceGetMtLinkState(const MtmlDevice* device, unsigned int linkId, MtmlMtLinkState* state);
        Function(
            "mtmlDeviceGetMtLinkState",
            MtmlReturn,
            [ctypes.POINTER(MtmlDevice), ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)],
        ),
        # MtmlReturn MTML_API mtmlDeviceGetMtLinkRemoteDevice(const MtmlDevice* device, unsigned int linkId, MtmlDevice** remoteDevice);
        Function(
            "mtmlDeviceGetMtLinkRemoteDevice",
            MtmlReturn,
            [
                ctypes.POINTER(MtmlDevice),
                ctypes.c_uint,
                ctypes.POINTER(ctypes.POINTER(MtmlDevice)),
            ],
        ),
        # MtmlReturn MTML_API mtmlDeviceGetUUID(const MtmlDevice *dev, char *uuid, unsigned int length);
        Function(
            "mtmlDeviceGetUUID",
            MtmlReturn,
            [ctypes.POINTER(MtmlDevice), ctypes.c_char_p, ctypes.c_uint],
        ),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        so_file = "/usr/lib/libmtml.so"  # mtml default path
        if so_file not in MTMLLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            MTMLLibrary.path_to_library_cache[so_file] = lib

        self.lib = MTMLLibrary.path_to_library_cache[so_file]

        if so_file not in MTMLLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in MTMLLibrary.exported_functions:
                try:
                    f = getattr(self.lib, func.name)
                    f.restype = func.restype
                    f.argtypes = func.argtypes
                    cname = None
                    for k in self.mtml_funcs:
                        if k == func.name:
                            cname = k
                            break
                    if cname is None:
                        raise RuntimeError(
                            f"Function {func.name} not found in mtml_funcs"
                        )
                    _funcs[cname] = f
                except AttributeError as e:
                    raise
            MTMLLibrary.path_to_dict_mapping[so_file] = _funcs

        self.funcs = MTMLLibrary.path_to_dict_mapping[so_file]

    def nvmlInit(self) -> None:
        lib_ptr = ctypes.POINTER(MtmlLibrary)()
        ret = self.funcs["mtmlLibraryInit"](ctypes.byref(lib_ptr))
        if ret != 0:
            raise self.NVMLError(ret, "mtmlLibraryInit failed")

        self._lib_handle = lib_ptr

    def nvmlShutdown(self) -> None:
        if self._lib_handle is None:
            return
        ret = self.funcs["mtmlLibraryShutDown"](self._lib_handle)
        if ret != 0:
            raise RuntimeError(f"mtmlLibraryShutDown failed, code={ret}")
        self._lib_handle = None

    def nvmlDeviceGetHandleByIndex(self, index: int) -> ctypes.POINTER(MtmlDevice):
        if self._lib_handle is None:
            raise self.NVMLError(-1, "Library not initialized. Call nvmlInit first.")
        dev_ptr = ctypes.POINTER(MtmlDevice)()
        ret = self.funcs["mtmlLibraryInitDeviceByIndex"](
            self._lib_handle, index, ctypes.byref(dev_ptr)
        )
        if ret != MTML_SUCCESS:
            raise self.NVMLError(ret, "mtmlDeviceGetHandleByIndex failed")
        return dev_ptr

    def nvmlDeviceGetP2PStatus(
        self, dev1, dev2, caps: MtmlDeviceP2PCaps
    ) -> MtmlDeviceP2PStatus:
        # Default to NOT_OK
        status = MtmlDeviceP2PStatus()
        status.value = self.NVML_P2P_STATUS_NOT_OK

        try:
            # Get two devices' UUIDs
            dev1_uuid = (ctypes.c_char * self.DEVICE_UUID_BUFFER_SIZE)()
            ret = self.funcs["mtmlDeviceGetUUID"](
                dev1, dev1_uuid, self.DEVICE_UUID_BUFFER_SIZE
            )
            if ret != MTML_SUCCESS:
                return status
            dev2_uuid = (ctypes.c_char * self.DEVICE_UUID_BUFFER_SIZE)()
            ret = self.funcs["mtmlDeviceGetUUID"](
                dev2, dev2_uuid, self.DEVICE_UUID_BUFFER_SIZE
            )
            if ret != MTML_SUCCESS:
                return status

            spec = MtmlMtLinkSpec()
            ret = self.funcs["mtmlDeviceGetMtLinkSpec"](dev1, ctypes.byref(spec))
            if ret != MTML_SUCCESS:
                return status

            # Find if there is any mtlink connecting dev1 and dev2
            for link_id in range(spec.linkNum):
                link_state = ctypes.c_uint()
                ret = self.funcs["mtmlDeviceGetMtLinkState"](
                    dev1, link_id, ctypes.byref(link_state)
                )
                if ret != MTML_SUCCESS:
                    continue

                remote_dev_ptr = ctypes.POINTER(MtmlDevice)()
                ret = self.funcs["mtmlDeviceGetMtLinkRemoteDevice"](
                    dev1, link_id, ctypes.byref(remote_dev_ptr)
                )
                if ret != MTML_SUCCESS:
                    continue

                remote_dev_uuid = (ctypes.c_char * self.DEVICE_UUID_BUFFER_SIZE)()
                ret = self.funcs["mtmlDeviceGetUUID"](
                    remote_dev_ptr, remote_dev_uuid, self.DEVICE_UUID_BUFFER_SIZE
                )
                if ret != MTML_SUCCESS:
                    continue

                if (
                    dev2_uuid == remote_dev_uuid
                    and link_state.value == self.MTML_MTLINK_STATE_UP
                ):
                    status.value = self.NVML_P2P_STATUS_OK
                    break

        except Exception as e:
            logger.exception(f"Unexpected error in mtmlMtlinkStatus: {e}")

        return status.value
