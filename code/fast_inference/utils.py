from __future__ import print_function
from onnx.backend.base import Device

# HACK Should look for a better way/place to do this
from ctypes import cdll, c_char_p

libcudart = cdll.LoadLibrary('libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p

def cudaSetDevice(device_id):
    ret = libcudart.cudaSetDevice(device_id)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + error_string)
