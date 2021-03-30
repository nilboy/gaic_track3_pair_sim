from __future__ import print_function
import tensorrt as trt
from onnx.backend.base import Device
import onnx
import six

# HACK Should look for a better way/place to do this
from ctypes import cdll, c_char_p

libcudart = cdll.LoadLibrary('libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p

def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + error_string)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def convert_onnx_to_tensorrt(onnx_model_path: str,
                             trt_model_path: str,
                             device: str='CUDA:0',
                             fp16: bool=True,
                             max_workspace_size: int=None,
                             max_batch_size: int=32):
    if not isinstance(device, Device):
        device= Device(device)
    cudaSetDevice(device.device_id)
    logger = TRT_LOGGER
    builder = trt.Builder(logger)
    builder.fp16_mode = fp16
    network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not trt.init_libnvinfer_plugins(TRT_LOGGER, ""):
        msg = "Failed to initialize TensorRT's plugin library."
        raise RuntimeError(msg)

    if not parser.parse_from_file(onnx_model_path):
        error = parser.get_error(0)
        msg = "While parsing node number %i:\n" % error.node()
        msg += ("%s:%i In function %s:\n[%i] %s" %
                (error.file(), error.line(), error.func(),
                 error.code(), error.desc()))
        raise RuntimeError(msg)
    if max_workspace_size is None:
        max_workspace_size = 1 << 28
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = max_workspace_size
    # for layer in network:
    #     print(layer.name)
    print(network[-1].get_output(0).shape)
    trt_engine = builder.build_cuda_engine(network)
    if trt_engine is None:
        raise RuntimeError("Failed to build TensorRT engine from network")
    serialized_engine = trt_engine.serialize()
    with open(trt_model_path, 'wb') as f:
        f.write(bytearray(serialized_engine))
