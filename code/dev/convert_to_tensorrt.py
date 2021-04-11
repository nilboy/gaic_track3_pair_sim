from fast_inference.convert_onnx_to_tensorrt import convert_onnx_to_tensorrt

convert_onnx_to_tensorrt("deploy-model/onnx/model-32/model",
                         "deploy-model/tensorrt/model-32.engine",
                         'CUDA:0',
                         max_batch_size=1)
convert_onnx_to_tensorrt("deploy-model/onnx/model-128/model",
                         "deploy-model/tensorrt/model-128.engine",
                         'CUDA:0',
                         max_batch_size=1)
