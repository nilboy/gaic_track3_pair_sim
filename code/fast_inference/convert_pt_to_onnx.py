from typing import Dict, List

import numpy as np
import torch
from torch import nn

def convert_pt_to_onnx(model: nn.Module,
                       inputs: Dict[str, np.array],
                       input_names: List[str],
                       output_names: List[str],
                       output_model_path: str,
                       use_cuda: bool = True,
                       opset_version: int=10,
                       use_external_data_format=False):
    model = model.eval()
    inputs = {k: torch.tensor(v) for k, v in inputs.items()}
    if use_cuda:
        model = model.cuda()
        for k, v in inputs.items():
            inputs[k] = v.cuda()
    else:
        model = model.cpu()
    torch.onnx.export(model,
                      tuple([inputs[input_name] for input_name in input_names]),
                      output_model_path,
                      export_params=True,
                      training=False,
                      opset_version=opset_version,
                      do_constant_folding=True,
                      input_names=input_names,
                      output_names=output_names,
                      use_external_data_format=use_external_data_format)
