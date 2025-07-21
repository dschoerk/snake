



import onnx
import torch
from network import DQN

import config

model = DQN(n_observations=config.n_observations, n_actions=config.n_actions)
model.load_state_dict(torch.load('best.pt', weights_only=True))

# Example input tensor
input_tensor = (torch.randn(1, config.n_observations),)
onnx_model = torch.onnx.export(
    model,
    input_tensor,
    "dqn_model.onnx",
    dynamo=False,
    )

onnx_model = onnx.load("dqn_model.onnx")
onnx.checker.check_model(onnx_model)