import onnx
import torch
from network import DQN

import config

model = DQN(n_observations=config.n_observations, n_actions=config.n_actions)
model.load_state_dict(torch.load('best.pt', weights_only=True))
model.eval()

input_tensor = torch.randn(1, config.n_observations)
torch.onnx.export(
    model,
    (input_tensor,),
    "dqn_model.onnx",
    dynamo=False,
    input_names=["observations"],
    output_names=["q_values"],
)

onnx_model = onnx.load("dqn_model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX export successful")
