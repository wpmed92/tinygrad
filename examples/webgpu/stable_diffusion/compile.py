import os
from extra.export_model import export_model
from examples.stable_diffusion import StableDiffusion
from tinygrad.nn.state import get_state_dict, safe_save, safe_load_metadata, torch_load, load_state_dict
from tinygrad.tensor import Tensor
from tinygrad import Device, dtypes
from tinygrad.helpers import fetch
from typing import NamedTuple, Any, List
from pathlib import Path
import requests
import argparse
import numpy as np

def fetch_dep(file, url):
  with open(file, "w", encoding="utf-8") as f:
    f.write(requests.get(url).text.replace("https://huggingface.co/wpmed/tinygrad-sd-f16/raw/main/bpe_simple_vocab_16e6.mjs", "./bpe_simple_vocab_16e6.mjs"))

if __name__ == "__main__":
  fetch_dep(os.path.join(os.path.dirname(__file__), "clip_tokenizer.js"), "https://huggingface.co/wpmed/tinygrad-sd-f16/raw/main/clip_tokenizer.js")
  fetch_dep(os.path.join(os.path.dirname(__file__), "bpe_simple_vocab_16e6.mjs"), "https://huggingface.co/wpmed/tinygrad-sd-f16/raw/main/bpe_simple_vocab_16e6.mjs")
  parser = argparse.ArgumentParser(description='Run Stable Diffusion', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--remoteweights', action='store_true', help="Use safetensors from Huggingface, or from local")
  args = parser.parse_args()
  Device.DEFAULT = "WEBGPU"

  Tensor.no_grad = True
  model = StableDiffusion()

  # load in weights
  load_state_dict(model, torch_load(fetch('https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt', 'sd-v1-4.ckpt'))['state_dict'], strict=False)

  # convert weights to f16
  for k,v in get_state_dict(model).items():
    if k.startswith("model"):
      v.replace(v.cast(dtypes.float16).realize())

  class Step(NamedTuple):
    name: str = ""
    input: List[Tensor] = []
    forward: Any = None
    state_from: Any = None
    prefix: str = ""

  sub_steps = [
    Step(name = "textModel", input = [Tensor.randn(1, 77)], forward = model.cond_stage_model.transformer.text_model, prefix="cond_stage_model.transformer.text_model."),
    Step(name = "diffusor", input = [Tensor.randn(1, 77, 768), Tensor.randn(1, 77, 768), Tensor.randn(1,4,64,64), Tensor.rand(1), Tensor.randn(1), Tensor.randn(1), Tensor.randn(1)], forward = model),
    Step(name = "decoder", input = [Tensor.randn(1,4,64,64)], forward = model.decode, state_from = model.first_stage_model, prefix="first_stage_model.")
  ]

  for step in sub_steps:
    prg, inp_sizes, out_sizes, state = export_model(
      step.forward, 
      Device.DEFAULT.lower(), 
      *step.input, 
      model_name=step.name,
      state_from = step.state_from,
      prefix = step.prefix
    )

    dirname = Path(__file__).parent

    if step.name == "diffusor":
      safe_save(state, (dirname / "net.safetensors").as_posix())

    with open(dirname / f"net_{step.name}.js", "w") as program_file:
       program_file.write(prg)

