import torch
import os

from src.nvidia_tacotron2.hubconf import nvidia_tacotron2

torch.hub.set_dir(os.environ["PT_HUB"])
tacotron2 = nvidia_tacotron2(path_=os.path.join(os.environ["PT_HUB"], "nvidia_tacotron2pyt_fp32_20190306.pth"))

tacotron2.eval()