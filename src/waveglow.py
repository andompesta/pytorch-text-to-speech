import torch
import os
from src.nvidia_tacotron2.hubconf import nvidia_waveglow

waveglow = nvidia_waveglow(path_=os.path.join(os.environ["PT_HUB"], "nvidia_waveglowpyt_fp32_20190306.pth"))
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow.eval()
