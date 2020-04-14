from src.tacotron import tacotron2
from src.waveglow import waveglow

import torch
import numpy as np
from scipy.io.wavfile import write


text = "hi Nico, how are you doing ? Do you like my artificial voice ?"

# preprocessing
sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.from_numpy(sequence).to(dtype=torch.int64)

# run the models
with torch.no_grad():
    _, mel, _, _ = tacotron2.infer(sequence)
    audio = waveglow.infer(mel)
audio_numpy = audio[0].data.cpu().numpy()
rate = 22050


write("audio.wav", rate, audio_numpy)
