from transformers import AutoProcessor, AutoModel
import scipy
import numpy as np

max_lenght = 128
processor = AutoProcessor.from_pretrained(
    "suno/bark",
    cache_dir="checkpoints/bark/processor",
)
model = AutoModel.from_pretrained(
    "suno/bark",
    cache_dir="checkpoints/bark/model",
)
model.to("cuda")
model = model.to_bettertransformer()
sampling_rate = model.generation_config.sample_rate


wave_collector = []
for sentence in [
    "This paper presents fairseq S^2, a fairseq extension for speech synthesis.",
    "We implement a number of autoregressive (AR) and non-AR text-to-speech models, and their multi-speaker variants.",
    "To enable training speech synthesis models with less curated data, a number of preprocessing tools are built and their importance is shown empirically.",
    "To facilitate faster iteration of development and analysis, a suite of automatic metrics is included.",
    "Apart from the features added specifically for this extension, fairseq S^2 also benefits from the scalability offered by fairseq and can be easily integrated with other state-of-the-art systems provided in this framework.",
    "The code, documentation, and pre-trained models are available at this https URL.",
]:
    inputs = processor(
        text=sentence,
        voice_preset="v2/en_speaker_6",
        return_tensors="pt",
    )

    for key, value in inputs.items():
        inputs[key] = value.to("cuda")

    speech_values = model.generate(**inputs, semantic_temperature=0.6)
    wave_collector.append(
        speech_values.cpu().numpy().flatten()
    )

scipy.io.wavfile.write(
    "bark_out_.wav",
    rate=sampling_rate,
    data=np.concatenate(wave_collector),
)
