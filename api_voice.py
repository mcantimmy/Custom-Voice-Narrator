from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torchaudio

config = XttsConfig()
config.load_json("./model/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./model/", eval=True)
#model.cuda()

outputs = model.synthesize(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    config,
    speaker_wav="lichking_samples/The Lich King Audio Part 1.wav",
    gpt_cond_len=3,
    language="en",
)

# Output path for generated speech
output_path = "xtts_cloned_speech.wav"

torchaudio.save(output_path, outputs["wav"], outputs["sample_rate"])
print(f"Speech generated successfully and saved to {output_path}")