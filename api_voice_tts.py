from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# generate speech by cloning a voice using default settings
tts.tts_to_file(#text="I am the Scourge, I am FEAR, I am DEATH, I AM NAIR ZOOL, I AM... THE LICH KING!",
                text="For my father, the king!",
                file_path="output_lk.wav",
                #speaker_wav="arthas_samples/DK_WC3.wav",
                speaker_wav="voice_samples/Recording0005.wav",
                emotion="angry",
                language="en")