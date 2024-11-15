from voice_cloner import VoiceCloner
import os

# Create a directory for models in your current directory
model_dir = os.path.join(os.getcwd(), "voice_cloner_models")

# Initialize the cloner
cloner = VoiceCloner(model_dir)

# Replace with your audio file path
voice_sample = "voice_samples/Recording0003.wav"
text = "This is a test of the voice cloning system."
output_path = "output.wav"

# Generate speech
speaker_embeddings = cloner.extract_speaker_embedding(voice_sample)
cloner.generate_speech(text, speaker_embeddings, output_path)