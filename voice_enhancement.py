import torch
import torchaudio
from transformers import VitsModel, AutoProcessor
from speechbrain.pretrained import EncoderClassifier
import soundfile as sf
from pathlib import Path
import numpy as np

class VoiceCloner:
    def __init__(self):
        # Initialize models and processor
        self.processor = AutoProcessor.from_pretrained("facebook/mms-tts")
        self.model = VitsModel.from_pretrained("facebook/mms-tts")
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        
        # Move models to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def extract_speaker_embedding(self, wav_directory):
        """
        Extract speaker embedding from multiple WAV files in a directory using X-vector
        """
        print(Path(wav_directory).glob("*.wav"))
        wav_files = list(Path(wav_directory).glob("*.wav"))
        if not wav_files:
            raise ValueError(f"No WAV files found in {wav_directory}")

        embeddings = []
        
        for wav_path in wav_files:
            # Load and resample audio if necessary
            waveform, sample_rate = torchaudio.load(wav_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Extract embedding using SpeechBrain's X-vector
            with torch.no_grad():
                embedding = self.speaker_encoder.encode_batch(waveform)
                embedding = embedding.squeeze()
                embeddings.append(embedding.cpu())
        
        # Average all embeddings and ensure correct shape
        average_embedding = torch.mean(torch.stack(embeddings), dim=0)
        average_embedding = average_embedding.reshape(1, -1)
        return average_embedding

    def synthesize_speech(self, text, speaker_embedding, output_path, language="eng"):
        """
        Synthesize speech from text using MMS-TTS
        """
        # Prepare text input
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            language=language
        ).to(self.device)
        
        # Generate speech
        with torch.no_grad():
            # Forward pass through the model
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                speaker_embeddings=speaker_embedding.to(self.device)
            )
            
            # Get the generated waveform
            waveform = outputs.waveform.squeeze().cpu().numpy()
        
        # Save the generated speech
        sf.write(output_path, waveform, samplerate=16000)
        
    def clone_and_speak(self, voice_samples_dir, text, output_path, language="eng"):
        """
        Main function to clone voice and generate speech
        """
        print("Extracting speaker embedding...")
        speaker_embedding = self.extract_speaker_embedding(voice_samples_dir)
        
        print(f"Embedding shape: {speaker_embedding.shape}")  # Debug print
        
        print("Generating speech...")
        self.synthesize_speech(text, speaker_embedding, output_path, language)
        print(f"Speech generated and saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Initialize the voice cloner
    cloner = VoiceCloner()
    
    # Directory containing voice samples (.wav files)
    voice_samples_dir = "voice_samples"
    
    # Text to synthesize
    text = "Hello, this is a test of voice cloning using Facebook MMS-TTS."
    
    # Output path for generated speech
    output_path = "cloned_speech.wav"
    
    # Clone voice and generate speech (you can change language to any supported code)
    # Supported languages: eng, fra, deu, ita, esp, pol, tur, rus, nld, cmn, por, jpn
    cloner.clone_and_speak(voice_samples_dir, text, output_path, language="eng")