import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier
import soundfile as sf
from pathlib import Path
import numpy as np

class VoiceCloner:
    def __init__(self):
        # Initialize models and processor
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        
        # Move models to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_model.to(self.device)
        self.vocoder.to(self.device)

    def extract_speaker_embedding(self, wav_directory):
        """
        Extract speaker embedding from multiple WAV files in a directory using X-vector
        """
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
                # Remove extra dimensions and get the raw 512-dim vector
                embedding = embedding.squeeze()
                embeddings.append(embedding.cpu())
        
        # Average all embeddings and ensure correct shape
        average_embedding = torch.mean(torch.stack(embeddings), dim=0)
        # SpeechT5 expects shape [1, 512]
        average_embedding = average_embedding.reshape(1, -1)
        return average_embedding

    def synthesize_speech(self, text, speaker_embedding, output_path):
        """
        Synthesize speech from text using the extracted speaker embedding
        """
        # Prepare text input
        inputs = self.processor(text=text, return_tensors="pt")
        
        # Generate speech
        with torch.no_grad():
            speech = self.tts_model.generate_speech(
                inputs["input_ids"].to(self.device), 
                speaker_embeddings=speaker_embedding.to(self.device),
                vocoder=self.vocoder
            )
        
        # Save the generated speech
        sf.write(output_path, speech.cpu().numpy(), samplerate=16000)
        
    def clone_and_speak(self, voice_samples_dir, text, output_path):
        """
        Main function to clone voice and generate speech
        """
        print("Extracting speaker embedding...")
        speaker_embedding = self.extract_speaker_embedding(voice_samples_dir)
        
        print(f"Embedding shape: {speaker_embedding.shape}")  # Debug print
        
        print("Generating speech...")
        self.synthesize_speech(text, speaker_embedding, output_path)
        print(f"Speech generated and saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Initialize the voice cloner
    cloner = VoiceCloner()
    
    # Directory containing voice samples (.wav files)
    voice_samples_dir = "voice_samples"
    
    # Text to synthesize
    text = "Hello, this is a test of voice cloning using SpeechT5 with X-vector embeddings."
    
    # Output path for generated speech
    output_path = "generated_speech.wav"
    
    # Clone voice and generate speech
    cloner.clone_and_speak(voice_samples_dir, text, output_path)