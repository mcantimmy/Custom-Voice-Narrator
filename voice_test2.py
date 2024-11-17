import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier
import soundfile as sf
import torch.nn.functional as F

@dataclass
class AudioConfig:
    target_sample_rate: int = 16000
    min_silence_duration: float = 0.5
    silence_threshold: float = 0.05
    chunk_duration: float = 10.0

class VoiceCloner:
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the voice cloner with optional model caching
        """
        self.config = AudioConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models with caching if specified
        self.processor = SpeechT5Processor.from_pretrained(
            "microsoft/speecht5_tts",
            cache_dir=cache_dir
        )
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts",
            cache_dir=cache_dir
        ).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan",
            cache_dir=cache_dir
        ).to(self.device)
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            #savedir=cache_dir if cache_dir else None
        )

    def preprocess_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Preprocess audio by resampling, converting to mono, and removing silence
        """
        # Resample if necessary
        if sample_rate != self.config.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.config.target_sample_rate)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Remove silence
        return self.remove_silence(waveform)

    def remove_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Remove silence from audio using energy-based VAD
        """
        # Calculate energy
        energy = torch.norm(waveform, dim=0)
        energy = F.pad(energy, (1, 1))
        
        # Create mask for non-silent regions
        mask = energy > self.config.silence_threshold
        
        # Only keep segments longer than minimum silence duration
        min_samples = int(self.config.min_silence_duration * self.config.target_sample_rate)
        mask = self._remove_short_segments(mask, min_samples)
        
        return waveform[:, mask[1:-1]]

    @staticmethod
    def _remove_short_segments(mask: torch.Tensor, min_length: int) -> torch.Tensor:
        """
        Remove segments shorter than min_length
        """
        changes = torch.diff(mask.int())
        change_points = changes.nonzero().squeeze(-1)
        
        if len(change_points) < 2:
            return mask
            
        segments = change_points.reshape(-1, 2)
        valid_segments = segments[:, 1] - segments[:, 0] >= min_length
        
        new_mask = mask.clone()
        for start, end in segments[~valid_segments]:
            new_mask[start:end] = False
            
        return new_mask

    def chunk_audio(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """
        Split audio into chunks to avoid memory issues
        """
        chunk_size = int(self.config.chunk_duration * self.config.target_sample_rate)
        return torch.split(waveform, chunk_size, dim=1)

    def extract_speaker_embedding(self, wav_directory: str) -> torch.Tensor:
        """
        Extract speaker embedding from multiple WAV files in a directory
        """
        wav_files = list(Path(wav_directory).glob("*.wav"))
        if not wav_files:
            raise ValueError(f"No WAV files found in {wav_directory}")

        embeddings = []
        
        for wav_path in wav_files:
            try:
                # Load audio
                waveform, sample_rate = torchaudio.load(wav_path)
                
                # Preprocess audio
                waveform = self.preprocess_audio(waveform, sample_rate)
                
                # Process in chunks if necessary
                chunks = self.chunk_audio(waveform)
                
                # Extract embedding for each chunk
                for chunk in chunks:
                    if chunk.shape[1] < 100:  # Skip very short chunks
                        continue
                    with torch.no_grad():
                        embedding = self.speaker_encoder.encode_batch(chunk)
                        embedding = embedding.squeeze()
                        embeddings.append(embedding.cpu())
                
            except Exception as e:
                print(f"Warning: Error processing {wav_path}: {str(e)}")
                continue

        if not embeddings:
            raise ValueError("No valid embeddings could be extracted from the audio files")

        # Average all embeddings with outlier removal
        embeddings_tensor = torch.stack(embeddings)
        mean_embedding = torch.mean(embeddings_tensor, dim=0)
        distances = torch.norm(embeddings_tensor - mean_embedding, dim=1)
        
        # Remove embeddings that are too far from the mean (potential outliers)
        valid_embeddings = embeddings_tensor[distances < torch.mean(distances) + torch.std(distances)]
        average_embedding = torch.mean(valid_embeddings, dim=0).reshape(1, -1)
        
        return average_embedding

    def synthesize_speech(self, text: str, speaker_embedding: torch.Tensor, output_path: str):
        """
        Synthesize speech from text using the extracted speaker embedding
        """
        # Split long text into sentences to avoid memory issues
        sentences = text.split('.')
        audio_segments = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Prepare text input
            inputs = self.processor(text=sentence.strip() + '.', return_tensors="pt")
            
            # Generate speech
            with torch.no_grad():
                speech = self.tts_model.generate_speech(
                    inputs["input_ids"].to(self.device), 
                    speaker_embeddings=speaker_embedding.to(self.device),
                    vocoder=self.vocoder
                )
                audio_segments.append(speech.cpu().numpy())

        # Concatenate all segments
        final_speech = np.concatenate(audio_segments)
        
        # Add small silence between sentences
        silence = np.zeros(int(0.2 * self.config.target_sample_rate))
        final_speech = np.concatenate([
            final_speech,
            silence
        ])
        
        # Save the generated speech
        sf.write(output_path, final_speech, samplerate=self.config.target_sample_rate)

    def clone_and_speak(self, voice_samples_dir: str, text: str, output_path: str):
        """
        Main function to clone voice and generate speech with error handling
        """
        try:
            print("Extracting speaker embedding...")
            speaker_embedding = self.extract_speaker_embedding(voice_samples_dir)
            
            print("Generating speech...")
            self.synthesize_speech(text, speaker_embedding, output_path)
            print(f"Speech generated and saved to {output_path}")
            
        except Exception as e:
            print(f"Error during voice cloning: {str(e)}")
            raise

# Example usage with improved error handling
if __name__ == "__main__":
    try:
        # Initialize with optional cache directory
        cloner = VoiceCloner(cache_dir="./model_cache")
        
        voice_samples_dir = "voice_samples"
        text = "Hello, this is a test of voice cloning using SpeechT5 with X-vector embeddings."
        output_path = "generated_speech2.wav"
        
        cloner.clone_and_speak(voice_samples_dir, text, output_path)
        
    except Exception as e:
        print(f"Failed to complete voice cloning: {str(e)}")