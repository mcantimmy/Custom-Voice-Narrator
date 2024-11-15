import torch
import torchaudio
from transformers import (
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model
)
import numpy as np
import librosa
import os
from typing import List, Union, Optional
from dataclasses import dataclass

@dataclass
class AudioStats:
    duration: float
    sample_rate: int
    num_samples: int
    max_amplitude: float
    min_amplitude: float

class VoiceCloner:
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the voice cloner with a specific model directory
        
        Args:
            model_dir: Directory to store model files. If None, uses ./models/
        """
        # Set up model directory
        if model_dir is None:
            model_dir = os.path.join(os.getcwd(), "models")
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            # Initialize TTS models
            print("Loading TTS model...")
            self.processor = SpeechT5Processor.from_pretrained(
                "microsoft/speecht5_tts",
                cache_dir=self.model_dir
            )
            
            self.model = SpeechT5ForTextToSpeech.from_pretrained(
                "microsoft/speecht5_tts",
                cache_dir=self.model_dir
            ).to(self.device)
            
            print("Loading vocoder model...")
            self.vocoder = SpeechT5HifiGan.from_pretrained(
                "microsoft/speecht5_hifigan",
                cache_dir=self.model_dir
            ).to(self.device)
            
            # Initialize speaker embedding model (using Wav2Vec2 instead of SpeechBrain)
            print("Loading speaker encoder model...")
            self.speaker_processor = Wav2Vec2FeatureExtractor.from_pretrained(
                "facebook/wav2vec2-base",
                cache_dir=self.model_dir
            )
            
            self.speaker_model = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base",
                cache_dir=self.model_dir
            ).to(self.device)
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    
    def _load_and_process_audio(self, audio_path: str, target_sr: int = 16000) -> torch.Tensor:
        """
        Load and preprocess audio file
        """
        try:
            print(f"Processing audio file: {audio_path}")
            # Load audio
            audio, sr = librosa.load(audio_path, sr=target_sr)
            
            # Remove silence and normalize
            audio = librosa.effects.trim(audio, top_db=20)[0]
            audio = librosa.util.normalize(audio)
            
            # Ensure minimum duration
            min_samples = target_sr * 1  # 1 second minimum
            if len(audio) < min_samples:
                print("Audio too short, padding...")
                audio = np.pad(audio, (0, min_samples - len(audio)))
            
            return torch.FloatTensor(audio)
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            raise
    
    def extract_speaker_embedding(self, audio_paths: Union[str, List[str]]) -> torch.Tensor:
        """
        Extract speaker embedding from voice samples using Wav2Vec2
        """
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        
        embeddings_list = []
        
        for audio_path in audio_paths:
            print(f"Extracting embedding from: {audio_path}")
            # Load and process audio
            audio = self._load_and_process_audio(audio_path)
            
            # Prepare for Wav2Vec2
            inputs = self.speaker_processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            )
            
            # Extract embeddings
            with torch.no_grad():
                outputs = self.speaker_model(
                    inputs.input_values.to(self.device),
                    attention_mask=inputs.attention_mask.to(self.device)
                )
                # Use the mean of the last hidden state as the embedding
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings_list.append(embedding)
        
        # Average the embeddings if multiple samples
        final_embedding = torch.mean(torch.stack(embeddings_list), dim=0)
        
        # Project to the expected size for SpeechT5 (256)
        projection = torch.nn.Linear(768, 256).to(self.device)
        final_embedding = projection(final_embedding)
        
        return final_embedding.unsqueeze(0)
    
    def generate_speech(self, 
                       text: str,
                       speaker_embeddings: torch.Tensor,
                       output_path: str,
                       speed_factor: float = 1.0) -> None:
        """
        Generate speech from text using speaker embeddings
        """
        try:
            print("Generating speech...")
            # Prepare text input
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Move speaker embeddings to device
            speaker_embeddings = speaker_embeddings.to(self.device)
            
            # Generate speech
            speech = self.model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings,
                vocoder=self.vocoder
            )
            
            # Apply speed modification if needed
            if speed_factor != 1.0:
                speech = librosa.effects.time_stretch(speech.cpu().numpy(), rate=speed_factor)
                speech = torch.from_numpy(speech)
            
            # Save the audio
            speech = speech.cpu().numpy()
            speech = speech / np.abs(speech).max()
            
            torchaudio.save(
                output_path,
                torch.tensor(speech).unsqueeze(0),
                sample_rate=16000
            )
            print(f"Speech saved to: {output_path}")
            
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            raise

def main():
    try:
        # Initialize with a specific model directory
        model_dir = os.path.join(os.getcwd(), "voice_cloner_models")
        print(f"Initializing voice cloner with model directory: {model_dir}")
        cloner = VoiceCloner(model_dir)
        
        # Set paths
        voice_sample = "path/to/your/sample.wav"  # Replace with your audio file
        output_path = "generated_speech.wav"
        text = "This is a test of the voice cloning system."
        
        # Extract speaker embedding
        print("Extracting speaker embedding...")
        speaker_embeddings = cloner.extract_speaker_embedding(voice_sample)
        
        # Generate speech
        print("Generating speech...")
        cloner.generate_speech(text, speaker_embeddings, output_path)
        
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()