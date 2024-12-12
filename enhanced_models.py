import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Optional
import logging
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
from TTS.config import load_config
from TTS.tts.models import setup_model
from torch.nn import functional as F

class EnhancedVoiceCloner:
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the voice cloning system with Coqui TTS
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Initialize TTS model manager
        self.manager = ModelManager()
        
        # Get YourTTS model
        model_name = "tts_models/multilingual/multi-dataset/your_tts"
        model_path, config_path, _ = self.manager.download_model(model_name)
        
        # Load configs
        self.config = load_config(config_path)
        self.config.use_cuda = torch.cuda.is_available()
        
        # Load model
        self.model = setup_model(self.config)
        self.model.load_checkpoint(self.config, model_path)
        self.model.eval()
        
        if self.config.use_cuda:
            self.model.cuda()

        # Set up synthesizer
        self.synthesizer = Synthesizer(
            self.model,
            self.config,
            use_cuda=self.config.use_cuda
        )
        
        # Set up default configurations
        self.config.audio = {
            'sample_rate': 22050,  # YourTTS default
            'hop_length': 256,
            'win_length': 1024,
            'n_mels': 80,
            'mel_fmin': 0,
            'mel_fmax': None
        }

    def _preprocess_audio(self, waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
        """Preprocess audio with advanced normalization"""
        # Resample if necessary
        if original_sr != self.config.audio['sample_rate']:
            resampler = torchaudio.transforms.Resample(
                original_sr, self.config.audio['sample_rate']
            )
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply advanced preprocessing
        waveform = F.layer_norm(waveform, waveform.shape)
        return waveform

    def extract_speaker_embedding(self, wav_directory: str) -> torch.Tensor:
        """
        Extract speaker embedding using YourTTS's built-in speaker encoder
        """
        wav_files = list(Path(wav_directory).glob("*.wav"))
        if not wav_files:
            raise ValueError(f"No WAV files found in {wav_directory}")

        embeddings = []
        
        for wav_path in wav_files:
            waveform, sample_rate = torchaudio.load(wav_path)
            waveform = self._preprocess_audio(waveform, sample_rate)
            
            # Convert to numpy for Coqui TTS
            wav = waveform.squeeze().numpy()
            
            # Extract embedding using YourTTS's speaker encoder
            speaker_embedding = self.model.speaker_manager.compute_embedding(wav)
            embeddings.append(speaker_embedding)

        # Average the embeddings
        average_embedding = np.mean(embeddings, axis=0)
        return average_embedding

    def synthesize_speech(
        self, 
        text: str, 
        speaker_embedding: np.ndarray, 
        output_path: str,
        language: str = "en",
        duration_scale: float = 1.0,
        energy_scale: float = 1.0,
        f0_scale: float = 1.0
    ):
        """
        Synthesize speech with YourTTS
        """
        try:
            # Generate speech
            wav = self.synthesizer.tts(
                text,
                speaker_embedding=speaker_embedding,
                language=language,
                duration_scale=duration_scale,
                energy_scale=energy_scale,
                f0_scale=f0_scale
            )

            # Convert to torch tensor for post-processing
            wav_tensor = torch.FloatTensor(wav)
            
            # Post-process audio
            wav_tensor = self._postprocess_audio(wav_tensor)

            # Save with high-quality settings
            sf.write(
                output_path, 
                wav_tensor.numpy(), 
                samplerate=self.config.audio['sample_rate'],
                subtype='PCM_24'  # 24-bit for better quality
            )
            
        except Exception as e:
            logging.error(f"Error in speech synthesis: {str(e)}")
            raise

    def _postprocess_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply post-processing for better audio quality"""
        # Normalize
        waveform = waveform / torch.max(torch.abs(waveform))
        
        # Apply subtle compression
        threshold = 0.95
        ratio = 0.2
        mask = torch.abs(waveform) > threshold
        waveform[mask] = threshold + (torch.abs(waveform[mask]) - threshold) * ratio * torch.sign(waveform[mask])
        
        return waveform

    def clone_and_speak(
        self,
        voice_samples_dir: str,
        text: str,
        output_path: str,
        language: str = "en",
        prosody_control: dict = None
    ):
        """
        Main function with enhanced control and error handling
        """
        try:
            logging.info("Extracting speaker embedding...")
            speaker_embedding = self.extract_speaker_embedding(voice_samples_dir)

            logging.info("Generating speech...")
            prosody_params = prosody_control or {
                "duration_scale": 1.0,
                "energy_scale": 1.0,
                "f0_scale": 1.0
            }
            
            self.synthesize_speech(
                text,
                speaker_embedding,
                output_path,
                language=language,
                **prosody_params
            )
            
            logging.info(f"Speech generated successfully: {output_path}")
            
        except Exception as e:
            logging.error(f"Error in voice cloning process: {str(e)}")
            raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the enhanced voice cloner
    cloner = EnhancedVoiceCloner()
    
    # Example usage with prosody control
    voice_samples_dir = "voice_samples"
    text = "Hello, this is a test of enhanced voice cloning using YourTTS."
    output_path = "enhanced_speech.wav"
    
    prosody_control = {
        "duration_scale": 1.1,  # Slightly slower
        "energy_scale": 1.0,    # Normal energy
        "f0_scale": 1.0         # Normal pitch
    }
    
    # Clone voice and generate speech
    cloner.clone_and_speak(
        voice_samples_dir,
        text,
        output_path,
        language="en",
        prosody_control=prosody_control
    )