import torch
import torchaudio
import numpy as np
from pathlib import Path
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
import warnings
import yaml

class TTSFramework:
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the TTS Framework with safety measures and logging.
        
        Args:
            model_path: Path to the StyleTTS2 model checkpoint
            config_path: Path to configuration file
            device: Device to run inference on
        """
        self.device = device
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Initialize model
        self.model = self._load_model()
        self.sample_rate = self.config.get("audio", {}).get("sampling_rate", 22050)
        
        # Setup logging and tracking
        self.log_dir = Path("tts_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize watermarking
        self.watermark_key = hashlib.sha256(b"TTS_WATERMARK").digest()
        
    def _load_model(self) -> torch.nn.Module:
        """Load and initialize the StyleTTS2 model with safety checks."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
            
        # Load model architecture and weights
        model = torch.load(self.model_path, map_location=self.device)
        model.eval()
        return model
    
    def _add_watermark(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Add an inaudible watermark to the generated audio.
        
        Args:
            audio: Generated audio tensor
            
        Returns:
            Watermarked audio tensor
        """
        # Create pseudo-random sequence using the watermark key
        rng = np.random.RandomState(seed=int.from_bytes(self.watermark_key[:4], 'big'))
        watermark = torch.from_numpy(
            rng.uniform(-0.0001, 0.0001, size=audio.shape)
        ).to(audio.device)
        
        return audio + watermark
    
    def _log_generation(
        self,
        text: str,
        metadata: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Log details of audio generation for tracking and auditing."""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "text": text,
            "output_path": str(output_path),
            "metadata": metadata
        }
        
        log_file = self.log_dir / f"generation_log_{timestamp[:10]}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def generate_speech(
        self,
        text: str,
        output_path: Optional[str] = None,
        speaker_id: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate speech from text with safety measures.
        
        Args:
            text: Input text to synthesize
            output_path: Optional path to save audio file
            speaker_id: Speaker identity for multi-speaker models
            metadata: Optional metadata for logging
            
        Returns:
            Path to generated audio file
        """
        if not text:
            raise ValueError("Empty text provided")
            
        if len(text) > self.config.get("max_text_length", 1000):
            raise ValueError("Text exceeds maximum allowed length")
        
        # Set default output path if not provided
        if output_path is None:
            output_path = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        output_path = Path(output_path)
        
        # Generate mel-spectrogram
        with torch.no_grad():
            mel_output = self.model.generate(
                text,
                speaker_id=speaker_id
            )
            
            # Convert mel-spectrogram to audio
            audio = self.model.vocoder(mel_output)
            
            # Add watermark
            audio = self._add_watermark(audio)
            
            # Save audio
            torchaudio.save(
                output_path,
                audio.cpu(),
                self.sample_rate
            )
        
        # Log generation
        self._log_generation(
            text=text,
            metadata=metadata or {},
            output_path=output_path
        )
        
        return output_path
    
    def verify_watermark(self, audio_path: str) -> bool:
        """
        Verify if audio file contains the framework's watermark.
        
        Args:
            audio_path: Path to audio file to verify
            
        Returns:
            Boolean indicating if watermark is present
        """
        audio, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            warnings.warn(f"Audio sample rate {sr} differs from model rate {self.sample_rate}")
            
        # Extract watermark
        rng = np.random.RandomState(seed=int.from_bytes(self.watermark_key[:4], 'big'))
        watermark = torch.from_numpy(
            rng.uniform(-0.0001, 0.0001, size=audio.shape)
        ).to(audio.device)
        
        # Compare correlation of the audio and watermark
        correlation = torch.corrcoef(
            torch.stack([audio.flatten(), watermark.flatten()])
        )[0, 1]
        
        return correlation > 0.75  # Threshold for watermark detection

# Example usage
if __name__ == "__main__":
    # Load configuration
    config_path = "voice_watermark/config.yaml"
    model_path = "voice_watermark/epoch_2nd_00100.pth"
    
    # Initialize framework
    tts = TTSFramework(
        model_path=model_path,
        config_path=config_path
    )
    
    # Generate speech with metadata
    output_path = tts.generate_speech(
        text="This is a test of the speech synthesis framework.",
        metadata={
            "purpose": "testing",
            "user": "researcher_id",
            "project": "speech_synthesis_research"
        }
    )
    
    print(f"Generated audio saved to: {output_path}")
    
    # Verify watermark
    is_valid = tts.verify_watermark(output_path)
    print(f"Watermark verification: {is_valid}")