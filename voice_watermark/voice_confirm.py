import torch
import torchaudio
import numpy as np
from pathlib import Path
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, Union
import warnings
import yaml
from functools import lru_cache

class TTSFramework:
    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32
    ):
        """
        Initialize the TTS Framework with safety measures and logging.
        
        Args:
            model_path: Path to the StyleTTS2 model checkpoint
            config_path: Path to configuration file
            device: Device to run inference on
            batch_size: Batch size for processing multiple inputs
        """
        self.device = device
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.batch_size = batch_size
        
        # Load configuration
        self.config = self._load_config()
            
        # Initialize model
        self.model = self._load_model()
        self.sample_rate = self.config.get("audio", {}).get("sampling_rate", 22050)
        
        # Setup logging and tracking
        self.log_dir = Path("tts_logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize watermarking
        self._init_watermark()

    @lru_cache(maxsize=1)
    def _load_config(self) -> dict:
        """Cache config loading for better performance."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _init_watermark(self) -> None:
        """Pre-compute watermark key for reuse."""
        self.watermark_key = hashlib.sha256(b"TTS_WATERMARK").digest()
        self.watermark_seed = int.from_bytes(self.watermark_key[:4], 'big')
        
    @torch.no_grad()  # Explicitly disable gradients
    def _load_model(self) -> torch.nn.Module:
        """Load and initialize the StyleTTS2 model with safety checks."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
            
        # Load model architecture and weights
        model = torch.load(self.model_path, map_location=self.device)
        model.eval()  # Set to evaluation mode
        
        # Enable model optimization
        if self.device == "cuda":
            model = model.half()  # Use FP16 for faster inference
            torch.cuda.empty_cache()  # Clear GPU memory
            
        return model
    
    def _add_watermark(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Add an inaudible watermark to the generated audio.
        
        Args:
            audio: Generated audio tensor
            
        Returns:
            Watermarked audio tensor
        """
        # Create pseudo-random sequence using pre-computed seed
        rng = np.random.RandomState(seed=self.watermark_seed)
        watermark = torch.from_numpy(
            rng.uniform(-0.0001, 0.0001, size=audio.shape)
        ).to(audio.device, dtype=audio.dtype)  # Match audio dtype
        
        return audio.add_(watermark)  # In-place addition
    
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
        
        # Use daily log files for better I/O performance
        log_file = self.log_dir / f"generation_log_{timestamp[:10]}.jsonl"
        with open(log_file, "a", buffering=8192) as f:  # Add buffering
            f.write(json.dumps(log_entry) + "\n")
    
    @torch.no_grad()  # Disable gradients for inference
    def generate_speech(
        self,
        text: Union[str, list],
        output_path: Optional[str] = None,
        speaker_id: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate speech from text with safety measures.
        
        Args:
            text: Input text to synthesize (str or list for batch processing)
            output_path: Optional path to save audio file
            speaker_id: Speaker identity for multi-speaker models
            metadata: Optional metadata for logging
            
        Returns:
            Path to generated audio file
        """
        if not text:
            raise ValueError("Empty text provided")
            
        if isinstance(text, str):
            if len(text) > self.config.get("max_text_length", 1000):
                raise ValueError("Text exceeds maximum allowed length")
            texts = [text]
        else:
            texts = text
        
        # Set default output path if not provided
        if output_path is None:
            output_path = f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        output_path = Path(output_path)
        
        # Process in batches for better efficiency
        audio_chunks = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Generate mel-spectrogram
            mel_outputs = self.model.generate(
                batch,
                speaker_id=speaker_id
            )
            
            # Convert mel-spectrogram to audio
            audio = self.model.vocoder(mel_outputs)
            audio_chunks.append(audio)
            
        # Concatenate audio chunks
        audio = torch.cat(audio_chunks, dim=0)
        
        # Add watermark
        audio = self._add_watermark(audio)
        
        # Save audio with optimized settings
        torchaudio.save(
            output_path,
            audio.cpu(),
            self.sample_rate,
            encoding="PCM_S",
            bits_per_sample=16
        )
        
        # Log generation
        self._log_generation(
            text=text,
            metadata=metadata or {},
            output_path=output_path
        )
        
        return output_path
    
    @torch.no_grad()  # Disable gradients for verification
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
            
        # Extract watermark using pre-computed seed
        rng = np.random.RandomState(seed=self.watermark_seed)
        watermark = torch.from_numpy(
            rng.uniform(-0.0001, 0.0001, size=audio.shape)
        ).to(audio.device, dtype=audio.dtype)
        
        # Optimize correlation calculation
        audio_flat = audio.flatten()
        watermark_flat = watermark.flatten()
        correlation = torch.corrcoef(
            torch.stack([audio_flat, watermark_flat])
        )[0, 1]
        
        return correlation > 0.75  # Threshold for watermark detection

# Example usage
if __name__ == "__main__":
    # Load configuration
    config_path = "voice_watermark/config.yaml"
    model_path = "voice_watermark/epoch_2nd_00100.pth"
    
    # Initialize framework with batch processing
    tts = TTSFramework(
        model_path=model_path,
        config_path=config_path,
        batch_size=32
    )
    
    # Generate speech with metadata
    output_path = tts.generate_speech(
        text=["This is a test of the speech synthesis framework."] * 5,  # Batch example
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