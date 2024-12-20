import torch
import torchaudio
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt, medfilt
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import StandardScaler
import librosa
import pyloudnorm as pyln

nltk.download('punkt')
nltk.download('punkt_tab')

class EmbeddingProjection(torch.nn.Module):
    """Projects speaker embeddings from ECAPA-TDNN dimension to SpeechT5 dimension"""
    def __init__(self, input_dim=960, output_dim=1280):
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, output_dim)
        
        # Initialize weights to preserve embedding structure
        torch.nn.init.orthogonal_(self.projection.weight)
        torch.nn.init.zeros_(self.projection.bias)
    
    def forward(self, x):
        return self.projection(x)

class EnhancedVoiceCloner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use the larger, more accurate model variant
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        # Use a more sophisticated speaker encoder
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",  # Updated to ECAPA-TDNN
            run_opts={"device": self.device}
        )
        
        # Initialize embedding projection layer
        self.embedding_projection = EmbeddingProjection().to(self.device)

        self.tts_model = self.tts_model.to(self.device)
        self.vocoder = self.vocoder.to(self.device)
        
        # Improved audio processing parameters
        self.sample_rate = 22050  # Increased for better quality
        self.chunk_size = 16384  # Increased for better context
        self.overlap = 1024
        # Initialize loudness meter
        self.meter = pyln.Meter(self.sample_rate)

    def preprocess_training_audio(self, waveform, sample_rate):
        """Enhanced preprocessing for training audio samples"""
        # Resample if necessary
        if sample_rate != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        # Convert to mono if needed
        if len(waveform.shape) > 1:
            waveform = librosa.to_mono(waveform)
            
        # Remove silence and normalize
        waveform = self._trim_silence(waveform)
        waveform = self._normalize_audio(waveform)
        
        # Apply subtle denoising
        waveform = self._denoise_audio(waveform)
        
        return waveform

    def _trim_silence(self, audio, top_db=30):
        """Remove silence from audio using librosa"""
        return librosa.effects.trim(audio, top_db=top_db)[0]

    def _normalize_audio(self, audio):
        """Improved audio normalization with loudness targeting"""
        # Measure the loudness of the audio
        loudness = self.meter.integrated_loudness(audio)
        
        # Normalize to target loudness (-23 LUFS is standard for broadcast)
        normalized_audio = pyln.normalize.loudness(audio, loudness, -23.0)
        
        # Apply peak normalization
        normalized_audio = librosa.util.normalize(normalized_audio)
        
        return normalized_audio

    def _denoise_audio(self, audio):
        """Apply sophisticated denoising"""
        # Median filtering to remove impulse noise
        audio = medfilt(audio, kernel_size=3)
        
        # Spectral subtraction
        S = librosa.stft(audio)
        mag = np.abs(S)
        phase = np.angle(S)
        
        # Estimate noise floor
        noise_floor = np.mean(np.min(mag, axis=1))
        mag = np.maximum(mag - noise_floor, 0)
        
        # Reconstruct signal
        S_cleaned = mag * np.exp(1j * phase)
        audio = librosa.istft(S_cleaned)
        
        return audio

    def enhance_prosody(self, audio):
        """Enhance speech prosody"""
        # Extract pitch contour
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        
        # Smooth pitch contour
        f0_cleaned = medfilt(f0[voiced_flag], kernel_size=5)
        
        # Apply pitch correction
        return self._apply_pitch_correction(audio, f0_cleaned, voiced_flag)

    def _apply_pitch_correction(self, audio, f0, voiced_flag):
        """Apply subtle pitch correction to improve naturalness"""
        # Implementation of pitch correction using f0 contour
        correction_strength = 0.3  # Subtle correction
        
        y_shifted = librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=correction_strength * np.mean(f0[voiced_flag])
        )
        
        return y_shifted

    def extract_speaker_embedding(self, wav_directory):
        """Enhanced speaker embedding extraction with dimension projection"""
        wav_files = list(Path(wav_directory).glob("*.wav"))
        if not wav_files:
            raise ValueError(f"No WAV files found in {wav_directory}")

        embeddings = []
        scaler = StandardScaler()
        
        for wav_path in wav_files:
            # Load and preprocess audio
            waveform, sample_rate = librosa.load(wav_path, sr=None)
            waveform = self.preprocess_training_audio(waveform, sample_rate)
            waveform = torch.FloatTensor(waveform).unsqueeze(0)
            
            # Extract embedding with augmentation
            with torch.no_grad():
                # Original embedding
                embedding = self.speaker_encoder.encode_batch(waveform)
                embeddings.append(embedding.cpu().numpy())
                
                # Pitch-shifted augmentation
                waveform_shifted = librosa.effects.pitch_shift(
                    waveform.numpy().squeeze(),
                    sr=self.sample_rate,
                    n_steps=0.5
                )
                waveform_shifted = torch.FloatTensor(waveform_shifted).unsqueeze(0)
                embedding_shifted = self.speaker_encoder.encode_batch(waveform_shifted)
                embeddings.append(embedding_shifted.cpu().numpy())

        # Normalize embeddings
        embeddings = np.vstack(embeddings)
        embeddings_normalized = scaler.fit_transform(embeddings.squeeze())
        
        # Calculate weighted average
        weights = np.ones(len(embeddings_normalized))
        weights[:len(wav_files)] = 1.2  # Give more weight to original embeddings
        average_embedding = np.average(embeddings_normalized, weights=weights, axis=0)
        
        # Convert to tensor and project to correct dimension
        embedding_tensor = torch.FloatTensor(average_embedding).reshape(1, -1).to(self.device)
        projected_embedding = self.embedding_projection(embedding_tensor)
        
        return projected_embedding

    def synthesize_speech(self, text, speaker_embedding, output_path):
        """Speech synthesis with projected embeddings"""
        #text = self.add_natural_pauses(text)
        sentences = sent_tokenize(text)
        audio_segments = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            sentence = self._add_prosody_markers(sentence)
            inputs = self.processor(text=sentence, return_tensors="pt")
            
            # Speaker embedding is already projected to correct dimension
            with torch.no_grad():
                speech = self.tts_model.generate_speech(
                    inputs["input_ids"].to(self.device),
                    speaker_embeddings=speaker_embedding,  # Already projected
                    vocoder=self.vocoder,
                    #temperature=0.7
                )
                audio = speech.cpu().numpy()
                
                audio = self.process_audio(audio)
                audio = self.enhance_prosody(audio)
                audio_segments.append(audio)
                
                silence_duration = min(0.3, len(sentence) * 0.02)
                silence = np.zeros(int(silence_duration * self.sample_rate))
                audio_segments.append(silence)

        final_speech = np.concatenate(audio_segments)
        final_speech = self._finalize_audio(final_speech)
        
        sf.write(output_path, final_speech, samplerate=self.sample_rate)

    def _add_prosody_markers(self, text):
        """Add markers for better prosody control"""
        # Add emphasis markers for important words
        words = text.split()
        processed_words = []
        
        for word in words:
            if word.istitle() or word.isupper():
                word = f"<emphasis level='strong'>{word}</emphasis>"
            processed_words.append(word)
        
        return " ".join(processed_words)

    def _finalize_audio(self, audio):
        """Final processing steps for the complete audio"""
        # Apply subtle compression
        audio = self._apply_compression(audio)
        
        # Enhance clarity
        audio = self._enhance_clarity(audio)
        
        # Final normalization
        audio = self._normalize_audio(audio)
        
        return audio

    def _apply_compression(self, audio, threshold=-20, ratio=4):
        """Apply subtle compression to improve dynamics"""
        # Convert to dB
        db = 20 * np.log10(np.abs(audio) + 1e-8)
        
        # Apply compression
        mask = db > threshold
        db[mask] = threshold + (db[mask] - threshold) / ratio
        
        # Convert back to linear
        return np.sign(audio) * (10 ** (db / 20))

    def _enhance_clarity(self, audio):
        """Enhance speech clarity"""
        # Apply subtle high-shelf boost
        y = librosa.effects.preemphasis(audio, coef=0.97)
        
        # Enhance formants
        S = librosa.stft(y)
        mag = np.abs(S)
        phase = np.angle(S)
        
        # Spectral sharpening
        mag = mag ** 1.2
        
        # Reconstruct
        S_enhanced = mag * np.exp(1j * phase)
        y_enhanced = librosa.istft(S_enhanced)
        
        return y_enhanced

    def clone_and_speak(self, voice_samples_dir, text, output_path):
        """Main function with improved error handling and logging"""
        try:
            print("Extracting and processing speaker embedding...")
            speaker_embedding = self.extract_speaker_embedding(voice_samples_dir)
            
            print("Generating enhanced speech...")
            self.synthesize_speech(text, speaker_embedding, output_path)
            print(f"Enhanced speech generated and saved to {output_path}")
            
        except Exception as e:
            print(f"Error during voice cloning: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    cloner = EnhancedVoiceCloner()
    
    voice_samples_dir = "voice_samples"
    text = "Hello, this is a test of enhanced voice cloning. Notice the improved naturalness and clarity."
    output_path = "enhanced_speech_v2.wav"
    
    cloner.clone_and_speak(voice_samples_dir, text, output_path)