import torch
import torchaudio
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from speechbrain.pretrained import EncoderClassifier
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')

class EnhancedVoiceCloner:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            run_opts={"device": self.device}
        )
        
        # Move models to device
        self.tts_model = self.tts_model.to(self.device)
        self.vocoder = self.vocoder.to(self.device)
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.chunk_size = 8192
        self.overlap = 512

    def butter_lowpass(self, cutoff, fs, order=5):
        """Design a lowpass filter"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def apply_lowpass_filter(self, data, cutoff, fs, order=5):
        """Apply lowpass filter to smooth audio"""
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def smooth_transitions(self, audio):
        """Apply fade in/out to reduce abrupt transitions"""
        fade_length = 128
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        
        audio[:fade_length] *= fade_in
        audio[-fade_length:] *= fade_out
        return audio

    def process_audio(self, audio):
        """Apply various audio processing techniques"""
        # Convert to float32 if not already
        audio = audio.astype(np.float32)
        
        # Apply lowpass filter to smooth high frequencies
        audio = self.apply_lowpass_filter(audio, cutoff=7000, fs=self.sample_rate)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Apply subtle compression
        threshold = 0.3
        ratio = 0.7
        audio = np.where(np.abs(audio) > threshold,
                        threshold + (np.abs(audio) - threshold) * ratio,
                        audio)
        
        # Smooth transitions
        audio = self.smooth_transitions(audio)
        
        return audio

    def extract_speaker_embedding(self, wav_directory):
        """Extract and process speaker embeddings"""
        wav_files = list(Path(wav_directory).glob("*.wav"))
        if not wav_files:
            raise ValueError(f"No WAV files found in {wav_directory}")

        embeddings = []
        
        for wav_path in wav_files:
            waveform, sample_rate = torchaudio.load(wav_path)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Convert to mono and normalize
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform / torch.max(torch.abs(waveform))
            
            with torch.no_grad():
                embedding = self.speaker_encoder.encode_batch(waveform)
                embedding = embedding.squeeze()
                embeddings.append(embedding.cpu())
        
        # Average embeddings
        average_embedding = torch.mean(torch.stack(embeddings), dim=0).reshape(1, -1)
        return average_embedding

    def add_natural_pauses(self, text):
        """Add natural pauses and pacing to text"""
        sentences = sent_tokenize(text)
        processed_text = []
        
        for sentence in sentences:
            # Add slight pause after sentence
            processed_text.append(sentence + ".")
        
        return " ".join(processed_text)

    def synthesize_speech(self, text, speaker_embedding, output_path):
        """Synthesize speech with enhanced processing"""
        # Add natural pauses
        text = self.add_natural_pauses(text)
        
        # Split into sentences for better processing
        sentences = sent_tokenize(text)
        audio_segments = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Add pause markers for more natural speech
            sentence = sentence.strip() + '.'
            inputs = self.processor(text=sentence, return_tensors="pt")
            
            with torch.no_grad():
                speech = self.tts_model.generate_speech(
                    inputs["input_ids"].to(self.device),
                    speaker_embeddings=speaker_embedding.to(self.device),
                    vocoder=self.vocoder
                )
                audio = speech.cpu().numpy()
                
                # Process audio segment
                audio = self.process_audio(audio)
                audio_segments.append(audio)
                
                # Add small silence between sentences
                silence = np.zeros(int(0.2 * self.sample_rate))
                audio_segments.append(silence)

        # Concatenate all segments
        final_speech = np.concatenate(audio_segments)
        
        # Final processing on complete audio
        final_speech = self.process_audio(final_speech)
        
        # Save the generated speech
        sf.write(output_path, final_speech, samplerate=self.sample_rate)

    def clone_and_speak(self, voice_samples_dir, text, output_path):
        """Main function for voice cloning and speech synthesis"""
        try:
            print("Extracting speaker embedding...")
            speaker_embedding = self.extract_speaker_embedding(voice_samples_dir)
            
            print("Generating speech...")
            self.synthesize_speech(text, speaker_embedding, output_path)
            print(f"Speech generated and saved to {output_path}")
            
        except Exception as e:
            print(f"Error during voice cloning: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    cloner = EnhancedVoiceCloner()
    
    voice_samples_dir = "voice_samples"
    text = "Hello, this is a test of voice cloning using SpeechT5 with X-vector embeddings."
    output_path = "enhanced_speech.wav"
    
    cloner.clone_and_speak(voice_samples_dir, text, output_path)