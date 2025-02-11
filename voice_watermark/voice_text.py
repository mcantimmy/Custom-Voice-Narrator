import speech_recognition as sr
import pyaudio
import wave
import time

def record_audio(filename, duration=5):
    """
    Record audio from microphone
    :param filename: Output filename
    :param duration: Recording duration in seconds
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)

    print("* recording")

    frames = []
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def speech_to_text(audio_file=None):
    """
    Convert speech to text
    :param audio_file: Audio file to transcribe (if None, uses microphone)
    :return: Transcribed text
    """
    recognizer = sr.Recognizer()

    if audio_file is None:
        # Use microphone as source
        with sr.Microphone() as source:
            print("Say something!")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, 
                                      #timeout=20.0,           # Maximum time to wait for phrase to start (seconds)
                                      #phrase_time_limit=10.0  # Maximum time for a phrase (seconds)
                                      )                     
    else:
        # Use audio file as source
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)

    try:
        # Using Google's speech recognition
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

def main():
    # Option 1: Record and transcribe
    #audio_file = "output.wav"
    #record_audio(audio_file, duration=5)
    #return speech_to_text(audio_file)

    # Option 2: Real-time transcription
    return speech_to_text()

if __name__ == "__main__":
    main()