import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer

# Initialize model
model = Model("/Users/theboii/Downloads/vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, flush=True)
    audio_queue.put(bytes(indata))

def main():
    print("Listening... say 'i need help' to trigger response.")
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip().lower()
                if text:
                    print("Heard:", text)
                    if "i need help" in text:
                        print("Help is on the way")
                        break

if __name__ == "__main__":
    main()