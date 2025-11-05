import json
import os
import wave
import subprocess
from vosk import Model, KaldiRecognizer
from rapidfuzz import fuzz

# ----------------------------
# CONFIGURATION
# ----------------------------
trigger_phrases = ["i need help"]

my_words = [
    "be", "to", "of", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "all", "would", "there", "their", "what", "so",
    "up", "out", "if", "about", "who", "get", "which", "go", "me", "when",
    "list", "hand", "milk",
    "i", "need", "help", "dinosaur", "hell"
]

model_path = "/Users/theboii/Downloads/vosk-model-small-en-us-0.15"
test_files = ["test1.wav", "test2.wav"]  # Add more here
output_file = "batch_results.txt"

# ----------------------------
# INITIALIZE MODEL
# ----------------------------
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000, json.dumps(my_words))

# ----------------------------
# CONVERT AUDIO IF NEEDED
# ----------------------------
def ensure_pcm16(file_path):
    """
    Converts any audio file to 16 kHz, mono, 16-bit PCM WAV.
    Returns path to the converted file.
    """
    base, _ = os.path.splitext(file_path)
    fixed_path = base + "_converted.wav"

    # Use ffmpeg to standardize the format
    cmd = [
        "ffmpeg", "-y", "-i", file_path,
        "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", fixed_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return fixed_path

# ----------------------------
# PROCESS FILE FUNCTION
# ----------------------------
def process_audio(file_path):
    wf = wave.open(file_path, "rb")

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        raise ValueError(f"Audio file {file_path} must be 16kHz, mono, 16-bit PCM WAV.")

    recognizer.Reset()
    detected = False

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip().lower()
            if text:
                for phrase in trigger_phrases:
                    score = fuzz.ratio(text, phrase)
                    if score > 90:
                        detected = True
                        break
        else:
            _ = recognizer.PartialResult()

    # Check final result
    final = json.loads(recognizer.FinalResult())
    text = final.get("text", "").strip().lower()
    if not detected and text:
        for phrase in trigger_phrases:
            score = fuzz.ratio(text, phrase)
            if score > 90:
                detected = True
                break

    wf.close()
    return detected

# ----------------------------
# MAIN LOOP
# ----------------------------
with open(output_file, "w") as f:
    f.write("Sample Name | Detected (Y/N)\n")
    f.write("--------------------------------\n")
    for file_name in test_files:
        print(f"Processing {file_name}...")
        try:
            clean_file = ensure_pcm16(file_name)
            detected = process_audio(clean_file)
            f.write(f"{file_name} | {'Y' if detected else 'N'}\n")
            print(f" → {'Detected ✅' if detected else 'Not detected ❌'}")
        except Exception as e:
            f.write(f"{file_name} | ERROR: {e}\n")
            print(f"Error with {file_name}: {e}")

print(f"\n✅ Testing complete. Results saved to {output_file}")