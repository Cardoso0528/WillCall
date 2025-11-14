import json
import os
import wave
import subprocess
from vosk import Model, KaldiRecognizer
from rapidfuzz import fuzz
import noisereduce as nr
import soundfile as sf
import numpy as np

# ----------------------------
# CONFIGURATION
# ----------------------------
TRIGGER_PHRASE = "i need help"

## GRAMMAR CONFIGURATION ##
# Phrase-level grammar: Only accepts exact wake phrase
# This provides the strictest wake word detection with minimal false positives
GRAMMAR_MODE = "phrase"  # "phrase" for strict, "word" for flexible

# Phrase-level grammar (strictest - only exact phrase)
wake_phrases = [
    "i need help"
]

# Word-level grammar (flexible - for fallback/testing)
wake_words = [
    # Core trigger phrase words
    "i", "need", "help",
    
    # Common confusion words (helps model recognize but validation still rejects)
    "on"
    "he", "me", "we",  # can sound like "i"
    "kneed", "neat",  # can sound like "need"
    "held", "hell", "milk" # can sound like "help"
]

# Select grammar based on mode
if GRAMMAR_MODE == "phrase":
    my_words = wake_phrases
    print(f" Using PHRASE-LEVEL grammar: Only exact phrase '{TRIGGER_PHRASE}' will be detected")
else:
    my_words = wake_words
    print(f" Using WORD-LEVEL grammar: Flexible matching with validation")

model_path = "/Users/theboii/Downloads/vosk-model-small-en-us-0.15"
test_files = os.listdir("Test")
test_files = [f for f in test_files if not f.endswith("_converted.wav") and not f.startswith(".")]
test_files = [os.path.join("Test", file) for file in test_files]
output_file = "batch_results.txt"

# Noise reduction settings
ENABLE_NOISE_REDUCTION = True
NOISE_REDUCTION_STRENGTH = 0.6

# Recognition settings
FUZZY_MATCH_THRESHOLD = 85
ENABLE_WORD_LEVEL_MATCHING = True
ENABLE_PARTIAL_MATCHING = False
ENABLE_AUDIO_NORMALIZATION = True

# Safety settings for live wake word detection
REQUIRE_I_PREFIX = True
ALLOW_HELL_SUBSTITUTE = False
WORD_LEVEL_THRESHOLD = 85
PARTIAL_MATCH_THRESHOLD = 90
MIN_PHRASE_WORDS = 3
MIN_TEXT_LENGTH = 2

# Ensemble settings
ENABLE_ENSEMBLE = True
ENSEMBLE_VOTING_THRESHOLD = 0.6

# ----------------------------
# INITIALIZE MODEL
# ----------------------------
model = Model(model_path)
recognizer = KaldiRecognizer(model, 16000, json.dumps(my_words))

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def normalize_audio(audio_data):
    """Normalize audio to prevent clipping while maximizing volume."""
    if ENABLE_AUDIO_NORMALIZATION:
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            return audio_data / max_val * 0.95
    return audio_data

def convert_audio_format(file_path, output_path):
    """Convert audio to 16kHz, mono, 16-bit PCM WAV."""
    cmd = ["ffmpeg", "-y", "-i", file_path, "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", output_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def check_word_level_match(words, phrase):
    """Check if words match the trigger phrase using word-level matching."""
    words_lower = [w.lower() for w in words]
    has_i = "i" in words_lower
    has_need = "need" in words_lower
    has_help = "help" in words_lower
    has_hell = "hell" in words_lower
    
    # Reject "hell" if not allowed
    if not ALLOW_HELL_SUBSTITUTE and has_hell and not has_help:
        return False, 0
    
    if REQUIRE_I_PREFIX:
        if not (has_i and has_need and has_help):
            return False, 0
        
        i_idx = words_lower.index("i")
        need_idx = words_lower.index("need")
        help_idx = words_lower.index("help")
        
        # Validate order and proximity (max 1 word between each)
        if need_idx <= i_idx or help_idx <= need_idx:
            return False, 0
        if (need_idx - i_idx > 1) or (help_idx - need_idx > 1):
            return False, 0
    else:
        if not (has_need and has_help):
            return False, 0
        
        need_idx = words_lower.index("need")
        help_idx = words_lower.index("help")
        if help_idx - need_idx > 1:
            return False, 0
    
    full_text = " ".join(words)
    score = fuzz.ratio(full_text, phrase)
    return score >= WORD_LEVEL_THRESHOLD, score

def process_text_segment(text, phrase, threshold, best_match_score, best_match_text):
    """Process a text segment for phrase matching."""
    detected = False
    
    # Exact phrase match
    if phrase in text:
        return True, 100.0, phrase
    
    # Full ratio match
    score = fuzz.ratio(text, phrase)
    if score > best_match_score:
        best_match_score = score
        best_match_text = text
    if score >= threshold:
        detected = True
    
    # Partial matching (if enabled)
    if not detected and ENABLE_PARTIAL_MATCHING:
        partial_score = fuzz.partial_ratio(text, phrase)
        if partial_score > best_match_score:
            best_match_score = partial_score
            best_match_text = text
        if partial_score >= PARTIAL_MATCH_THRESHOLD:
            detected = True
    
    return detected, best_match_score, best_match_text

def recognize_audio(wf, phrase, threshold):
    """Recognize speech from audio file and return detected status, text, and scores."""
    recognizer.Reset()
    detected = False
    all_text = []
    best_match_score = 0
    best_match_text = ""
    
    # Process audio chunks
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip().lower()
            if text:
                all_text.append(text)
                detected_seg, best_match_score, best_match_text = process_text_segment(
                    text, phrase, threshold, best_match_score, best_match_text
                )
                if detected_seg:
                    detected = True
        else:
            _ = recognizer.PartialResult()
    
    # Check final result
    final = json.loads(recognizer.FinalResult())
    text = final.get("text", "").strip().lower()
    if text:
        all_text.append(text)
        detected_seg, best_match_score, best_match_text = process_text_segment(
            text, phrase, threshold, best_match_score, best_match_text
        )
        if detected_seg:
            detected = True
    
    full_text = " ".join(all_text) if all_text else "[no speech detected]"
    return detected, full_text, best_match_score, best_match_text

def validate_detection(full_text, best_match_score, detected, phrase=TRIGGER_PHRASE):
    """Final validation to prevent false positives in live safety wake word system."""
    # Reject if no speech detected or detection not triggered
    if full_text == "[no speech detected]" or not detected:
        return False
    
    words = full_text.split()
    
    # Reject single-word matches
    if len(words) < MIN_TEXT_LENGTH:
        return False
    
    single_word_false_positives = ["help", "need", "hell", "he", "i"]
    if len(words) == 1 and words[0] in single_word_false_positives:
        return False
    
    # Accept exact phrase match
    if phrase in full_text:
        return True
    
    # For non-exact matches, require high score and key words
    if best_match_score < FUZZY_MATCH_THRESHOLD:
        return False
    
    # Check word-level matching
    if ENABLE_WORD_LEVEL_MATCHING:
        matches, score = check_word_level_match(words, phrase)
        if matches:
            return True
    
    return False

# ----------------------------
# AUDIO PROCESSING
# ----------------------------
def reduce_noise(file_path, output_folder="Converted", strength=None):
    """Apply noise reduction to audio file."""
    if strength is None:
        strength = NOISE_REDUCTION_STRENGTH
    
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.basename(file_path)
    base, _ = os.path.splitext(filename)
    noise_reduced_path = os.path.join(output_folder, base + "_noisereduced.wav")
    
    audio_data, sample_rate = sf.read(file_path)
    reduced_noise = nr.reduce_noise(
        y=audio_data, sr=sample_rate, stationary=True,
        prop_decrease=strength, freq_mask_smooth_hz=500, time_mask_smooth_ms=50
    )
    reduced_noise = normalize_audio(reduced_noise)
    sf.write(noise_reduced_path, reduced_noise, sample_rate)
    
    return noise_reduced_path

def ensure_pcm16(file_path, output_folder="Converted", apply_noise_reduction=True):
    """Convert audio to 16kHz, mono, 16-bit PCM WAV."""
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.basename(file_path)
    base, _ = os.path.splitext(filename)
    fixed_path = os.path.join(output_folder, base + "_converted.wav")
    
    if apply_noise_reduction:
        temp_path = os.path.join(output_folder, base + "_temp.wav")
        convert_audio_format(file_path, temp_path)
        noise_reduced_path = reduce_noise(temp_path, output_folder)
        os.rename(noise_reduced_path, fixed_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
    else:
        convert_audio_format(file_path, fixed_path)
    
    return fixed_path

# ----------------------------
# RECOGNITION FUNCTIONS
# ----------------------------
def process_audio(file_path):
    """Process audio file and detect trigger phrase."""
    wf = wave.open(file_path, "rb")
    
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        wf.close()
        raise ValueError(f"Audio file {file_path} must be 16kHz, mono, 16-bit PCM WAV.")
    
    detected, full_text, best_match_score, best_match_text = recognize_audio(
        wf, TRIGGER_PHRASE, FUZZY_MATCH_THRESHOLD
    )
    wf.close()
    
    # Enhanced matching on full text
    words = full_text.split()
    if len(words) >= MIN_TEXT_LENGTH and not detected and full_text != "[no speech detected]":
        # Check exact phrase in full text
        if TRIGGER_PHRASE in full_text:
            detected = True
            best_match_score = 100.0
            best_match_text = TRIGGER_PHRASE
        # Word-level matching
        elif ENABLE_WORD_LEVEL_MATCHING:
            matches, score = check_word_level_match(words, TRIGGER_PHRASE)
            if matches:
                detected = True
                if score > best_match_score:
                    best_match_score = score
                    best_match_text = full_text
    
    detected = validate_detection(full_text, best_match_score, detected, TRIGGER_PHRASE)
    return detected, full_text, best_match_score, best_match_text

def process_audio_ensemble(file_path):
    """Process audio with multiple configurations and vote on result."""
    configs = [
        (0.5, 85, "Light NR, Strict Match"),
        (0.6, 85, "Medium NR, Strict Match"),
        (0.7, 85, "Heavy NR, Strict Match"),
        (0.0, 85, "No NR, Strict Match"),
        (0.6, 90, "Medium NR, Very Strict"),
    ]
    
    results = []
    output_folder = "Converted"
    filename = os.path.basename(file_path)
    base, _ = os.path.splitext(filename)
    
    for noise_strength, threshold, description in configs:
        # Prepare audio file
        if noise_strength > 0 and ENABLE_NOISE_REDUCTION:
            temp_path = os.path.join(output_folder, base + "_temp.wav")
            convert_audio_format(file_path, temp_path)
            
            audio_data, sample_rate = sf.read(temp_path)
            reduced_noise = nr.reduce_noise(
                y=audio_data, sr=sample_rate, stationary=True,
                prop_decrease=noise_strength, freq_mask_smooth_hz=500, time_mask_smooth_ms=50
            )
            reduced_noise = normalize_audio(reduced_noise)
            
            test_path = os.path.join(output_folder, base + f"_test_{noise_strength}.wav")
            sf.write(test_path, reduced_noise, sample_rate)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            test_path = os.path.join(output_folder, base + "_test_no_nr.wav")
            convert_audio_format(file_path, test_path)
            
            if ENABLE_AUDIO_NORMALIZATION:
                audio_data, sample_rate = sf.read(test_path)
                audio_data = normalize_audio(audio_data)
                sf.write(test_path, audio_data, sample_rate)
        
        # Process audio
        wf = wave.open(test_path, "rb")
        detected, full_text, best_match_score, best_match_text = recognize_audio(
            wf, TRIGGER_PHRASE, threshold
        )
        wf.close()
        
        # Enhanced matching on full text
        words = full_text.split()
        if len(words) >= MIN_TEXT_LENGTH and not detected and full_text != "[no speech detected]":
            if TRIGGER_PHRASE in full_text:
                detected = True
                best_match_score = 100.0
                best_match_text = TRIGGER_PHRASE
            elif ENABLE_WORD_LEVEL_MATCHING:
                matches, score = check_word_level_match(words, TRIGGER_PHRASE)
                if matches:
                    detected = True
                    if score > best_match_score:
                        best_match_score = score
                        best_match_text = full_text
        
        detected = validate_detection(full_text, best_match_score, detected, TRIGGER_PHRASE)
        
        results.append({
            'config': description,
            'detected': detected,
            'text': full_text,
            'score': best_match_score,
            'best_match': best_match_text
        })
        
        # Clean up test file
        if os.path.exists(test_path):
            os.remove(test_path)
    
    # Voting
    detections = sum(1 for r in results if r['detected'])
    total_configs = len(configs)
    confidence = detections / total_configs
    
    # Get best result
    best_result = max(results, key=lambda x: x['score'])
    full_text = best_result['text']
    best_match_score = best_result['score']
    best_match_text = best_result['best_match']
    
    # Final decision
    final_detected = confidence >= ENSEMBLE_VOTING_THRESHOLD
    
    # Check for "hell" confusion with stricter requirements
    if not ALLOW_HELL_SUBSTITUTE and full_text != "[no speech detected]":
        words_lower = [w.lower() for w in full_text.split()]
        has_hell = "hell" in words_lower
        has_help = "help" in words_lower
        if has_hell and not has_help:
            min_configs = max(4, int(total_configs * 0.8))
            if best_match_score < 90 or detections < min_configs:
                final_detected = False
            else:
                final_detected = validate_detection(full_text, best_match_score, confidence >= 0.8, TRIGGER_PHRASE)
    
    if final_detected:
        final_detected = validate_detection(full_text, best_match_score, final_detected, TRIGGER_PHRASE)
    
    # Reject if score too low unless exact phrase found
    if best_match_score < FUZZY_MATCH_THRESHOLD and TRIGGER_PHRASE not in full_text:
        final_detected = False
    
    ensemble_details = f"{detections}/{total_configs} configs detected (confidence: {confidence:.1%})"
    return final_detected, full_text, best_match_score, best_match_text, ensemble_details

# ----------------------------
# MAIN LOOP
# ----------------------------
def write_summary(f, detected_files, not_detected_files, error_files, total_files):
    """Write summary to file."""
    f.write("\n" + "="*50 + "\n")
    f.write("SUMMARY\n")
    f.write("="*50 + "\n\n")
    
    f.write(f"DETECTED ({len(detected_files)}/{total_files}):\n")
    f.write("-" * 50 + "\n")
    for file in detected_files:
        f.write(f"{file}\n")
    
    f.write(f"\nNOT DETECTED ({len(not_detected_files)}/{total_files}):\n")
    f.write("-" * 50 + "\n")
    for file in not_detected_files:
        f.write(f"{file}\n")
    
    if error_files:
        f.write(f"\nERRORS ({len(error_files)}):\n")
        f.write("-" * 50 + "\n")
        for file, error in error_files:
            f.write(f"{file}: {error}\n")

def print_summary(detected_files, not_detected_files, error_files, total_files):
    """Print summary to console."""
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    print(f"\n DETECTED: {len(detected_files)}/{total_files}")
    for file in detected_files:
        print(f"   {file}")
    
    print(f"\nNOT DETECTED: {len(not_detected_files)}/{total_files}")
    for file in not_detected_files:
        print(f"   {file}")
    
    if error_files:
        print(f"\n ERRORS: {len(error_files)}")
        for file, error in error_files:
            print(f"   {file}: {error}")

detected_files = []
not_detected_files = []
error_files = []

with open(output_file, "w") as f:
    header = "Sample Name | Detected (Y/N) | Recognized Text | Best Match Score"
    if ENABLE_ENSEMBLE:
        header += " | Ensemble Details"
    f.write(header + "\n")
    f.write("-" * 120 + "\n")
    
    for file_name in test_files:
        print(f"Processing {file_name}...")
        try:
            if ENABLE_ENSEMBLE:
                detected, full_text, best_score, best_text, ensemble_details = process_audio_ensemble(file_name)
                f.write(f"{file_name} | {'Y' if detected else 'N'} | {full_text} | {best_score}% | {ensemble_details}\n")
                print(f" → {'Detected ' if detected else 'Not detected '}")
                print(f"    Recognized: '{full_text}'")
                if best_text:
                    print(f"    Best match: '{best_text}' (score: {best_score}%)")
                print(f"    Ensemble: {ensemble_details}")
            else:
                clean_file = ensure_pcm16(file_name, apply_noise_reduction=ENABLE_NOISE_REDUCTION)
                detected, full_text, best_score, best_text = process_audio(clean_file)
                f.write(f"{file_name} | {'Y' if detected else 'N'} | {full_text} | {best_score}%\n")
                print(f" → {'Detected ' if detected else 'Not detected '}")
                print(f"    Recognized: '{full_text}'")
                if best_text:
                    print(f"    Best match: '{best_text}' (score: {best_score}%)")
            
            if detected:
                detected_files.append(file_name)
            else:
                not_detected_files.append(file_name)
        except Exception as e:
            error_msg = f"{file_name} | ERROR: {e} | N/A | N/A"
            if ENABLE_ENSEMBLE:
                error_msg += " | N/A"
            f.write(error_msg + "\n")
            print(f"Error with {file_name}: {e}")
            error_files.append((file_name, str(e)))
    
    write_summary(f, detected_files, not_detected_files, error_files, len(test_files))

print_summary(detected_files, not_detected_files, error_files, len(test_files))
print(f"\n Testing complete. Results saved to {output_file}")
