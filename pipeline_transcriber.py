import whisper
import warnings
import os

from romanise import romanise
import re

def convert_to_roman(mixed_text):
    """
    Takes mixed Devanagari & English text from Whisper.
    Isolates the English words to preserve their exact spelling/capitalisation,
    and runs the romanise pipeline only on the Hindi (Devanagari) parts.
    """
    # Regex splits the string by isolating Devanagari words/characters.
    # U+0900 to U+097F is the Devanagari block
    parts = re.split(r'([\u0900-\u097F]+)', mixed_text)
    
    final_output = []
    for part in parts:
        if re.search(r'[\u0900-\u097F]', part):
            # If it contains Devanagari, convert to Hinglish
            final_output.append(romanise(part))
        else:
            # If it's English/Numbers/Punctuation, keep exactly as Whisper wrote it
            final_output.append(part)
            
    return "".join(final_output)


def run_pipeline(audio_path, model_name="turbo"):
    warnings.filterwarnings("ignore")
    
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found.")
        return

    print("Step 1: Loading Whisper model...")
    model = whisper.load_model(model_name)

    print(f"Step 2: Transcribing '{audio_path}' natively in Devanagari...")
    # Force language="hi", task="transcribe" and condition_on_previous_text=False to completely stop 
    # the Whisper model from hallucinating English translations for Hindi sentences.
    dev_prompt = "यह एक हिंदी वाक्य है।"
    result = model.transcribe(
        audio_path, 
        language="hi", 
        task="transcribe", 
        initial_prompt=dev_prompt, 
        condition_on_previous_text=False,
        fp16=False,
        word_timestamps=True
    )
    
    srt_filename = audio_path.rsplit(".", 1)[0] + "_romanised.srt"
    
    def format_timestamp(seconds):
        """Converts float seconds to SRT timestamp format: HH:MM:SS,mmm"""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msec = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02}:{mins:02}:{secs:02},{msec:03}"

    from hinglish_corrector import correct_hinglish_sentence

    MAX_WORDS_PER_SEGMENT = 7
    MAX_DURATION_PER_SEGMENT = 3.0

    with open(srt_filename, "w", encoding="utf-8") as f:
        global_idx = 1
        for segment in result["segments"]:
            words = segment.get("words", [])
            if not words:
                # Fallback if words are missing
                devanagari = segment['text'].strip()
                devanagari = devanagari.replace('ॉ', 'ो').replace('ऑ', 'ओ').replace('ॅ', 'े').replace('ऍ', 'ए')
                romanised = convert_to_roman(devanagari)
                final_subtitle = correct_hinglish_sentence(romanised)
                
                f.write(f"{global_idx}\n")
                f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
                f.write(f"{final_subtitle}\n\n")
                global_idx += 1
                continue

            current_group = []
            group_start = words[0]["start"]
            
            for word_info in words:
                if not current_group:
                    group_start = word_info["start"]
                    
                current_group.append(word_info)
                
                duration = word_info["end"] - group_start
                if len(current_group) >= MAX_WORDS_PER_SEGMENT or duration >= MAX_DURATION_PER_SEGMENT:
                    # Process and write the group
                    text = "".join([w["word"] for w in current_group]).strip()
                    # Clean Devanagari
                    text = text.replace('ॉ', 'ो').replace('ऑ', 'ओ').replace('ॅ', 'े').replace('ऍ', 'ए')
                    romanised = convert_to_roman(text)
                    final_subtitle = correct_hinglish_sentence(romanised)
                    
                    f.write(f"{global_idx}\n")
                    f.write(f"{format_timestamp(group_start)} --> {format_timestamp(word_info['end'])}\n")
                    f.write(f"{final_subtitle}\n\n")
                    
                    global_idx += 1
                    current_group = []
            
            # Handle remaining words in segment
            if current_group:
                text = "".join([w["word"] for w in current_group]).strip()
                text = text.replace('ॉ', 'ो').replace('ऑ', 'ओ').replace('ॅ', 'े').replace('ऍ', 'ए')
                romanised = convert_to_roman(text)
                final_subtitle = correct_hinglish_sentence(romanised)
                
                f.write(f"{global_idx}\n")
                f.write(f"{format_timestamp(group_start)} --> {format_timestamp(current_group[-1]['end'])}\n")
                f.write(f"{final_subtitle}\n\n")
                global_idx += 1

    print(f"\nSuccess! Subtitles have been saved to '{srt_filename}'")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Romanised Hindi (.srt) from video or audio.")
    parser.add_argument("media_file", nargs="?", default="profile.mp3", help="Path to input video or audio file")
    parser.add_argument("--model", default="turbo", help="Whisper model size (default: 'turbo')")
    args = parser.parse_args()
    
    run_pipeline(args.media_file, model_name=args.model)

