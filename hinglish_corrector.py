import re
import os
import pickle
import argparse
from autocorrect import Speller
from hinglish_classifier import HinglishClassifier

spell = Speller(lang='en')

# A curated set of extremely common Hinglish words that spellcheck might accidentally turn to English.
# Added from common spoken hindi.
COMMON_HINDI_SET = {
    "kya", "hai", "ka", "ki", "ke", "ko", "haan", "nahi", "nahin", "yeh", "voh", 
    "ek", "mein", "par", "se", "aur", "hota", "hoti", "hote", "raha", "rahi", "rahe", 
    "tha", "thi", "the", "toh", "hi", "bhi", "kuch", "koi", "sab", "bahut", "thoda",
    "karna", "karo", "karti", "karte", "kar", "jana", "jao", "jata", "jati", "jate",
    "aana", "aao", "aata", "aati", "aate", "dekh", "dekho", "dekha", "suno", "suna",
    "kaha", "kaho", "bol", "bolo", "bola", "kabhi", "jaise", "kaise", "waisa", "waise",
    "mera", "meri", "mere", "tera", "teri", "tere", "tum", "tumhara", "apna", "apni",
    "apne", "hum", "hamara", "iski", "iska", "iske", "uski", "uska", "uske", "jiski",
    "jiska", "jiske", "kaun", "kahan", "yahan", "wahan", "jab", "tab", "ab", "kab",
    "isliye", "lekin", "magar", "agar", "phir", "kyon", "kyunki", "baat", "din", "raat",
    "log", "aadmi", "shok", "shoka"
}

ml_model = HinglishClassifier()

def custom_spell(word):
    # General dictionary for the most common English loan words in Hindi
    # that Whisper consistently transliterates phonetically.
    phonetic_map = {
        'aphis': 'office',
        'skul': 'school',
        'kalej': 'college',
        'klas': 'class',
        'taim': 'time',
        'pliz': 'please',
        'sari': 'sorry',
        'thaink': 'thank',
        'thainks': 'thanks',
        'helau': 'hello',
        'phon': 'phone',
        'nambar': 'number',
        'pulis': 'police',
        'daktar': 'doctor',
        'haspital': 'hospital',
        'tikat': 'ticket',
        'marning': 'morning',
        'maik': 'mic',
        'baink': 'bank',
        'kalar': 'color',
        'tren': 'train',
        'bas': 'bus',
        'kampyutar': 'computer',
        'intaranet': 'internet',
        'siti': 'city',
        'miting': 'meeting',
        'inglish': 'english',
        'hind': 'hindi',
        'muvi': 'movie',
        'vat': 'what',
        'iph': 'if',
        'ekshili': 'actually', # Very common phonetic distortion
        'sirisli': 'seriously', # Very common phonetic distortion
        'audiens': 'audience',
        'besik': 'basic',
        'phand': 'funda',
        'tainis': 'tennis',
        'betar': 'better',
        'injar': 'injured',
        'desain': 'design',
        'desaind': 'designed',
        'phrej': 'phrase',
        'phorm': 'form',
        'myujik': 'music',
        'rilij': 'release',
        'rekard': 'record'
    }
    
    word_lower = word.lower()
    
    # 1. Phonetic override check (direct match)
    if word_lower in phonetic_map:
        return phonetic_map[word_lower], "en"
        
    # 2. Prevent checking common isolated letters except 'a' and 'i'
    if len(word) == 1 and word_lower not in ['a', 'i']:
        return word, "hi"
        
    # 3. Known Hindi list bypass
    if word_lower in COMMON_HINDI_SET:
        return word, "hi"
        
    # 4. Use ML logic combined with Spell distance
    spelled = spell(word_lower)
    
    if spelled != word_lower:
        classification = ml_model.classify(word_lower)
        # If ML model guarantees it is Hindi, it's likely a misclassified Hindi word.
        if classification == 'hi':
            return word, "hi"
        return spelled, "en"
        
    # 5. Advanced Generalizer Fallback: 
    # If standard autocorrect doesn't recognize it, testing common phonetic replacements
    # to see if it unlocks an English word. (e.g. 'ph' -> 'f' or 'k' -> 'c')
    if spelled == word_lower:
        test_word = word_lower.replace('ph', 'f').replace('k', 'c')
        spelled_fallback = spell(test_word)
        if spelled_fallback != test_word:
            # Check if this newly unlocked word is confidently English
            classifier_fallback = ml_model.classify(spelled_fallback)
            if classifier_fallback == 'en':
                return spelled_fallback, "en"
                
    classification = ml_model.classify(word_lower)
    return word, classification


def correct_hinglish_sentence(sentence):
    # Find all words (preserving punctuation spacing)
    tokens = re.findall(r"[\w']+|[.,!?; ]", sentence)
    
    corrected_tokens = []
    
    for tk in tokens:
        if tk.strip() and re.search(r'[a-zA-Z]', tk):  # Is a word
            # Preserve capitalization
            is_upper = tk.isupper()
            is_title = tk.istitle()
            
            corrected_word, lang = custom_spell(tk)
            
            if is_upper:
                corrected_word = corrected_word.upper()
            elif is_title:
                corrected_word = corrected_word.title()
                
            corrected_tokens.append(corrected_word)
        else:
            corrected_tokens.append(tk)
            
    final_sentence = "".join(corrected_tokens)
    
    # Sentence capitalize
    def capitalize_match(match):
        return match.group(1) + match.group(2).upper()
    final_sentence = re.sub(r'(^|[\.\?\!]\s+)([a-z])', capitalize_match, final_sentence)
    
    return final_sentence
    
def clean_srt_file(input_srt, output_srt=None):
    if output_srt is None:
        name, ext = os.path.splitext(input_srt)
        output_srt = f"{name}_cleaned{ext}"

    with open(input_srt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    cleaned_lines = []
    for line in lines:
        if re.match(r'^\d+$', line.strip()) or '-->' in line:
            cleaned_lines.append(line)
        elif line.strip() == '':
            cleaned_lines.append(line)
        else:
            # Clean and classify
            cleaned_lines.append(correct_hinglish_sentence(line) + "\n" if not line.endswith("\n") else correct_hinglish_sentence(line))
            
    with open(output_srt, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)
        
    return output_srt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("srt_file")
    parser.add_argument("--inplace", action="store_true")
    args = parser.parse_args()
    out = args.srt_file if args.inplace else None
    print(f"Cleaning SRT using Classification Model: {args.srt_file}")
    res = clean_srt_file(args.srt_file, out)
    print(f"Done! {res}")
