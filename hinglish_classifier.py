import re
import pickle
import os
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from romanise import romanise

def train_and_save_model(model_path="hinglish_model.pkl"):
    print("Training character-level Hinglish vs English classifier...")

    try:
        nltk_words = nltk.corpus.words.words()
    except LookupError:
        nltk.download('words')
        nltk_words = nltk.corpus.words.words()
        
    try:
        hindi_words_raw = nltk.corpus.indian.words('hindi.pos')
    except LookupError:
        nltk.download('indian')
        hindi_words_raw = nltk.corpus.indian.words('hindi.pos')

    print("Transliterating Hindi corpus...")
    # Clean the Hindi words and Romanise them
    hindi_romanised = []
    for hw in hindi_words_raw:
        # Keep only Devanagari characters
        hw_clean = re.sub(r'[^\u0900-\u097F]', '', hw)
        if len(hw_clean) > 1: # Ignore single letters
            roman = romanise(hw_clean).strip()
            if len(roman) > 1:
                hindi_romanised.append(roman)

    hindi_romanised = list(set(hindi_romanised))
    
    # Let's augment with some common Hindi words
    common_hindi = [
        "haan", "nahin", "kyon", "kya", "kab", "kaise", "kahan", "kaun", "mera", "tumhara",
        "uski", "isliye", "lekin", "aur", "ya", "phir", "bhi", "hi", "toh", "tak", "mein",
        "par", "se", "ne", "ko", "ke", "ki", "kuch", "koi", "sab", "hamesha", "kabhi",
        "bilkul", "bahut", "thoda", "zyada", "kam", "achha", "kharab", "bura", "ganda",
        "saaf", "naya", "purana", "bada", "chhota", "karna", "hona", "jana", "aana",
        "khana", "pina", "sona", "uthna", "baithna", "dekhna", "sunna", "bolna", "kaha",
        "raha", "rahi", "rahe", "tha", "thi", "the", "hai", "hain", "hoon", "hota", "hoti",
        "hotay", "hoten", "hote", "kafi", "shayad", "sirf", "balki", "chahiye", "sakte", "sakti"
    ]
    hindi_romanised.extend(common_hindi)
    hindi_romanised = list(set(hindi_romanised))

    # Take an equal number of english words (we shuffle or pick random common ones, but 
    # taking top ~2000 from common words is fine. The nltk wordlist is huge ~236k. 
    # Taking random 4000 to match ratio).
    import random
    random.seed(42)
    english_words = random.sample(nltk_words, 4000)
    # Also add top common english words that might be misclassified
    common_english = ["and", "is", "of", "to", "a", "in", "it", "you", "that", "he", "was",
                      "for", "on", "are", "with", "as", "I", "his", "they", "be", "at", 
                      "one", "have", "this", "from", "or", "had", "by", "word", "but",
                      "what", "some", "we", "can", "out", "other", "were", "all", "there",
                      "when", "up", "use", "your", "how", "said", "an", "each", "she",
                      "which", "do", "their", "time", "if", "will", "way", "about", "many",
                      "then", "them", "write", "would", "like", "so", "these", "her", "long",
                      "make", "thing", "see", "him", "two", "has", "look", "more", "day",
                      "could", "go", "come", "did", "number", "sound", "no", "most", "people",
                      "my", "over", "know", "water", "than", "call", "first", "who", "may",
                      "down", "side", "been", "now", "find", "any", "new", "work", "part",
                      "take", "get", "place", "made", "live", "where", "after", "back",
                      "little", "only", "round", "man", "year", "came", "show", "every",
                      "good", "me", "give", "our", "under", "name", "very", "through", "just",
                      "form", "sentence", "great", "think", "say", "help", "low", "line",
                      "differ", "turn", "cause", "much", "mean", "before", "move", "right",
                      "boy", "old", "too", "same", "tell", "does", "set", "three", "want",
                      "air", "well", "also", "play", "small", "end", "put", "home", "read",
                      "hand", "port", "large", "spell", "add", "even", "land", "here", "must",
                      "big", "high", "such", "follow", "act", "why", "ask", "men", "change",
                      "went", "light", "kind", "off", "need", "house", "picture", "try", "us",
                      "again", "animal", "point", "mother", "world", "near", "build", "self",
                      "earth", "father", "head", "stand", "own", "page", "should", "country"]
    english_words.extend(common_english)
    english_words = list(set([w.lower() for w in english_words if len(w) > 1]))

    # Now we have our datasets
    X = hindi_romanised + english_words
    # Label 0: Hinglish, 1: English
    y = [0] * len(hindi_romanised) + [1] * len(english_words)

    # Clean X
    X = [w.lower() for w in X]

    print(f"Training on {len(hindi_romanised)} Hinglish words and {len(english_words)} English words.")
    # Use char level TFIDF to find common substrings like "bb", "gh", "jh"
    pipeline = Pipeline([
        ('vec', TfidfVectorizer(analyzer='char', ngram_range=(2, 4), min_df=2)),
        ('clf', LogisticRegression(C=10.0, max_iter=500))
    ])

    pipeline.fit(X, y)

    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Model saved to {model_path}.")
    return pipeline

class HinglishClassifier:
    def __init__(self, model_path="hinglish_model.pkl"):
        if not os.path.exists(model_path):
            self.model = train_and_save_model(model_path)
        else:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
                
    def classify(self, word):
        """
        Returns 'en' if English, 'hi' if Hinglish.
        """
        # Exclude very short words or purely numeric
        if len(word) < 2 or not re.search(r'[a-zA-Z]', word):
            return 'hi' # default to Hindi
            
        pred = self.model.predict([word.lower()])[0]
        return 'en' if pred == 1 else 'hi'

if __name__ == "__main__":
    classifier = HinglishClassifier()
    test_words = [
        "tainis", "tennis",
        "phrej", "phrase",
        "ekshili", "actually",
        "sirisli", "seriously",
        "vat", "what",
        "iph", "if",
        "port", "sport",
        "betar", "better",
        "injar", "injured",
        "haan", "nahin",
        "shoka", "shok"
    ]
    
    print("\n--- Testing Model Predictions ---")
    for tw in test_words:
        lang = classifier.classify(tw)
        print(f"{tw:15} -> {lang.upper()}")
