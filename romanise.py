import re
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def itrans_to_hinglish(itrans_text):
    """
    Converts formal ITRANS notation to casual Romanised Hindi (Hinglish).
    Rules are applied in a specific order (longest/most specific first) to
    avoid partial-match errors.
    """
    t = itrans_text

    # ----------------------------------------------------------------
    # 1. CHANDRABINDU (ँ)
    # ----------------------------------------------------------------
    t = re.sub(r'A\.N', 'aan', t)   # आँ -> aan (e.g., chA.Nda -> chaand)
    t = re.sub(r'a\.N', 'an', t)    # अँ -> an
    t = re.sub(r'i\.N', 'in', t)    # इँ -> in
    t = re.sub(r'u\.N', 'un', t)    # उँ -> un
    t = re.sub(r'e\.N', 'en', t)    # एँ -> en
    t = re.sub(r'o\.N', 'on', t)    # ओँ -> on
    t = re.sub(r'\.N',  'n', t)     # Any remaining .N -> n

    # ----------------------------------------------------------------
    # 2. NUKTA characters
    # ----------------------------------------------------------------
    t = t.replace('.D',  'r')    # ड़  (sadak, sadak) -> r
    t = t.replace('.Dh', 'rh')   # ढ़
    t = t.replace('K',   'kh')   # ख़ (talkh) -> kh
    t = t.replace('q',   'q')    # क़ -> q

    # ----------------------------------------------------------------
    # 3. MULTI-CHARACTER RETROFLEX sequences
    # ----------------------------------------------------------------
    t = t.replace('Sh',  'sh')   # ष / श -> sh
    t = t.replace('Th',  'th')   # ठ -> th
    t = t.replace('Dh',  'dh')   # ढ -> dh
    t = t.replace('Ch',  'chh')  # छ -> chh

    # ----------------------------------------------------------------
    # 4. ANUSVARA (ं) - ITRANS outputs M (capital m)
    # ----------------------------------------------------------------
    t = re.sub(r'M([kgcjtdn])', r'n\1', t)   # before dental/velar/palatal -> n
    t = re.sub(r'M([pbfm])',    r'm\1', t)   # before labial -> m
    t = re.sub(r'M',            'n',    t)   # elsewhere (end of word, etc.) -> n

    # ----------------------------------------------------------------
    # 5. SINGLE-CHARACTER RETROFLEX / LONG ITRANS
    # ----------------------------------------------------------------
    t = t.replace('T',  't')    # ट -> t
    t = t.replace('D',  'd')    # ड -> d
    t = t.replace('N',  'n')    # ण -> n
    t = t.replace('R',  'r')    # ड़ -> r
    t = t.replace('G',  'ng')   # ङ -> ng
    t = t.replace('L',  'l')    # ळ -> l

    # ----------------------------------------------------------------
    # 6. SCHWA DELETION (Internal and Word-Final)
    # ----------------------------------------------------------------
    # Rule 1: Drop word-final short schwa (lowercase 'a')
    t = re.sub(r'(?<=\w{2})a\b', '', t)
    
    # Rule 2: Internal schwa deletion
    cons = r'(kh|gh|sh|ch|ph|bh|th|dh|rh|ng|[bcdfghjklmnpqrstvwxyz])'
    t = re.sub(r'(?<=[aeiouAEIOU])' + cons + r'a' + cons + r'([AEIOU])\b', r'\1\2\3', t)

    # ----------------------------------------------------------------
    # 7. VISARGA (ः)
    # ----------------------------------------------------------------
    t = re.sub(r'H\b', '', t)

    # ----------------------------------------------------------------
    # 8. LONG VOWELS
    # ----------------------------------------------------------------
    t = t.replace('A', 'a')
    t = t.replace('I', 'i')
    t = t.replace('U', 'u')

    # ----------------------------------------------------------------
    # 9. PUNCTUATION & CLEANUP
    # ----------------------------------------------------------------
    # Convert Hindi full stops (which ITRANS makes '|') to English full stops
    t = t.replace('||', '..')
    t = t.replace('|', '.')
    t = t.replace('।', '.')
    t = t.replace('॥', '..')
    
    t = re.sub(r' +', ' ', t)
    t = t.lower().strip()

    return t

def romanise(hindi_text):
    """
    Two-step pipeline:
    1. Devanagari -> ITRANS (using indic_transliteration)
    2. ITRANS -> Casual Hinglish (custom natural grammar rules)
    """
    if not hindi_text or not hindi_text.strip():
        return ""
    itrans = transliterate(hindi_text.strip(), sanscript.DEVANAGARI, sanscript.ITRANS)
    return itrans_to_hinglish(itrans)
