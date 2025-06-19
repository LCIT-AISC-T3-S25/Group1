import re
import emoji
import spacy
import json
from utils.slang_dict import slang_dict

# Load abbreviation dictionary
with open("utils/abbr_dict.json", "r", encoding="utf-8") as f:
    abbr_dict = json.load(f)

nlp = spacy.load("en_core_web_sm")

def expand_terms(text, mapping_dict):
    count = 0
    for short, expanded in mapping_dict.items():
        pattern = r'\b' + re.escape(short) + r'\b'
        matches = len(re.findall(pattern, text))
        if matches > 0:
            text = re.sub(pattern, expanded, text)
            count += matches
    return text, count

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = emoji.demojize(text).replace(":", " ").replace("_", " ")

    text, _ = expand_terms(text, slang_dict)
    text, _ = expand_terms(text, abbr_dict)

    text = re.sub(r"http\S+|www\.\S+", "", text)     # URLs
    text = re.sub(r"@\w+", "", text)                    # Mentions
    text = re.sub(r"#\w+", "", text)                    # Hashtags
    text = re.sub(r"&\w+;", "", text)                   # HTML entities
    text = re.sub(r"[^a-z\s]", "", text)                # Non-letter characters
    text = re.sub(r"\s+", " ", text).strip()            # Whitespace

    return text

def handle_negation(text):
    doc = nlp(text)
    negated = {token.head.i for token in doc if token.dep_ == "neg"}
    return " ".join(["NOT_" + token.text if i in negated else token.text for i, token in enumerate(doc)])