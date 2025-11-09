# gender.py - functions in relation to gender

import re
from collections import defaultdict, Counter
from utils import load_natural_gender
from pathlib import Path

GENDERED_NOUNS = load_natural_gender(Path("./data/en_gendered.json")) # from file

def get_gender_from_suffix(word : str, suffix_trie : defaultdict[str,Counter], suffix_len : int = 3):
    """
    Predicts the gender of a word based on its suffix using a suffix trie.
    
    Iteratively checks the last 1 to [suffix_len] characters of the word and looks them up in the trie. 
    If a match is found, the most frequent gender associated with that suffix is returned.
    
    Args:
        word (str): the word to analyze.
        suffix_trie (dict): a trie mapping of suffixes to gender frequency counters.
        suffix_len (int): default 3, desired suffix length. 
    Returns:
        str or None: The predicted gender ("masc", "fem", or "neut") or None if no match is found.
    """
    for i in range(1, min(len(word), suffix_len)+1):
        suffix = word[-i:]
        if suffix in suffix_trie:
            gender_counts = suffix_trie[suffix]
            # the most frequent one
            if gender_counts:
                return gender_counts.most_common(1)[0][0]  
    return None

def get_gender_from_context(word : str, 
                            context_model : defaultdict[str,Counter], 
                            index : defaultdict[str, list[str]], 
                            window_size : int = 3, 
                            max_sentences : int =100) -> str | None:
    """
    Predicts the gender of a word based on its surrounding context in a monolingual corpus.
    
    Uses a pre-built context model and an inverted index to find gender-associated words that 
    co-occur near the target word. Scores are accumulated from both left and right contexts 
    within a specified window size across up to [max_sentences].
    
    Args:
        word (str): the word to analyze.
        context_model (dict): a model mapping context words to gender frequency counters.
        index (dict): an inverted index mapping words to the sentences they appear in.
        window_size (int, optional): number of tokens to consider on each side, default value is 3.
        max_sentences (int, optional): maximum number of sentences to sample, default value is 100.
    
    Returns:
        str or None: The predicted gender ("M", "F", or "N") or None if insufficient data.
    """
    if word not in index:
        return None
    
    gender_counter = Counter()

    # limit context search for a max. of 100 sentences
    for i, sentence in enumerate(index[word][:max_sentences]):
        tokens = re.findall(r'\w+', sentence.lower())
        
        for i, token in enumerate(tokens):
            if token == word:
                # left
                left_context = tokens[max(0, i-window_size):i]
                # right
                right_context = tokens[i+1:i+1+window_size]
                context_words = left_context + right_context
                # update counts
                for ctx_word in context_words:
                    if ctx_word in context_model:
                        gender_counter.update(context_model[ctx_word])
    if gender_counter:
        # the most frequent one
        return gender_counter.most_common(1)[0][0]
    return None

# ---------------------------  auxiliary

def has_natural_gender(word : str) -> bool:
    """Off a handwritten list of inherently-gendered words in English,
    determines whether a given word is "naturally" gendered."""
    return word in GENDERED_NOUNS

def get_natural_gender(word : str) -> str | None:
    """Off a handwritten list of inherently-gendered words in English,
    retrieves a noun word's natural gender (if any)."""
    return str(GENDERED_NOUNS.get(word, None))

