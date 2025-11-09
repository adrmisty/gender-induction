# words.py - functions in relation to morphology and grammar
from gender import *
from utils import save_model, load_model
import spacy as sp
from collections import defaultdict, Counter

# first do: python -m spacy download en_core_web_sm
nlp_en = sp.load('en_core_web_sm')

def get_seeds(dictionary : list[tuple[str,str]]) -> list[tuple[str,str]]:
    """
    Extracts seed noun-gender pairs from a bilingual dictionary using PoS tagging on the English side
    (using a Spacy model tagger).
    
    This function scans the English part of each dictionary pair, identifies nouns,
    and checks whether they have natural gender (e.g., man->M, woman->F...).
    
    Args:
        dictionary (list): a list of (foreign_word, english_word) tuples.
    
    Returns:
        list: a list of (foreign_word, gender) tuples, such that the
        respective translation in English is a noun with natural gender.
    """
    print("⚙️ Extracting gendered seed nouns...")
    nouns = []

    for f, e in dictionary:
        # english tagging
        # (!) on my previous work on Multilingual NLP, all tagging/parsing/anything
        # morphology related I have done was using UdPipe... unfortunately it takes
        # a lot of resources that sometimes it renders my apps useless and I cannot
        # effectively gather meaningful results (or at least with decent accuracies)
        # > resorting to using Spacy instead (used it for a past project years ago
        # related to word embeddings)
        doc = nlp_en(e)
        for token in doc:
            # only process nouns + those inherently gendered!
            if token.pos_ == "NOUN" and has_natural_gender(token.text.lower()):
                gender = get_natural_gender(token.text.lower())
                nouns.append((f, gender))
    return nouns

def get_trie(seed_nouns : list[tuple[str,str]], suffix_patterns : dict[str,str], max_len : int = 5) -> defaultdict[str,Counter]:
    """
    Builds a suffix trie from seed nouns and reference grammar suffix patterns.
    
    The trie maps suffixes to gender frequencies based on their appearance in seed nouns and grammar rules.
    
    Args:
        seed_nouns (list): list of (word, gender) tuples to extract suffixes from.
        suffix_patterns (dict): reference suffix-to-gender mapping with assumed weights.
        max_len (int, optional): maximum suffix length to consider, default value is 5.
    
    Returns:
        defaultdict: a trie-like structure (dict of Counters) mapping suffixes to gender frequency counts.
    """
    print("⚙️ Building gender suffix trie...")
    trie = defaultdict(Counter)
    
    # suffixes from seed nouns
    for word, gender in seed_nouns:
        suffixes = [word[-i:] for i in range(1, min(len(word), max_len)+1)]
        for suffix in suffixes:
            trie[suffix][gender] += 1

    # reference grammar suffix patterns
    """ For my NPFL094 course project, I gathered:
        "masc-ος": ["ος"],  # άνθρωπος - man
        "masc-ας": ["ας"],  #  πατέρας - father
        "masc-ης": ["ης", "ής"],  # ποιητής - poet
        "neut-μα": ["μα"],  # πρόβλημα - problem
        "neut-ι": ["ι", "ός"], # σπίτι - house, ποτάμι/ός - river
        "neut-ο": ["ο", "ό"],   # βιβλίο - book, βουνό - mount
        "fem-α": ["α"],  # χώρα - country
        "fem-η": ["η", "ή"],  # ψυχή - soul
    """
    if suffix_patterns:
        for suffix, gender in suffix_patterns.items():
            trie[suffix][gender] += 10  # inc. weight

    return trie

def get_context(path : Path, corpus : list[str], seed_nouns : list[tuple[str,str]], window_size : int = 3) -> defaultdict[str,Counter]:
    """
    Builds (or loads, if already built) a gender-aware context model based on seed nouns in a monolingual corpus.
    
    This model captures co-occurrence statistics of words surrounding seed nouns within a specified window size.
    If a model already exists at the given path, it is loaded instead of rebuilt.
    
    Args:
        path (Path): path to load/save the context model.
        corpus (list): a list of sentence strings.
        seed_nouns (list): a list of (foreign_word, gender) tuples, such that the respective translation in English is a noun with natural gender.
        window_size (int, optional): Number of words to the left and right to consider as context. Defaults to 3.
    
    Returns:
        dict: A mapping from context words to gender frequency Counters.
    """
    context_model = load_model(path)
    if context_model:
        print("⚙️ Loaded context model!")
        return context_model
    
    print("⚙️ Building context model...")
    context_model = defaultdict(Counter)
        
    seed_dict = dict(seed_nouns)
    seed_set = set(seed_dict)

    tokenized_corpus = [
            [word.lower() for word in re.findall(r'\w+', sentence)]
            for sentence in corpus
    ]

    for tokens in tokenized_corpus:
        len_tokens = len(tokens)
        for i, token in enumerate(tokens):
            if token in seed_set:
                gender = seed_dict[token]
                start = max(0, i - window_size)
                end = min(len_tokens, i + 1 + window_size)
                for j in range(start, end):
                    if j != i:
                        context_model[tokens[j]][gender] += 1
    save_model(context_model, path)
    print(f"\tSaved built context model in {path}!")

    return context_model

def get_corpus_index(corpus : list[str]) -> defaultdict[str, list[str]]:
    """
    Creates an index mapping each unique word in the corpus to the list of sentences where it appears.
    
    Args:
        corpus (list): A list of sentences (strings).
    
    Returns:
        dict: A dictionary mapping words to lists of sentences they occur in.
    """
    index = defaultdict(list[str])
    for sentence in corpus:
        for word in set(re.findall(r'\w+', sentence.lower())):
            index[word].append(sentence)
    return index
