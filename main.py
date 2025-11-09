# main.py
import sys
from pathlib import Path

# logging
log_file = Path("./results/induced_el.log")
log_file.parent.mkdir(parents=True, exist_ok=True)
sys.stdout = open(log_file, "w", encoding="utf-8")
sys.stderr = sys.stdout

# logic
from utils import load_corpus, load_dictionary, load_suffixes, save_predictions
from gender import *
from words import *
import re, argparse



def main(langs: list[str], 
         corpus_path : Path, dict_path : Path, suffix_path : Path, results_path  : Path):
    """
    Runs the full gender induction pipeline for a given language pair.
    
    Loads all necessary resources, extracts linguistic patterns, and runs gender induction 
    on a target language corpus using both suffix and contextual cues.
    
    Args:
        langs (list): A list containing two language codes, e.g., ["en", "el"].
        corpus_path (v): Path to the cleaned monolingual corpus file (.txt format).
        dict_path (Path): Path to the bilingual dictionary file (.tei format).
        suffix_path (Path): Path to the suffix-gender mapping file (.csv format).
        results_path (Path): Path where predictions should be saved (.json format).
    """
    dict, corp, suff = get_resources(langs, dict_path, corpus_path, suffix_path)
    seeds, trie, context = process_resources(dict, corp, suff)
    induce_gender(corp, seeds, context, trie, dict, langs, results_path)

    
def induce_gender(corpus : list[str], 
                  seed_nouns : list[tuple[str,str]], 
                  context_model : defaultdict[str,Counter], 
                  suffix_trie : defaultdict[str,Counter], 
                  bil_dictionary: list[tuple[str,str]], 
                  langs : list[str], 
                  results : Path,
                  weights : tuple[float,float]=(0.7, 0.3)) -> dict[str,str]:
    """
    Induces grammatical gender for words in a monolingual corpus using suffix and context-based analysis.
    
    This function excludes seed nouns from prediction and scores each candidate word using a 
    weighted combination of suffix-based and context-based gender estimations. Results are saved to a JSON file.
    
    Args:
        corpus (list): list of sentences in the target language.
        seed_nouns (list): list of (word, gender) tuples serving as known gender anchors.
        context_model (dict): a context co-occurrence model mapping surrounding words to gender counts.
        suffix_trie (dict): trie mapping word suffixes to gender counts.
        bil_dictionary (list): bilingual dictionary as (source_word, target_word) pairs.
        langs (list): language pair involved in the task, e.g., ["en", "el"].
        weights (tuple, optional): tuple of weights (suffix_score, context_score), default settings are (0.7, 0.3).
        results_path (Path): File path to save the predicted gender mappings.

    Returns:
        dict: a dictionary mapping candidate words to their predicted gender labels ("M", "F", "N").
    """
    # run induction for all words found in the dictionary and in the corpus
    _, target_lang_words = zip(*bil_dictionary)
    dictionary_words = set(word.lower() for word in target_lang_words)
    tokenized_corpus = [
        re.findall(r'\w+', sentence.lower())
        for sentence in corpus
    ]
    vocabulary = set(word for sentence in tokenized_corpus for word in sentence)
    seed_words = {word for word, _ in seed_nouns}

    # candidate words: words in both corpus and dictionary, excluding seed nouns
    candidates = (vocabulary & dictionary_words) - seed_words
    
    n = len(candidates)
    predictions = {}
    index = get_corpus_index(corpus)

    print(f"🧠 Inducing gender for {len(candidates)} words in [{langs[1]}], based on suffix patterns and context analysis...")
    
    for i, word in enumerate(candidates):
        if i % 100 == 0:
            print(f"\t🗣️ [{i}/{n}] word in vocabulary...")
            
        suffix_gender = None
        if suffix_trie:
            suffix_gender = get_gender_from_suffix(word, suffix_trie)
        
        context_gender = get_gender_from_context(word, context_model, index)
        
        # weighted-majority voting
        scores = defaultdict(float)
        if suffix_gender:
            scores[suffix_gender] += weights[0]
        if context_gender:
            scores[context_gender] += weights[1]
        
        if scores:
            predicted_gender = max(scores.items(), key=lambda x: x[1])[0]
            predictions[word] = predicted_gender
    
    # permanence to file
    save_predictions(predictions, results)
    print(f"> 🪄 Gender induction for {langs} pairs finished!\n\n")
    return predictions

def process_resources(dictionary : list[tuple[str,str]], 
                      corpus : list[str], 
                      suffixes : dict[str,str]) -> tuple[list[tuple[str,str]], 
                                                         defaultdict[str,Counter], 
                                                         defaultdict[str,Counter]]:
    """
    Processes linguistic resources to prepare for gender induction.
    
    Extracts gendered seed nouns from the bilingual dictionary, constructs a suffix trie 
    using known suffix-gender mappings, and builds a gender-aware context model from a monolingual corpus.
    
    Args:
        dictionary (list): bilingual dictionary as a list of (foreign_word, english_translation) tuples.
        corpus (list): monolingual corpus in the target language.
        suffixes (dict): reference suffix-gender mapping for the target language.
    
    Returns:
        tuple: (seed_nouns, suffix_trie, context_model)
    """
    seed_nouns = get_seeds(dictionary)
    suffix_trie = get_trie(seed_nouns, suffixes)
    context_model = get_context(context, corpus, seed_nouns)
    print("> 🪄 Resource processing finished!\n\n")

    return seed_nouns, suffix_trie, context_model


def get_resources(langs: list[str], dict_path : Path, corpus_path : Path, suffix_path : Path) -> tuple[list[tuple[str,str]], 
                                                                                                 list[str], 
                                                                                                 dict[str,str]]:
    """
    Loads all necessary, minimum data resources for the gender induction pipeline, being:
      - A bilingual dictionary between English and the target language.
      > has been pre-processed from https://github.com/freedict/fd-dictionaries/tree/master/eng-ell.
      - A cleaned monolingual corpus in the target language.
      > downloaded from Opus https://opus.nlpl.eu/results/en&el/corpus-result-table.
      - Optional suffix pattern mappings for the target language.
      > taken from a reference grammar https://www.foundalis.com/lan/grknouns.htm.
      
    Args:
        langs (list): A list containing two language codes, e.g., ["en", "el"].
        corpus_path (Path): Path to the cleaned monolingual corpus file (.txt format).
        dict_path (Path): Path to the bilingual dictionary file (.tei format).
        suffix_path (Path): Path to the suffix-gender mapping file (.csv format).
    
    Returns:
        tuple: (dictionary, corpus, suffixes)
            - dictionary (list): bilingual dictionary pairs.
            - corpus (list): monolingual corpus sentences.
            - suffixes (dict): suffix-to-gender mappings, optional.
    """
    dictionary = load_dictionary(dict_path, langs)
    corpus = load_corpus(corpus_path)
    suffixes = load_suffixes(suffix_path, langs)
    print("> 🪄 Resource loading finished!\n\n")
    
    return dictionary, corpus, suffixes


# ------------------------------------------------------------------------------------------------


# paths
bilingual_dict = Path("./data/en-el_dictionary.tei")
context = Path("./data/context_model.pkl")
monolingual_corpus_raw = Path("./data/el_corpus.txt")
monolingual_corpus_clean = Path("./data/el_corpus_clean.txt")
patterns = Path("./data/el_suffixes.csv")
induced = Path("./results/induced_el.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gender induction pipeline.")
    
    parser.add_argument(
        "--langs", nargs=2, metavar=("SRC_LANG", "TGT_LANG"),
        help="Specify source and target language codes, e.g., --langs en el",
        default=["en", "el"]
    )

    parser.add_argument(
        "--dict", type=Path,
        help="Path to bilingual dictionary file (.tei format)",
        default=bilingual_dict
    )

    parser.add_argument(
        "--corpus", type=Path,
        help="Path to monolingual corpus file (.txt format)",
        default=monolingual_corpus_raw
    )

    parser.add_argument(
        "--suffixes", type=Path,
        help="Path to suffix-gender list file (.csv format)",
        default=patterns
    )

    parser.add_argument(
        "--results", type=Path,
        help="Path to output predictions file (.json format)",
        default=induced
    )
    args = parser.parse_args()

    main(
        langs=args.langs,
        corpus_path=args.corpus,
        dict_path=args.dict,
        suffix_path=args.suffixes,
        results_path=args.results
    )
