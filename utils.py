# utils.py - file utils: (loading, saving, I/O) functions

import csv, json, xml.etree.ElementTree as ET
import re, os, pickle, sys
from collections import defaultdict, Counter
from pathlib import Path

def load_tei(tei_path  : Path, langs : list[str]) -> Path:
    """
    Parses a TEI bilingual dictionary and saves (foreign word, English translation) pairs to
    comma-separated format into a .csv file.
    """
    tree = ET.parse(tei_path)
    root = tree.getroot()

    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    entries = []

    print(f"⚙️ Parsing dictionary into CSV format...")

    for entry in root.findall('.//tei:entry', ns):
        source = None
        target = None

        # form in source language (English word)
        form = entry.find('.//tei:form/tei:orth', ns)
        if form is not None:
            text = str(form.text)
            source = text.strip()
        else:
            print(f'\t> Ignoring untranslated dictionary entry : [{entry}]', file=sys.stderr)
            continue
        
        # translation in target language (Greek translation)
        sense = entry.find('.//tei:sense/tei:def', ns)
        if sense is not None:
            # multiword translations: keep only 1st option
            sense = str(sense.text)
            target = sense.split(',')[0].strip()
        else:
            print(f'\t> Ignoring untranslated dictionary entry : [{entry}]', file=sys.stderr)
            continue
        
        if source and target:
            entries.append((source, target))

    # save to .csv
    csv_path = tei_path.with_suffix('.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([langs[1], langs[0]])
        writer.writerows(entries)

    print(f"✅ Processed {len(entries)} dictionary [{langs[1]}-{langs[0]}] entries from {tei_path.name}!\n\n")

    return csv_path

def load_csv(path  : Path, langs : list[str]) -> list[tuple[str,str]]:
    pairs = []
    with path.open(newline='', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip header
        # save (f,e) pairs
        for f, e in reader:
            pairs.append((f.strip(), e.strip()))

    print(f"✅ Loaded {len(pairs)} dictionary [{langs[1]}-{langs[0]}] entries from {path.name}!")

    return pairs


def load_dictionary(path  : Path, langs : list[str], csv : bool = False) -> list[tuple[str,str]]:
    """Loads a bilingual dictionary of comma-separated pairs of (source,target) words,
    parsed from a .tei file, into a list of those pairs (or directly from a .csv)."""
    csv_path = path if csv else load_tei(path, langs)
    return load_csv(csv_path, langs)

def load_corpus(path  : Path, preprocess=True) -> list[str]:
    """Loads and preprocesses a monolingual corpus."""
    text = path.read_text(encoding='utf-8')
    output_path = path.with_stem(path.stem + "_clean")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"⚙️ Preprocessing monolingual corpus...")

    if preprocess:
        # gather all sentences to be processed
        # eliminate sentences with too many non-Greek characters,
        # less than a potential (n=3) contextual window
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text)
        full_corpus = re.split(r'(?<=[.;!?])\s+', text)

        corpus = []

        for sentence in full_corpus:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # sentence must be in Greek, mostly
            # change/extend for any target language
            foreign_chars = sum(1 for c in sentence if '\u0370' <= c <= '\u03FF' or '\u1F00' <= c <= '\u1FFF')
            total_chars = len(sentence)
            
            if total_chars == 0 or (foreign_chars / total_chars) < 0.5:
                continue
            
            # normalize and tokenize sentence
            sentence = sentence.lower()
            sentence = re.sub(r'[“”‘’«»]', '"', sentence)
            sentence = re.sub(r'([.,;!?])(?=\S)', r'\1 ', sentence)
            sentence = re.sub(r'\.{2,}', '.', sentence)
            
            tokens = re.findall(r"[Α-Ωα-ωά-ώϊΐόύώ]+|[\w]+|[.,;!?]", sentence)
            if len(tokens) >= 3:
                corpus.append(sentence)

        output_path.write_text('\n'.join(corpus), encoding='utf-8')

        with open(output_path, 'w', encoding='utf-8') as out_f:
            for line in corpus:
                out_f.write(line.strip() + '\n')


    else:
        sentences = re.split(r'(?<=[.;!?])\s+', text)
        corpus = [s.strip() for s in sentences if s.strip()]
    
    print(f"✅ Preprocessed {len(corpus)} monolingual corpus sentences into {output_path}!\n\n")
    
    return corpus

def load_natural_gender(path  : Path) -> dict[str,str]:
    """Load a gendered nouns JSON and keeps only the natural M/F assignments,
    or loads the natural assignments directly."""
    
    data = json.loads(path.read_text(encoding='utf-8'))
    gendered_nouns = {}

    print(f"⚙️ Loading naturally-gendered words...")

    for entry in data:
        word = entry.get("word")
        if "_" in word:
            continue
        gender = entry.get("gender")

        # only applicable for m/f
        if gender == "m" or gender == "f":
            gendered_nouns[word] = gender.upper()
        else:
            # ignore 'n' (neutral), only apply for foreign language if possible
            print(f'\t> Ignoring neuter gender "n" for entry: [{entry}]', file=sys.stderr)
            
    output_path = path.with_stem(path.stem + "_natural").with_suffix(".json")
    output_path.write_text(json.dumps(gendered_nouns, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"✅ Loaded {len(gendered_nouns)} gendered nouns!\n\n")
    return gendered_nouns


def load_suffixes(path  : Path, langs : list) -> dict[str, str]:
    """Loads suffixes patterns as suffix-assigned gender pairs onto file (sorted by suffix-length priority)."""
    suffixes = {}
    
    print(f"⚙️ Loading suffix patterns...")

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip empty lines or comments
            parts = line.split(",")
            if len(parts) != 2:
                print(f'\t> Ignoring malformed line: [{line}]', file=sys.stderr)
                continue  # skip bad lines
            
            # suffix - gender pairs
            s, g = parts
            suffixes[s] = g

    # (!) priority: longer suffixes go first
    prioritised = dict(sorted(suffixes.items(), key=lambda x: -len(x[0])))
    print(f"✅ Loaded {len(suffixes)} suffix patterns [{langs[0]}-{langs[1]}] entries from {path}!\n\n")
    
    return prioritised

def save_predictions(results : dict[str, str], json_path  : Path):
    """Saves gender prediction results from word-gender pairs to file in JSON format."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"\n >>> 🗃️ Saved gender induction predictions to file {json_path.name}")

def save_model(model : defaultdict[str,Counter], path : Path):
    """Serializes object model into file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(model, f)

def load_model(path  : Path) -> defaultdict[str,Counter] | None:
    """Loads serialized object from file into object."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        context_model = pickle.load(f)
    
    if context_model:
        return context_model
    else:
        return None

