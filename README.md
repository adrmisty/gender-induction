# 🧠 [NPFL128 Project] Grammatical gender induction with minimum resources

This project implements a pipeline to **induce grammatical gender** for nouns and adjectives relying on:
- **Suffix-based morphological cues**, and
- **Contextual co-occurrence patterns**, 

and low resources:
- a **bilingual dictionary** between English and the target language.
- a **monolingual corpus** in the target language.
- **suffix patterns** from a reference grammar for the target language.

This project has been tested for English and Modern Greek pairs.

### 📖 Reference
This work is inspired by the methodology presented in:

> **Silviu Cucerzan and David Yarowsky (2002).** _“Bootstrapping A Multilingual Part-of-Speech Tagger in One Person Day”_  
> Proceedings of the 6th Conference on Natural Language Learning, CoNLL 2002, Held in cooperation with COLING 2002, Taipei, Taiwan, 2002 (ACL 2002).  [🔗](https://aclanthology.org/2021.acl-long.424)

> See also my presentation on this paper and research. [🔗](.\data\presentation\NPFL128_Adriana_-_presentation.pdf)


## 🚀 Scope

### ✅ Implemented:
- Resource loading and preprocessing:
  - A bilingual dictionary (`.tei`) with English-target language pairs
  - A monolingual corpus (`.txt`) in the target language
  - A suffix-to-gender pattern list (`.csv`) for the target language

- Data extraction:
  - Seed naturally-gendered nouns in English

- Modelling:
  - A suffix trie for morphological prediction
  - A context-based gender co-occurrence model

- Gender induction:
  - Combining previous models to induce gender via weighted voting

- File processing:
  - Loading and reading from file in several formats (.csv, .tei, .txt, .json)
  - Saving to file (.json)
  - Serializing objects (pickle)
- Save predictions in JSON format

### 🔜 Future work:
- Support for PoS filtering
    -> In general, improve the PoS tagging side and ensure proper parts of speech before gender induction (the program currently induces gender for both nouns and adjectives alike, despite being explicitly told to only do it for adjs.)
- Potentially integrate embeddings or language models for deeper context modeling
- Adding evaluation module using gold-annotated test sets


## 🛠️ How to run

### 1. Install dependencies and models
This project uses only built-in Python libraries (e.g., `argparse`, `re`, `collections`) and data processing ones (e.g., `json`, `lxml`, `csv`, `pickle`). For the main NLP task in the program, the library used is `spacy`.

When installing the requirements via the [requirements.txt](requirements.txt) file, make sure to run: ``` python -m spacy download en_core_web_sm``` in order to 

### 2. Gather resources
Ensure that all resource files are available, preferably in the `./data/` folder, at the moment of passing arguments.
Needed resources for the running of the pipeline include:
- a monolingual corpus in the target language, or a sample of it, so that significant context can be retrieved from it [🔗See OPUS examples](https://opus.nlpl.eu/results/es&el/corpus-result-table)
- a bilingual dictionary with English - target pairs, in .tei format. [🔗 See Freedict examples](https://github.com/freedict/fd-dictionaries/tree/master/eng-ell)
- a list of suffix patterns - gender mappings, in .csv format. [🔗 See reference grammar](https://www.foundalis.com/lan/grknouns.htm)

### 2. Run the main pipeline
Note that the order of the languages specified establishes the order of the gender induction.

```
python main.py \
  --langs en el \
  --dict ./data/en-el_dictionary.tei \
  --corpus ./data/el_corpus_clean.txt \
  --suffixes ./data/el_suffixes.csv \
  --results ./results/induced_el.json
```

### 3. Gather the results
The results will be a JSON file mapping each induced noun to its predicted gender (saved to your specified output path).
> [🔗 Greek gender induction prediction](./results/induced_el.json)

> [🔗 Greek gender induction CLI output](./results/output.txt)

```
  "φράουλα": "F", --> "strawberry" ✅ feminine
  "μαργαριτάρι": "N", --> "pearl" ✅ neuter
  "έμπορας": "M", --> "merchant" ✅ masculine
```


## 📁 Project Structure

| File/Folder                | Description |
|----------------------------|-------------|
| `main.py`                  | Entry point, CLI interface |
| `gender.py`, `words.py`    | Logic for word extraction, gender induction via suffix/context modeling |
| `utils.py`                 | General-purpose file loaders, serializers |
| `data/`                    | Input dictionary, corpus, suffix list, presentation |
| `results/`                 | Output files including predictions and logs |
| `README.md`                | Project description and usage guide |



### 👤 Author

**Adriana Rodríguez Flórez**  
Master's student - Computational Linguistics
Project for NPFL128 - Language Technologies in Practice
ÚFAL - Charles University, Prague  
📧 Contact: [adrirflorez@gmail.com](mailto:adrirflorez@gmail.com)  