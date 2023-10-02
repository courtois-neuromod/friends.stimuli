import random
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from bpemb import BPEmb
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.util import ngrams

nltk.download("omw-1.4")
nltk.download("stopwords")
nltk.download("wordnet")


from collections import Counter

from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# def convert_to_data_frame(item_range, value, key_1, key_2):
#     dictionary = {}
#     for item in range(1, item_range):
#         dictionary[item] = len(value[item])

#     dataframe = pd.DataFrame({key_1: value.keys(), key_2: value.values()})
#     return dataframe


def load_csv_file(datapath: str):
    data = pd.read_csv(datapath, encoding="ISO-8859-1")
    seasons = data["season"]
    episodes = data["episode"]
    corpus = data["line"]
    return seasons, corpus, episodes, data

    # if "narrator" in data:
    #     narrator = data["narrator"]
    #     return seasons, corpus, episodes, narrator
    # else:
    #     return seasons, corpus, episodes


# found outlawed words


def find_outlaw(word):
    """Find words that contain a same character 3+ times in a row."""
    is_outlaw = False
    for i, letter in enumerate(word):
        if i > 1:
            if word[i] == word[i - 1] == word[i - 2] and word[i].isalpha():
                is_outlaw = True
                break
    return is_outlaw


def word_characteristics(data):
    # concatanate all strings from all seasons
    narrative = "".join(data["line"])
    words = re.findall("\w+", narrative)
    unique_words = set(re.findall(r"\w+", narrative, re.UNICODE))
    return words, unique_words


def word_statistics(data):
    season_word = {}
    season_unique_word = {}
    freq_splits = {}
    len_unique_words = []
    len_word_per_season = []
    complete_narrative = []
    season_range = list(range(1, 11))

    for season in range(1, 11):
        rslt_df = data[data["season"] == season]
        narrative_per_season = "".join(rslt_df["line"])
        docwords = re.findall("\w+", narrative_per_season)
        season_word[season] = docwords
        season_unique_word[season] = set(
            re.findall(r"\w+", narrative_per_season, re.UNICODE)
        )
        freq_splits[season] = FreqDist(docwords)
        complete_narrative += docwords

    outlaws = [s for s in complete_narrative if find_outlaw(s)]
    res = np.array(outlaws)
    unique_res = np.unique(res)
    random.shuffle(unique_res)
    freq_splits_all_seasons = FreqDist(complete_narrative)
    frequent_word_list = []
    frequency_word_list = []
    for i in range(0, 20):
        frequent_word_list.append(freq_splits_all_seasons.most_common(20)[i][0])
        frequency_word_list.append(freq_splits_all_seasons.most_common(20)[i][1])

    return (
        season_word,
        season_unique_word,
        freq_splits,
        unique_res,
        frequent_word_list,
        frequency_word_list,
        outlaws,
    )


def stopword_statistics(tokens_norm):
    stop_words = set(stopwords.words("english"))
    for season in range(1, 11):
        freq_stopwords = [(sw, tokens_norm[season].count(sw)) for sw in stop_words]
        freq_stopwords.sort(key=lambda x: x[1], reverse=True)
    return stop_words, freq_stopwords


def token_statistics(data):
    number_of_tokens = {}
    tokens_norm = {}
    tokens_all = []
    total_number_of_tokens = 0
    tokens_season = {}
    bpemb_en = BPEmb(lang="en")
    tokenizer = RegexpTokenizer(r"\w+")
    lemmatiser = WordNetLemmatizer()
    for season in range(1, 11):
        rslt_df = data[data["season"] == season]
        narrative = "".join(rslt_df["line"])
        tokens_season[season] = bpemb_en.encode(narrative)
        tokens_norm[season] = [
            lemmatiser.lemmatize(t.lower(), "v") for t in bpemb_en.encode(narrative)
        ]
        total_number_of_tokens += len(tokens_season[season])
        tokens_all.append(tokens_season[season])
    return tokens_season, tokens_norm, tokens_all, total_number_of_tokens


def preprocess_text(rslt_df, stop_words):
    """Preprocess the trslt_dfext to its purest form without any punctuation and stop words."""

    narrative = "".join(rslt_df["line"])
    tokeniser = RegexpTokenizer(r"[A-Za-z]{2,}")
    tokens = tokeniser.tokenize(narrative)
    # Lowercase and lemmatise
    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(token.lower(), pos="v") for token in tokens]
    # Remove stopwords
    keywords = [lemma for lemma in lemmas if lemma not in stop_words]
    return keywords


def get_frequent_ngram(keywords, ngram, n=20):
    """Find most common n n-grams tokens."""

    # # Find ngrams per document
    n_grams = list(ngrams(keywords, ngram))

    # # Find frequency of ngrams
    n_grams_flattened = [item for sublist in n_grams for item in sublist]
    # freq_dist = FreqDist(n_grams)
    # top_freq = freq_dist.most_common(n)
    freq_dist = FreqDist(n_grams_flattened)
    top_freq = freq_dist.most_common(n)
    return pd.DataFrame(top_freq, columns=["ngram", "count"])


def ngram_extractor(text, n_gram, stop_words):
    token = [
        token
        for token in text.lower().split(" ")
        if token != ""
        if token not in stop_words
    ]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]


# Function to generate a dataframe with n_gram and top max_row frequencies
def generate_ngrams(df, n_gram, max_row, stop_words):
    temp_dict = defaultdict(int)
    for question in df:
        for word in ngram_extractor(question, n_gram, stop_words):
            temp_dict[word] += 1
    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(
        max_row
    )
    temp_df.columns = ["word", "wordcount"]
    return temp_df


def plot_corpus_characteristics(
    value, title: str, xlabel: str, ylabel: str, tilt: False, type: str
):
    plt.figure(figsize=(6, 4))

    if type == "bar":
        # plt.bar(data_range, value, color="#4958B5")
        sns.barplot(x=list(value.keys())[0], y=list(value.keys())[1], data=value)
    elif type == "count":
        sns.countplot(y=value)
    elif type == "ngram":
        fig, ax = plt.subplots()
        y_pos = np.arange(len(value.keys())[0])
        ax.barh(y_pos, list(value.keys())[1])
        ax.set_yticks(y_pos, labels=list(value.keys())[1])
        ax.invert_yaxis()

    plt.title(title, fontsize=12)
    plt.ylabel(ylabel, fontsize=10)
    plt.xlabel(xlabel, fontsize=10)
    if tilt == True:
        plt.xticks(rotation=45)

    plt.show()


# ['/home/isil/ComplexBrains/friends.stimuli/language_annotations', '/usr/lib/python310.zip', '/usr/lib/python3.10', '/usr/lib/python3.10/lib-dynload', '', '/home/isil/.local/lib/python3.10/site-packages', '/home/isil/ComplexBrains/friends_LLM_benchmark', '/usr/local/lib/python3.10/dist-packages', '/usr/lib/python3/dist-packages', '/usr/lib/python3/dist-packages/IPython/extensions', '/home/isil/.ipython']
