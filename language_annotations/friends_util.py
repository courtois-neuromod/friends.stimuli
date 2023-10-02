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


def load_csv_file(datapath: str):
    data = pd.read_csv(datapath, encoding="ISO-8859-1")
    seasons = data["season"]
    episodes = data["episode"]
    corpus = data["dialogue"]
    if "narrator" in data:
        narrator = data["narrator"].str.replace(":", "")
        narrator = narrator.str.capitalize()
        return seasons, corpus, episodes, narrator, data
    return seasons, corpus, episodes, data


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
    narrative = "".join(data["dialogue"])
    words = re.findall("\w+", narrative)
    unique_words = set(re.findall(r"\w+", narrative, re.UNICODE))
    return words, unique_words


def word_statistics(data):
    season_word = {}
    season_unique_word = {}
    freq_splits = {}
    unique_res = {}
    outlaws = {}
    complete_narrative = []

    for season in range(1, 11):
        rslt_df = data[data["season"] == season]
        narrative_per_season = "".join(rslt_df["dialogue"])
        docwords = re.findall("\w+", narrative_per_season)
        season_word[season] = docwords
        season_unique_word[season] = set(
            re.findall(r"\w+", narrative_per_season, re.UNICODE)
        )
        freq_splits[season] = FreqDist(docwords)
        complete_narrative += docwords
        outlaws[season] = [s for s in docwords if find_outlaw(s)]
        res = np.array(outlaws[season])
        unique_res[season] = np.unique(res)
        random.shuffle(unique_res[season])
    print(complete_narrative)

    freq_splits_all_seasons = FreqDist(complete_narrative)
    print(freq_splits_all_seasons)
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
    tokens_norm = {}
    unique_tokens = {}
    unique_tokens_list = {}
    tokens_all = []
    total_number_of_tokens = 0
    tokens_season = {}
    bpemb_en = BPEmb(lang="en")
    tokenizer = RegexpTokenizer(r"\w+")
    lemmatiser = WordNetLemmatizer()
    for season in range(1, 11):
        rslt_df = data[data["season"] == season]
        narrative = "".join(rslt_df["dialogue"])
        tokens_season[season] = bpemb_en.encode(narrative)
        tokens_norm[season] = [
            lemmatiser.lemmatize(t.lower(), "v") for t in bpemb_en.encode(narrative)
        ]

        # Use set to find unique tokens
        unique_tokens[season] = set(tokens_season[season])

        # Convert the set back to a list if needed
        unique_tokens_list[season] = list(unique_tokens[season])
        total_number_of_tokens += len(tokens_season[season])
        tokens_all.append(tokens_season[season])
    return (
        tokens_season,
        unique_tokens_list,
        tokens_norm,
        tokens_all,
        total_number_of_tokens,
    )


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


def character_dialog_counts(narrator):
    # remove characters who talks less than 1000 dialogues
    count = Counter(narrator)
    narrator_count = {}
    i = 0
    for j in count.keys():
        if list(count.values())[i] > 200:
            name = list(count.keys())[i]
            name = re.sub(":", "", name)
            name = name.capitalize()
            narrator_count[name] = list(count.values())[i]
        i += 1

    sorted_narrator_count = sorted(narrator_count.items(), key=lambda x: x[1])

    character_dialogues = {}

    for i in sorted_narrator_count:
        character_dialogues[i[0]] = i[1]

    return narrator_count, sorted_narrator_count, character_dialogues


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
        y_pos = np.arange(len(value))
        ax.barh(y_pos, value[list(value.keys())[1]])
        ax.set_yticks(y_pos, labels=value[list(value.keys())[0]])
        ax.invert_yaxis()

    plt.title(title, fontsize=12)
    plt.ylabel(ylabel, fontsize=10)
    plt.xlabel(xlabel, fontsize=10)
    if tilt == True:
        plt.xticks(rotation=45)

    plt.show()
