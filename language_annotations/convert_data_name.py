import glob
import json
import os

import pandas as pd

idir = "/home/isil/ComplexBrains/annotations_repos/friends.stimuli/language_annotations/assembly_AI_json/json_aa/"
output_dir = "/home/isil/ComplexBrains/annotations_repos/friends.stimuli/language_annotations/assembly_AI_json/json_converted/"
for season in range(1, 7):
    for tsv_path in sorted(glob.glob(f"{idir}/s{season}/friends_s*.tsv")):
        my_file = pd.read_csv(tsv_path, sep="\t")
        episode = tsv_path.split("/")[-1].split(".")[0][8:15]
        print(episode)
        tsv_filepath_full = os.path.join(
            output_dir, f"s{season}/task-{episode}_desc-utterances_events.tsv"
        )

        my_file.to_csv(tsv_filepath_full, sep="\t", index=False)


idir = "/home/isil/ComplexBrains/annotations_repos/friends.stimuli/language_annotations/assembly_AI_json/json_aa/"
output_dir = "/home/isil/ComplexBrains/annotations_repos/friends.stimuli/language_annotations/assembly_AI_json/json_converted/"
for season in range(1, 7):
    for json_path in sorted(glob.glob(f"{idir}/s{season}/friends_s*.json")):
        with open(json_path, "r") as f:
            data = json.load(f)
            episode = json_path.split("/")[-1].split(".")[0][8:15]
            print(episode)
            json_filepath_full = os.path.join(
                output_dir, f"s{season}/task-{episode}_desc-utterances_events.json"
            )
            os.rename(json_path, json_filepath_full)
