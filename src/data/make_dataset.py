import re
import os

"""
Run through all files in the directories given, and append the text to list after lowering and removing HTML characters
"""
def raw_data_extraction(directories):
    raw_text_lst = []
    raw_target_lst = []
    raw_text_id = []
    for target in range(len(directories)):
        directory = directories[target]
        for idx, filename in enumerate(os.listdir(directory)):
            if filename.endswith(".txt"):
#                if idx%1000 == 0:
#                    print('File {} of {} in {}'.format(idx, len(os.listdir(directory)),directory))
                raw_target_lst.append(target)
                raw_text_id.append(os.path.splitext(filename)[0])
                with open(directory+filename, 'r') as f:
                    text = f.read().lower()         # lower all text
                    text = re.sub('<[^<]+?>', " ", text)  # removes HTML characters
                    raw_text_lst.append(text)
                continue
    return raw_text_lst, raw_target_lst, raw_text_id

