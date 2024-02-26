import os
import glob
import random
from collections import defaultdict

def process_files(directory, frequent_words):
    # 各単語に対して、出現するファイルとその中のグループ位置を記録
    word_file_groups = defaultdict(lambda: defaultdict(list))

    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    for csv_file in csv_files:
        with open(csv_file, 'r') as file:
            lines = file.readlines()

        headers = lines[0].strip().split(',')
        try:
            word_index = headers.index('word')
        except ValueError:
            continue  # "word" 列がなければこのファイルをスキップ

        previous_word = None
        group_start_index = None
        for i, line in enumerate(lines[1:], start=1):
            data = line.strip().split(',')
            current_word = data[word_index]

            if current_word == previous_word or previous_word is None:
                if group_start_index is None:
                    group_start_index = i
            else:
                if previous_word in frequent_words:
                    word_file_groups[previous_word][csv_file].append((group_start_index, i-1))
                group_start_index = i if current_word in frequent_words else None

            previous_word = current_word

        # ファイルの最後の単語グループを記録
        if previous_word in frequent_words and group_start_index is not None:
            word_file_groups[previous_word][csv_file].append((group_start_index, len(lines)))

    # 各単語に対して余分なグループを削除
    for word, files in word_file_groups.items():
        total_groups = sum(len(groups) for groups in files.values())
        if total_groups <= 5:
            continue  # 5個以下のグループは削除不要

        # 余分なグループを削除するファイルとグループを選択
        groups_to_remove = total_groups - 5
        while groups_to_remove > 0:
            for file, groups in files.items():
                if groups_to_remove > 0 and groups:
                    # ランダムにグループを選択して削除
                    group_to_remove = random.choice(groups)
                    groups.remove(group_to_remove)
                    groups_to_remove -= 1

                    # ファイルから選択したグループを削除
                    with open(file, 'r') as f:
                        lines = f.readlines()
                    with open(file, 'w') as f:
                        f.writelines(line for i, line in enumerate(lines, start=1)
                                     if not (group_to_remove[0] <= i <= group_to_remove[1]))

# 単語リスト（6回以上出現した単語）を読み込む
with open('formatting/6timesormore.txt', 'r') as f:
    frequent_words = [line.strip() for line in f.readlines()]

# 処理を実行
directory = '/Users/satoreo/iplab/python/gesture_template/swipecsvs_filtered'
process_files(directory, frequent_words)
