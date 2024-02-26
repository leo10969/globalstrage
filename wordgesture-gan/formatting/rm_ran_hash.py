import os
import glob
import random
from collections import defaultdict

def remove_words_from_files(directory, words):
    word_files = defaultdict(set)

    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    for csv_file in csv_files:
        with open(csv_file, 'r') as file_reader:
            lines = file_reader.readlines()

        headers = lines[0].strip().split(',')
        try:
            word_index = headers.index("word")
        except ValueError:
            continue

        for line in lines[1:]:
            data = line.strip().split(',')
            word = data[word_index]
            if word in words:
                word_files[word].add(csv_file)

    for word, files in word_files.items():
        # files をリストに変換
        files_list = list(files)
        files_to_remove = random.sample(files_list, min(len(files_list), max(0, len(files_list) - 5)))
        for file in files_to_remove:
            with open(file, 'r') as file_reader:
                lines = file_reader.readlines()

            with open(file, 'w') as file_writer:
                file_writer.write(lines[0])
                for line in lines[1:]:
                    if word != line.strip().split(',')[word_index]:
                        file_writer.write(line)

# 単語リスト（6回以上出現した単語）
f = open('formatting/6timesormore.txt', 'r')
frequent_words = f.read()  # ここに6回以上出現した単語リストを入れる
# print(words)
f.close()

# 処理を実行
directory = '/Users/satoreo/iplab/python/gesture_template/swipecsvs_filtered'
remove_words_from_files(directory, frequent_words)
