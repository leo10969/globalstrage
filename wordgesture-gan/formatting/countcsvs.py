import os
import glob
from collections import Counter

def count_words_in_files(directory):
    total_words = Counter()  # 単語の総数（重複含む）を記録するカウンター
    unique_words = set()  # 異なる単語の集合
    previous_word = None  # 前の単語を追跡

    csv_files = glob.glob(os.path.join(directory, '*.csv'))

    for csv_file in csv_files:
        with open(csv_file, 'r') as file_reader:
            lines = file_reader.readlines()

        headers = lines[0].strip().split(',')
        try:
            word_index = headers.index("word")  # "word" 列のインデックスを見つける
        except ValueError:
            continue  # "word" 列がない場合はスキップ

        for line in lines[1:]:
            data = line.strip().split(',')
            current_word = data[word_index]
            if current_word != previous_word:  # 連続する同じ単語を避ける
                total_words[current_word] += 1
                unique_words.add(current_word)
            previous_word = current_word  # 現在の単語を前の単語として記録

    # 結果の表示
    print(f'Total different words: {len(unique_words)}')
    print(f'Total words (with duplicates, but not counting consecutive duplicates): {sum(total_words.values())}')

# ディレクトリパスを指定
directory = '/Users/satoreo/iplab/python/gesture_template/swipecsvs_formatted2'

# 関数を実行
count_words_in_files(directory)
