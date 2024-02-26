import os
import glob

from collections import Counter

def filter_csv_files_and_count_words(source_directory, target_directory, log_file_path):
    word_counts = Counter()  # 単語の出現回数を記録するためのカウンター
    touch_sequence_active = False  # タッチシーケンスのアクティブ状態を追跡
    current_word = ""  # 現在のタッチシーケンスで追跡されている単語

    csv_files = glob.glob(os.path.join(source_directory, '*.csv'))

    for csv_file in csv_files:
        try:
            base_name = os.path.basename(csv_file)
            formatted_file_name = base_name.replace('.csv', '_formatted.csv')
            formatted_file_path = os.path.join(target_directory, formatted_file_name)

            with open(csv_file, 'r') as file_reader:
                lines = file_reader.readlines()

            if not lines:
                continue

            headers = lines[0].strip().split(',')
            try:
                err_index = headers.index("is_err")
                word_index = headers.index("word")
                event_index = headers.index("event")
            except ValueError as e:
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f'Error in {csv_file}: {e}\n')
                continue

            with open(formatted_file_path, 'w') as file_writer:
                file_writer.write(lines[0])  # ヘッダー行を書き込む

                for line in lines[1:]:
                    data = line.strip().split(',')
                    if data[err_index] == '0' and len(data[word_index]) != 1:
                        file_writer.write(line)

                        if data[event_index] == "touchstart" or data[event_index] == "touchmove":
                            touch_sequence_active = True
                            current_word = data[word_index]
                        elif data[event_index] == "touchend" and touch_sequence_active:
                            word_counts[current_word] += 1
                            touch_sequence_active = False

        except Exception as e:
            with open(log_file_path, 'a') as log_file:
                log_file.write(f'Error processing file {csv_file}: {e}\n')

    # 単語の統計情報をログファイルに書き込む
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'Total different words: {len(word_counts)}\n')
        frequent_words = [word for word, count in word_counts.items() if count >= 6]
        log_file.write(f'Words occurring 6 or more times: {frequent_words}\n')

# ディレクトリパスを指定
source_directory = '/Users/satoreo/iplab/python/gesture_template/swipecsvwithcomma'
target_directory = '/Users/satoreo/iplab/python/gesture_template/swipecsvs_formatted'
log_file_path = '/Users/satoreo/iplab/python/gesture_template/process_log.txt'

# 関数を実行
filter_csv_files_and_count_words(source_directory, target_directory, log_file_path)

