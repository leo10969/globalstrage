import os
import glob

def remove_empty_files(directory):
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    for csv_file in csv_files:
        with open(csv_file, 'r') as file_reader:
            lines = file_reader.readlines()
            if len(lines) <= 1:
                os.remove(csv_file)
                print(f"Removed empty file: {csv_file}")


# ディレクトリパスを指定
directory = '/Users/satoreo/python_projects/word-gesture_gan/datasets/swipecsvs_filtered'

# 空のファイルを削除する関数を実行
remove_empty_files(directory)

