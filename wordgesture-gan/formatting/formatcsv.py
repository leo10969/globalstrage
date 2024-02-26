import os
import glob

def convert_log_to_csv_with_commas(source_directory, target_directory):
    # .log ファイルのリストを取得
    log_files = glob.glob(os.path.join(source_directory, '*.log'))

    for log_file in log_files:
        # 新しいファイル名を作成（.log を .csv に変更）
        base_name = os.path.basename(log_file)
        csv_file_name = base_name.replace('.log', '.csv')
        csv_file_path = os.path.join(target_directory, csv_file_name)

        # .log ファイルを読み込み、.csv ファイルとして書き出し
        with open(log_file, 'r') as file_reader:
            with open(csv_file_path, 'w') as file_writer:
                for line in file_reader:
                    # 空白で分割し、カンマで結合
                    data = line.split()
                    csv_line = ','.join(data)
                    file_writer.write(csv_line + '\n')

# ディレクトリパスを指定
source_directory = '/Users/satoreo/iplab/python/gesture_template/swipelogs'
target_directory = '/Users/satoreo/iplab/python/gesture_template/swipecsvwithcomma'

# 変換関数を実行
convert_log_to_csv_with_commas(source_directory, target_directory)
