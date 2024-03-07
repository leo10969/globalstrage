import os

# 訓練用またはテスト用のフォルダのパスを指定
folder_path = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/datasets_per_word/test_datasets'
# 訓練用またはテスト用のワードリストの保存先を指定
output_file_path = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/formatting/test_wordslist.txt'


# フォルダ内の全ての.csvファイルのファイル名を取得
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# ファイル名から単語の部分を抽出（小文字に変換）
words = [os.path.splitext(f)[0].lower() for f in csv_files]  # 拡張子を除去して単語部分だけを取得

# 単語のリストをテキストファイルに保存
with open(output_file_path, 'w') as f:
    for word in words:
        f.write(f"{word}\n")

print(f"単語のリストが{output_file_path}に保存されました。")


# #1つのファイルに訓練用（テスト用）ジェスチャをまとめている場合は以下のコードを使用

# import pandas as pd

# # CSVファイルのパスを指定
# csv_file_path = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/train_datasets/data_two.csv'
# # 結果を保存するテキストファイルのパス
# output_file_path = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/formatting/test_data_wordlist.txt'

# # CSVファイルを読み込む
# df = pd.read_csv(csv_file_path, usecols=range(12))

# # 'word'列を取得
# word_column = df['word']

# # 重複を排除し、単語のリストを作成
# unique_words = word_column.drop_duplicates().tolist()

# # リストをテキストファイルに保存
# with open(output_file_path, 'w') as f:
#     for word in unique_words:
#         f.write(f"{word}\n")

# print(f"単語のリストが{output_file_path}に保存されました。")
