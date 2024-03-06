import pandas as pd

# CSVファイルのパスを指定
csv_file_path = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/train_datasets/data_two.csv'
# 結果を保存するテキストファイルのパス
output_file_path = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/formatting/test_data_wordlist.txt'

# CSVファイルを読み込む
df = pd.read_csv(csv_file_path, usecols=range(12))

# 'word'列を取得
word_column = df['word']

# 重複を排除し、単語のリストを作成
unique_words = word_column.drop_duplicates().tolist()

# リストをテキストファイルに保存
with open(output_file_path, 'w') as f:
    for word in unique_words:
        f.write(f"{word}\n")

print(f"単語のリストが{output_file_path}に保存されました。")
