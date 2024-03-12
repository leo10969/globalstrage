import pandas as pd
import os
import glob
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf

# GPUを選択
GPU = 1

def merge_and_split_gestures(input_directory, train_output_folder, test_output_folder):
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(train_output_folder):
        os.makedirs(train_output_folder)
    if not os.path.exists(test_output_folder):
        os.makedirs(test_output_folder)

    csv_files = glob.glob(os.path.join(input_directory, '*.csv'))
    all_data = []  # 全ジェスチャデータを格納するリスト
    all_groups = []  # 各ジェスチャのグループID（ここでは単語）を格納するリスト

    for csv_file in csv_files:
        print(f'Processing {csv_file}')
        df = pd.read_csv(csv_file, usecols=range(12))
        all_data.append(df)
        # "word"列の値をグループIDとして使用
        all_groups.extend(df['word'].unique())  # 重複を避けるためにunique()を使用

    # 全データを結合
    merged_df = pd.concat(all_data, ignore_index=True)

    # GroupShuffleSplitを使用して訓練セットとテストセットに分割
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(merged_df, groups=merged_df['word']))

    train_df = merged_df.iloc[train_idx]
    test_df = merged_df.iloc[test_idx]

    unique_words_train = train_df['word'].unique()
    unique_words_test = test_df['word'].unique()

    # 訓練データセットに含まれる単語のファイルを保存
    for word in unique_words_train:
        word_train_df = train_df[train_df['word'] == word]
        train_filename = f'{word}.csv'
        word_train_df.to_csv(os.path.join(train_output_folder, train_filename), index=False)

    # テストデータセットに含まれる単語のファイルを保存
    for word in unique_words_test:
        word_test_df = test_df[test_df['word'] == word]
        test_filename = f'{word}.csv'
        word_test_df.to_csv(os.path.join(test_output_folder, test_filename), index=False)

# GPUデバイスの選択
with tf.device('/gpu:{}'.format(GPU)):
    # 入力ディレクトリと出力フォルダを指定
    input_directory = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/swipecsvs_128'
    train_output_folder = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/datasets_per_word/train_datasets'
    test_output_folder = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/datasets_per_word/test_datasets'

    # ジェスチャのまとまりを統合して分割
    merge_and_split_gestures(input_directory, train_output_folder, test_output_folder)
