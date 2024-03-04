import pandas as pd
import os
import glob
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf

# GPUを選択
GPU = 1

def merge_and_split_gestures(input_directory, output_folder):
    # 出力フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    csv_files = glob.glob(os.path.join(input_directory, '*.csv'))
    all_data = []  # 全ジェスチャデータを格納するリスト
    all_groups = []  # 各ジェスチャのグループID（ここでは単語）を格納するリスト

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, usecols=range(12))
        all_data.append(df)
        # "word"列の値をグループIDとして使用
        all_groups.extend(df['word'])

    # 全データを結合
    merged_df = pd.concat(all_data, ignore_index=True)

    # GroupShuffleSplitを使用して訓練セットとテストセットに分割
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(merged_df, groups=all_groups))

    train_df = merged_df.iloc[train_idx]
    test_df = merged_df.iloc[test_idx]

    # 分割したデータをCSVファイルに保存
    train_df.to_csv(os.path.join(output_folder, 'data_eight.csv'), index=False)
    test_df.to_csv(os.path.join(output_folder, 'data_two.csv'), index=False)

with tf.device('/gpu:{}'.format(GPU)):
    # 入力ディレクトリと出力フォルダを指定
    input_directory = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/swipecsvs_new'
    output_folder = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/train_datasets'

    # ジェスチャのまとまりを統合して分割
    merge_and_split_gestures(input_directory, output_folder)
