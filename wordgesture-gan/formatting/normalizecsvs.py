import pandas as pd
import os
import glob
import tensorflow as tf

# GPUを選択
GPU = 1

def normalize_positions_in_folder(directory):
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, usecols=range(12), keep_default_na=False)
        # "keyb_width"と"keyb_height"に基づいて"x_pos"と"y_pos"を-1から1に正規化
        # 注意：正規化する過程で、x_posは (x_pos / keyb_width) * 2 - 1 で計算
        # y_posは (y_pos / keyb_height) * 2 - 1 で計算
        df["x_pos"] = (df["x_pos"] / df["keyb_width"]) * 2 - 1
        df["y_pos"] = (df["y_pos"] / df["keyb_height"]) * 2 - 1
        
        # 処理後のDataFrameをCSVに再保存
        df.to_csv(csv_file, index=False)

# ディレクトリパスを指定
directory = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/swipecsvs_normalized'

# 関数を実行
with tf.device('/gpu:{}'.format(GPU)):
    normalize_positions_in_folder(directory)
