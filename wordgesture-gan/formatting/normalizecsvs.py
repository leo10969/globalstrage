import pandas as pd
import os
import glob
import tensorflow as tf

# GPUを選択
GPU = 1

def normalize_positions_in_folder(directory):
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, usecols=range(12))
        # print(df)
        # "keyb_width"と"keyb_height"に基づいて"x_pos"と"y_pos"を正規化
        #注意：正規化後のkeyb_widthは1，keyb_heightは1 * keyb_height/keyb_widthとなる
        df["x_pos"] = df["x_pos"] / df["keyb_width"]
        df["y_pos"] = df["y_pos"] / df["keyb_height"] * (df["keyb_height"] / df["keyb_width"])
        
        # 処理後のDataFrameをCSVに再保存
        df.to_csv(csv_file, index=False)

# ディレクトリパスを指定
directory = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/swipecsvs_normalized'

# 関数を実行
with tf.device('/gpu:{}'.format(GPU)):
    normalize_positions_in_folder(directory)
