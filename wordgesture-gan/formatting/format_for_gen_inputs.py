import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def process_group(group, original_columns):
    n_points = 128
    #128点にダウンサンプリング
    if len(group) > n_points:
        indices = np.round(np.linspace(0, len(group) - 1, n_points)).astype(int)
        new_group = group.iloc[indices].reset_index(drop=True)
    #128点に線形補完
    elif len(group) < n_points:
        interpolated_values = {}
        for col in ['x_pos', 'y_pos']:
            f = interp1d(np.arange(len(group)), group[col], kind='linear')
            interpolated_values[col] = f(np.linspace(0, len(group) - 1, n_points))
        # timestampの補間も行う
        f_timestamp = interp1d(np.arange(len(group)), group['timestamp'], kind='linear')
        interpolated_values['timestamp'] = f_timestamp(np.linspace(0, len(group) - 1, n_points))

        # 補間された値で新しいDataFrameを作成し、event列を'touchmove'に設定
        new_group = pd.DataFrame(interpolated_values)
        new_group['event'] = 'touchmove'
        
        # 残りの列について、最初の行の値で埋める
        for col in original_columns:
            if col not in new_group:
                new_group[col] = group[col].iloc[0]

        # 元のDataFrameの列の順序に合わせて並び替え
        new_group = new_group[original_columns]
    else:
        new_group = group
    
    # timestamp 列の処理: 1つ前の行からの経過時間を計算
    # 最初の行は0、それ以降は1つ前の行からの差分を秒単位で計算
    #ilocは，使えなくなるらしい？
    new_group['timestamp'] -= new_group['timestamp'].shift(1)
    new_group['timestamp'].iloc[0] = 0
    new_group['timestamp'] = new_group['timestamp'].fillna(0) / 1000.0

    return new_group

source_dir = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/datasets_per_word/train_datasets'
target_dir = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/datasets_per_word/train_datasets_formatted'

os.makedirs(target_dir, exist_ok=True)

for filename in os.listdir(source_dir):
    if filename.endswith('.csv'):
        file_path = os.path.join(source_dir, filename)
        df = pd.read_csv(file_path, usecols=range(12))
        
        original_columns = df.columns.tolist()  # 元の列の順序を保存
        df['temp_group'] = (df['sentence'] != df['sentence'].shift()).cumsum()
        
        processed_df = df.groupby('temp_group').apply(lambda x: process_group(x, original_columns)).reset_index(drop=True)
        # processed_df = processed_df.drop(columns='temp_group')

        target_file_path = os.path.join(target_dir, filename)
        processed_df.to_csv(target_file_path, index=False)
