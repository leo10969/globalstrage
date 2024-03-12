import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import SpectralNormalization
import torch
import pandas as pd
import numpy as np
import os
from wordgesture_gan_model import VariationalEncoder


#--------------ジェネレータへの入力の準備（word_prototypeとユーザの描いたジェスチャの結合）--------------
 
# パディングを行う関数
def pad_gestures(gesture_data, max_length):
    # gesture_dataの形状は(step, features)、ここでfeatures=2です
    padded = np.zeros((max_length, gesture_data.shape[1]))
    sequence_length = min(max_length, gesture_data.shape[0])
    padded[:sequence_length, :] = gesture_data[:sequence_length, :]
    return padded

# 単語のプロトタイプを読み込む関数
def load_word_prototype(word):
    prototype_df = pd.read_csv(f'/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/prototype/prototype_csv/{word}.csv')
    # ダミー座標のz_posを追加して、形状を(128, 3)にする
    prototype_df['z_pos'] = 0.0  # z_posはダミーの座標なので、ここでは全て0とする
    return prototype_df[['x_pos', 'y_pos', 'z_pos']].values

# CSVファイルから全てのジェスチャデータを保存，潜在コードzに変換後，それぞれのword_prototypeと結合する関数
def process_gestures_and_prototypes(csv_folder_path, words_list, encoder, max_steps):
    print('Processing gestures and prototypes...')
    all_real_data = []  # train_stepにてDiscriminatorに与える
    all_z = []  # train_stepにてgen-loss内のL_latを計算するために与える
    all_generator_inputs = []  # train_stepにてGeneratorに与える
    
    if max_steps == 0:
    # max_stepsが0の場合，最大ステップ数を見つけるために一度すべてのファイルをループ
        for word in words_list:
            csv_files = [file for file in os.listdir(csv_folder_path) if file.startswith(word) and file.endswith('.csv')]
            for file in csv_files:
                df = pd.read_csv(os.path.join(csv_folder_path, file), usecols=range(12))
                df['temp_group'] = ((df['sentence'] != df['sentence'].shift())).cumsum()
                last_max_steps = max_steps
                max_steps = max(max_steps, df.groupby('temp_group').size().max())
                if last_max_steps != max_steps:
                    print(f'New max steps at: {word}')
        # 最大ステップ数をファイルに保存
        with open('max_steps.txt', 'w') as f:
            f.write(str(max_steps))
    # ジェスチャデータのパディングと処理
    for word in words_list:
        word_prototype = load_word_prototype(word)  # word_prototypeは(128, 3)の形状
        csv_files = [file for file in os.listdir(csv_folder_path) if file.startswith(word) and file.endswith('.csv')]
        for file in csv_files:
            df = pd.read_csv(os.path.join(csv_folder_path, file), usecols=range(12))
            df['temp_group'] = ((df['sentence'] != df['sentence'].shift())).cumsum()
            #各ファイルのジェスチャ数（最大5つ）分の処理
            for group in df['temp_group'].unique():
                group_df = df[df['temp_group'] == group]
                gesture_data = group_df[['x_pos', 'y_pos']].values
                # パディングを行います（パディングのための最大長を指定する必要があります）
                padded_gesture_data = pad_gestures(gesture_data, max_steps)

                all_real_data.append(padded_gesture_data)
                # エンコーダに平坦化したデータを渡す前に適切な形状に変形する
                flattened_padded_gesture_data = padded_gesture_data.flatten()
                # エンコーダを適用して潜在ベクトルを取得
                mu, log_var = encoder(flattened_padded_gesture_data.reshape(1, -1))
                z = encoder.reparameterize(mu, log_var)
                all_z.append(z)

                # TensorFlowのテンソルをNumPy配列に変換
                z_transformed = z.numpy()
                # zを(128, 32)の形状に繰り返し拡張
                z_repeated = np.repeat(z_transformed, word_prototype.shape[0], axis=0)
                # word_prototypeと結合
                generator_input = np.concatenate([word_prototype, z_repeated], axis=1)
                all_generator_inputs.append(generator_input)
    print('Generator inputs processed.')
    return np.array(all_real_data), all_z, np.array(all_generator_inputs), max_steps

#-----------------各種入力をフォルダに保存-----------------

def save_processed_data(all_real_data, all_z, all_generator_inputs, max_steps, save_dir='processed_data'):
    # ディレクトリが存在しない場合は作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, 'all_real_data.npy'), all_real_data)
    np.save(os.path.join(save_dir, 'all_z.npy'), np.array([z.numpy() for z in all_z]))  # テンソルをNumpy配列に変換
    np.save(os.path.join(save_dir, 'all_generator_inputs.npy'), all_generator_inputs)
    with open(os.path.join(save_dir, 'max_steps.txt'), 'w') as f:
        f.write(str(max_steps))


#-----------------実行部分-----------------
# データセットの作成
csv_folder_path = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/datasets/datasets_per_word/train_datasets_sampled'
# 訓練データセットの単語リストを読み込む
with open('train_wordslist.txt', 'r') as f:
    train_words_list = f.read().splitlines()

with open('max_steps.txt', 'r') as f:
    max_steps = int(f.read())

# エンコーダの読み込み
encoder = VariationalEncoder()
all_real_data, all_z, all_generator_inputs, max_steps = process_gestures_and_prototypes(csv_folder_path, train_words_list, encoder, max_steps)

save_processed_data(all_real_data, all_z, all_generator_inputs, max_steps)