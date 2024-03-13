import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import re
import ast

# キーボードの各キーの正規化された中心座標
# キーボードの各キーの正規化された中心座標を-1から1の範囲で計算
key_centers = {
    #上段のキー
    'Q': (-0.9, -0.6666666666666666), 'W': (-0.7, -0.6666666666666666), 'E': (-0.5, -0.6666666666666666),
    'R': (-0.3, -0.6666666666666666), 'T': (-0.1, -0.6666666666666666), 'Y': (0.1, -0.6666666666666666),
    'U': (0.3, -0.6666666666666666), 'I': (0.5, -0.6666666666666666), 'O': (0.7, -0.6666666666666666),
    'P': (0.9, -0.6666666666666666), 
    #中段のキー
    'A': (-0.8, 0.0), 'S': (-0.6, 0.0), 'D': (-0.4, 0.0),
    'F': (-0.2, 0.0), 'G': (0.0, 0.0), 'H': (0.2, 0.0), 'J': (0.4, 0.0), 'K': (0.6, 0.0),
    'L': (0.8, 0.0), 
    #下段のキー
    'Z': (-0.6, 0.6666666666666666), 'X': (-0.4, 0.6666666666666666),
    'C': (-0.2, 0.6666666666666666), 'V': (0.0, 0.6666666666666666), 'B': (0.2, 0.6666666666666666),
    'N': (0.4, 0.6666666666666666), 'M': (0.6, 0.6666666666666666)
}


# # 描画する単語
# word = "found"

# アルファベットのみで構成されているかをチェックする関数
def is_alphabetical(word):
    return re.match(r'^[a-zA-Z]+$', word) is not None

# ファイルから辞書データを読み込む関数
def read_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()  # ファイルの内容を読み込む
        dict_data = ast.literal_eval(data)  # 文字列を辞書として評価

    # 条件に合致するキー（単語）を返す
    for word in dict_data.keys():
        if len(word) > 1 and is_alphabetical(word):  # 文字列の長さが1より大きく、アルファベットのみで構成されている場合
            yield word


# 各単語のプロット座標を生成し、CSVと画像を保存する関数
def generate_and_save_plots(words, csv_folder, img_folder):
    n = 128  # 点の総数
    for word in words:
        # 単語を構成する各文字の中心座標を取得
        x_coords = [key_centers[char.upper()][0] for char in word]
        y_coords = [key_centers[char.upper()][1] for char in word]

        # 点を均一に配置（前回のコードを使い、この部分は変更なし）
        x_values = []
        y_values = []
        for i in range(len(x_coords) - 1):
            x_values.extend(np.linspace(x_coords[i], x_coords[i + 1], (n - len(word)) // (len(word) - 1) + 1)[:-1])
            y_values.extend(np.linspace(y_coords[i], y_coords[i + 1], (n - len(word)) // (len(word) - 1) + 1)[:-1])
        x_values.append(x_coords[-1])  # 最後の点を追加
        y_values.append(y_coords[-1])  # 最後の点を追加

        # ここからCSVファイルの保存処理を修正
        csv_file_path = os.path.join(csv_folder, f"{word.lower()}.csv")
        with open(csv_file_path, 'w') as csv_file:
            # ヘッダー
            csv_file.write("event,x_pos,y_pos,word\n")
            # データの書き込み
            for i in range(n):
                event = 'touchmove'
                if i == 0:
                    event = 'touchstart'
                elif i == n - 1:  # 最後の点を 'touchend' イベントとして扱う
                    event = 'touchend'
                # x_values[i] と y_values[i] が n 個に満たない場合のエラーを避けるための処理
                x_pos = x_values[i] if i < len(x_values) else x_values[-1]
                y_pos = y_values[i] if i < len(y_values) else y_values[-1]
                csv_file.write(f"{event},{x_pos},{y_pos},{word}\n")


        # # 画像の保存
        # plt.figure(figsize=(10, 4.2))
        # plt.plot(x_values, y_values, marker='o', linestyle='-', markersize=2)
        # plt.title('{} prototype'.format(word))
        # plt.grid(True)
        # plt.xlim(0, 1)
        # plt.ylim(0, 0.42)
        # plt.gca().invert_yaxis()
        # img_file_path = os.path.join(img_folder, f"{word}.png")
        # plt.savefig(img_file_path)
        # plt.close()

# メイン関数
def main():
    # GPUを選択
    GPU = 1
    with tf.device('/gpu:{}'.format(GPU)):
        #word_freq.txtファイルのパス
        file_path = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/formatting/word_file_dict2.txt'
        csv_folder = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/prototype/prototype_csv'  # CSVファイルを保存するフォルダ
        img_folder = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/prototype/prototype_imgs'  # 画像を保存するフォルダ

        # フォルダが存在しなければ作成
        os.makedirs(csv_folder, exist_ok=True)
        os.makedirs(img_folder, exist_ok=True)

        # ファイルから単語を読み込み
        words = read_words(file_path)

        # 各単語のプロット座標を生成し、CSVと画像を保存
        generate_and_save_plots(words, csv_folder, img_folder)


if __name__ == "__main__":
    main()
