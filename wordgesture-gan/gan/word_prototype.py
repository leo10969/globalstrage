import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import re

# キーボードの各キーの正規化された中心座標
key_centers = {
    'Q': (0.05, 0.07), 'W': (0.15, 0.07), 'E': (0.25, 0.07), 'R': (0.35, 0.07), 'T': (0.45, 0.07),
    'Y': (0.55, 0.07), 'U': (0.65, 0.07), 'I': (0.75, 0.07), 'O': (0.85, 0.07), 'P': (0.95, 0.07),
    'A': (0.1, 0.21), 'S': (0.2, 0.21), 'D': (0.3, 0.21), 'F': (0.4, 0.21), 'G': (0.5, 0.21),
    'H': (0.6, 0.21), 'J': (0.7, 0.21), 'K': (0.8, 0.21), 'L': (0.9, 0.21),
    'Z': (0.2, 0.35), 'X': (0.3, 0.35), 'C': (0.4, 0.35), 'V': (0.5, 0.35), 'B': (0.6, 0.35),
    'N': (0.7, 0.35), 'M': (0.8, 0.35)
}

# # 描画する単語
# word = "found"

# アルファベットのみで構成されているかをチェックする関数
def is_alphabetical(word):
    return re.match(r'^[a-zA-Z]+$', word) is not None

# ファイルを読み込む関数
def read_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:  # エンコーディングを指定
        for line in file:
            print(line)
            _, word = line.strip().split()  # 1列目の文字列を取得
            if len(word) > 1 and is_alphabetical(word):  # 文字列の長さが1より大きく、アルファベットのみで構成されている場合のみ返す
                yield word


# 各単語のプロット座標を生成し、CSVと画像を保存する関数
def generate_and_save_plots(words, csv_folder, img_folder):
    n = 128  # 点の総数
    for word in words:
        print(f"Generating prototype for {word}")
        # 単語を構成する各文字の中心座標を取得
        x_coords = [key_centers[char.upper()][0] for char in word]
        y_coords = [key_centers[char.upper()][1] for char in word]

        # 点を均一に配置
        x_values = []
        y_values = []
        for i in range(len(x_coords) - 1):
            x_values.extend(np.linspace(x_coords[i], x_coords[i + 1], n // len(word) + 1)[:-1])
            y_values.extend(np.linspace(y_coords[i], y_coords[i + 1], n // len(word) + 1)[:-1])
        x_values.append(x_coords[-1])  # 最後の点を追加
        y_values.append(y_coords[-1])  # 最後の点を追加

        # # 計算されたプロットの座標を出力
        # for i in range(len(x_values)):
        #     print(f"Point {i+1}: ({x_values[i]}, {y_values[i]})")

        # CSVファイルの保存
        csv_file_path = os.path.join(csv_folder, f"{word}.csv")
        with open(csv_file_path, 'w') as csv_file:
            # ヘッダー
            csv_file.write("event,x_pos,y_pos,word\n")
            # データの書き込み
            for i in range(len(x_values)):
                event = 'touchmove'
                if i == 0:
                    event = 'touchstart'
                elif i == len(x_values) - 1:
                    event = 'touchend'
                csv_file.write(f"{event},{x_values[i]},{y_values[i]},{word}\n")

        # 画像の保存
        plt.figure(figsize=(10, 4.2))
        plt.plot(x_values, y_values, marker='o', linestyle='-', markersize=2)
        plt.title('{} prototype'.format(word))
        plt.grid(True)
        plt.xlim(0, 1)
        plt.ylim(0, 0.42)
        plt.gca().invert_yaxis()
        img_file_path = os.path.join(img_folder, f"{word}.png")
        plt.savefig(img_file_path)
        plt.close()

# メイン関数
def main():
    # GPUを選択
    GPU = 1
    with tf.device('/gpu:{}'.format(GPU)):
        #word_freq.txtファイルのパス
        file_path = '/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/formatting/words_freq.txt'
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
