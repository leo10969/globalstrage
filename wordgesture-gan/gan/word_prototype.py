import matplotlib.pyplot as plt
import numpy as np

# キーボードの各キーの正規化された中心座標
key_centers = {
    'Q': (0.05, 0.07), 'W': (0.15, 0.07), 'E': (0.25, 0.07), 'R': (0.35, 0.07), 'T': (0.45, 0.07),
    'Y': (0.55, 0.07), 'U': (0.65, 0.07), 'I': (0.75, 0.07), 'O': (0.85, 0.07), 'P': (0.95, 0.07),
    'A': (0.1, 0.21), 'S': (0.2, 0.21), 'D': (0.3, 0.21), 'F': (0.4, 0.21), 'G': (0.5, 0.21),
    'H': (0.6, 0.21), 'J': (0.7, 0.21), 'K': (0.8, 0.21), 'L': (0.9, 0.21),
    'Z': (0.2, 0.35), 'X': (0.3, 0.35), 'C': (0.4, 0.35), 'V': (0.5, 0.35), 'B': (0.6, 0.35),
    'N': (0.7, 0.35), 'M': (0.8, 0.35)
}

# 描画する単語
word = "this"
n = 128  # 点の総数

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

# 計算されたプロットの座標を出力
for i in range(len(x_values)):
    print(f"Point {i+1}: ({x_values[i]}, {y_values[i]})")


# グラフにプロット
plt.figure(figsize=(10, 4.2))
plt.plot(x_values, y_values, marker='o', linestyle='-', markersize=2)  # 点をプロット
plt.title(f"Stroke for the word '{word}' with uniform points")
plt.grid(True)

# X軸の範囲を0から1に設定
plt.xlim(0, 1)
plt.ylim(0, 0.42)

# X軸とY軸のラベル
plt.xlabel("Normalized X Coordinate")
plt.ylabel("Normalized Y Coordinate")

# # グラフのアスペクト比を実際のキーボードの比率に合わせる
# plt.gca().set_aspect(4.2/10)

# Y軸の方向を逆転して原点を左上に設定
plt.gca().invert_yaxis()

# グラフを表示
plt.show()