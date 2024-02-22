#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# TensorFlowを使用してMNISTデータをロード
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ランダムに100個の画像を読み込む
idx = np.random.choice(np.arange(0, x_train.shape[0]), 100)

imgs = x_train[idx]
labels = y_train[idx]

# 10x10マスに表示
fig, axes = plt.subplots(10, 10, figsize=(10, 10))

for i in range(imgs.shape[0]):
    ax = axes[i // 10, i % 10]
    img = imgs[i]
    ax.imshow(img, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    # # 画像の上にラベルを表示
    # ax.set_title(labels[i], fontsize=8)

#%%
fig.suptitle("MNIST Sample Images with Labels", fontsize=20)
plt.tight_layout()  # タイトルが重ならないようにレイアウトを調整
plt.savefig('mnist_sample_images.png')
plt.show()
