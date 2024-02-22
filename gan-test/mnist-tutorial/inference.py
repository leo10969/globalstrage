# inference.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

def generate_images(generator, batch_size=100):
    """
    生成器から画像を生成して表示する関数
    """
    W = int(np.sqrt(batch_size))
    H = int(np.sqrt(batch_size))

    # ノイズを生成
    noise = tf.random.normal([batch_size, 100])
    # 画像を生成
    generated_images = generator(noise, training=False)
    # 逆正規化
    generated_images = (generated_images * 127.5) + 127.5
    generated_images = np.clip(generated_images, 0, 255).astype(np.uint8)

    # 画像を表示
    fig, axes = plt.subplots(W, H, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i in range(batch_size):
        ax = axes[i // W, i % W]
        ax.imshow(generated_images[i, :, :, 0], cmap='gray')
        ax.axis('off')

    plt.suptitle("Generated Images", fontsize=20)
    plt.show()

if __name__ == "__main__":
    # コマンドライン引数からモデルのパスを取得
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_your_model>")
        sys.exit(1)

    model_path = '/workspaces/gan_test/gan_tutorial/gan_tutorial/modelfolder'

    # モデルのロード
    generator_loaded = tf.keras.models.load_model(model_path)

    # バッチサイズ
    batch_size = 100

    # 画像の生成と表示
    generate_images(generator_loaded, batch_size=batch_size)
