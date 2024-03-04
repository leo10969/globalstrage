#ライブラリーのインポート
import GPUtil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense
from tensorflow.keras.layers import Flatten, Input, LeakyReLU, MaxPooling2D, Reshape, UpSampling2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

#%%
# GPUを選択
GPU = 1
#%%
plt.ion()

# GPUの設定
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 最もメモリが空いているGPUを使用する設定
        # GPUメモリの成長を許可する
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # 特定のGPUを使用するようにTensorFlowに指示する
        tf.config.experimental.set_visible_devices(gpus[1], 'GPU')  # 例: 2番目のGPUを使用
    except RuntimeError as e:
        print(e)

#%%
# def select_gpu():
#     GPUs = GPUtil.getGPUs()
#     if GPUs:
#         # 最も空きメモリが多いGPUを選択
#         GPU = sorted(GPUs, key=lambda x: x.memoryFree, reverse=True)[0]
#         return f'/device:GPU:{GPU.id}'
#     else:
#         return '/cpu:0'

#生成器
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator,self).__init__()

        self.fc = layers.Dense(7*7*256,use_bias=False)
        self.bn = layers.BatchNormalization()
        self.relu = layers.LeakyReLU()
        self.reshape = layers.Reshape((7,7,256))
        self.convt1 = layers.Conv2DTranspose(128,(5,5),strides=(1,1),padding='same',use_bias=False)
        self.convt2 = layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False)
        self.convt3 = layers.Conv2DTranspose(1,(5,5),strides=(2,2),padding='same',use_bias=False)

    def call(self,x):
        #ノイズから画像を生成する
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.reshape(x)
        x = self.convt1(x)
        x = self.relu(x)
        x = self.convt2(x)
        x = self.relu(x)
        x = self.convt3(x)

        return x

#識別器
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.conv1 = layers.Conv2D(64,(5,5),strides=(2,2),padding='same')
        self.relu = layers.LeakyReLU()
        self.dropout = layers.Dropout(0.3)
        self.conv2 = layers.Conv2D(128,(5,5),strides=(2,2),padding='same')
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self,x):
        #画像の真偽を判定する
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

#%%
class NewDiscriminator(tf.keras.Model):
    def __init__(self):
        super(NewDiscriminator, self).__init__()
        self.conv1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')
        self.leaky_relu1 = layers.LeakyReLU(alpha=0.2)
        self.dropout1 = layers.Dropout(0.3)

        self.conv2 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.leaky_relu2 = layers.LeakyReLU(alpha=0.2)
        self.dropout2 = layers.Dropout(0.3)

        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

#%%
with tf.device('/gpu:{}'.format(GPU)):
    #適当なノイズから画像を生成する
    generator = Generator()

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise)

    #作成した画像が本物か偽物か判定する
    discriminator = NewDiscriminator()
    decision = discriminator(generated_image).numpy()

#結果を表示
print(decision[0,0])
plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()
# plt.savefig('generated_image.png')

#%%
with tf.device('/gpu:1'):
    #交差エントロピー(分類タスクの損失関数に登場する概念)
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#識別器の損失関数
def discriminator_loss(real_output,fake_output):
    #本物画像が本物と判定されるようにしたい
    #実画像での識別結果と1の配列を比較する
    real_loss = BCE(tf.ones_like(real_output),real_output)

    #偽物画像が偽物と判定されるようにしたい
    #偽画像での識別結果と0の配列を比較する
    fake_loss = BCE(tf.zeros_like(fake_output),fake_output)

    #両者を合わせたものがdiscriminatorの損失関数として定義される
    return real_loss + fake_loss

#生成器の損失関数
def generator_loss(fake_output):
    #本物そっくりの画像を目指す(識別器で1と返すことを目指す)
    return BCE(tf.ones_like(fake_output), fake_output)

#%%
#最適化手法(Adam)

#generator
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#discriminator
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


#%%

# GPUでのモデルの実装コードです
# deviceを引数に渡して、学習させます
def train_model(generator, discriminator, train_dataset, num_epoch, batch_size): 

    G_optimizer = tf.keras.optimizers.Adam(1e-4)
    D_optimizer = tf.keras.optimizers.Adam(1e-4)

    # 本物(1)か偽物(0)かのラベルを生成します
    label_real = tf.ones(shape=(batch_size, 1))
    label_fake = tf.zeros(shape=(batch_size, 1))

    for epoch in range(num_epoch): 
        training=True
        print("\nEpoch {}/{}".format(epoch+1, num_epoch))
        print("-----------")
        epoch_g_loss = 0.0 
        epoch_d_loss = 0.0 

        for train_x in train_dataset: 
            print(train_x.shape)
            ## 1. Discriminatorの学習

            # ノイズをガウス分布からサンプリングします
            input_z = tf.random.normal(shape=(batch_size, latent_dim))

            
            with tf.GradientTape() as tape: 
                # 真の画像を判定
                d_out_real = discriminator(train_x, training=True)
                # 偽の画像を生成して判別
                fake_img = generator(input_z, training=True)
                d_out_fake = discriminator(fake_img, training=True)
                # 損失を計算
                d_loss = discriminator_loss(d_out_real,d_out_fake) 
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            D_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

            d_loss_obj(d_loss)

            # 2. Generatorの学習
            with tf.GradientTape() as tape: 
                # 偽の画像を生成して判別
                fake_img = generator(input_z, training=True)
                d_out_fake = discriminator(fake_img, training=True)
                # 損失を計算
                g_loss = generator_loss(d_out_fake)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            G_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
            g_loss_obj(g_loss)

            # 記録
            epoch_g_loss += g_loss_obj.result()*batch_size # ミニバッチの損失の合計を加える
            epoch_d_loss += d_loss_obj.result()*batch_size

            # initialize loss object
            g_loss_obj.reset_state()
            d_loss_obj.reset_state()


        epoch_g_loss = epoch_g_loss / 60000 #トレーニングデータのバッチ数で割ります
        epoch_d_loss = epoch_d_loss / 60000 

        print("epoch {} || Epoch_G_Loss: {:.4f} || Epoch_D_Loss: {:.4f}".format(
            epoch+1, epoch_g_loss, epoch_d_loss))

    print("Training Done!")
    # 生成器と識別器を保存
    generator.save('/home/rsato/.vscode-server/data/User/globalStorage/gan-test/Generator')
    discriminator.save('/home/rsato/.vscode-server/data/User/globalStorage/gan-test/NewDis')

    return generator,discriminator

#%%
batch_size = 256
latent_dim = 100
num_epoch = 20 

d_loss_obj = tf.metrics.Mean()
g_loss_obj = tf.metrics.Mean()

# device = ['/device:GPU:0' if tf.test.gpu_device_name() =='/device:GPU:0' else '/cpu:0'][0]
# device = select_gpu()

with tf.device('/gpu:{}'.format(GPU)):
    # MNISTデータをロード
    (x_train, _), (_, _) = mnist.load_data()
    print(x_train.shape)

    # データを-1から1の範囲に正規化
    x_train = (x_train - 127.5) / 127.5
    print(x_train.shape)

    # データの形状を(batch_size, height, width, channels)に変更
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')

    print(x_train.shape)
    # データセットパラメータ
    BUFFER_SIZE = 60000  # データセットのサイズ
    BATCH_SIZE = 512  # バッチサイズ

    # バッチ処理とシャッフルを行うデータセットを作成
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    print(train_dataset)

    # train_model関数の呼び出し
    G_update, D_update = train_model(generator, discriminator, train_dataset, 20, BATCH_SIZE)
