import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import SpectralNormalization
import torch
import pandas as pd
import numpy as np
import os
import scipy.special as sp

#--------------------------------------Variational Encoder----------------------------------------

class VariationalEncoder(tf.keras.Model):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(192, activation=tf.keras.layers.LeakyReLU()),
            layers.Dense(96, activation=tf.keras.layers.LeakyReLU()),
            layers.Dense(48, activation=tf.keras.layers.LeakyReLU()),
            layers.Dense(32, activation=tf.keras.layers.LeakyReLU())
        ])
        self.mu = layers.Dense(32)
        self.log_var = layers.Dense(32)
    
    def call(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(shape=mu.shape)
        return mu + tf.exp(log_var / 2) * eps
    
    def L_lat(self, z, z_generated):
        """
        潜在符号化損失 L_lat を計算

        :param z: 元の潜在変数
        :param z_generated: 再エンコードされた潜在変数
        """
        # 元の潜在変数と再エンコードされた潜在変数の差のL1ノルムを計算
        return tf.reduce_mean(tf.abs(z - z_generated))
    
    # KL divergence loss（カルバック・ライブラー発散損失）
    def L_KLD(self, real_data_flat):
        """
        カルバック・ライブラー発散損失 L_KLD を計算

        :param mu: 変分エンコーダの出力した平均ベクトル
        :param log_var: 変分エンコーダの出力した対数分散ベクトル
        :return: KL divergence損失のスカラー値
        """
        real_data = tf.reshape(real_data_flat, [real_data_flat.shape[0], 128, 3])
        mu, log_var = self.call(real_data)
        return sp.kl_div(mu, log_var)



#--------------------------------------Discriminator----------------------------------------

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        #正規化のためにSpectralNormalizationを使用
        self.discriminator = tf.keras.Sequential([
            SpectralNormalization(layers.Dense(192, activation=tf.keras.layers.LeakyReLU())),
            SpectralNormalization(layers.Dense(96, activation=tf.keras.layers.LeakyReLU())),
            SpectralNormalization(layers.Dense(48, activation=tf.keras.layers.LeakyReLU())),
            SpectralNormalization(layers.Dense(24, activation=tf.keras.layers.LeakyReLU())),
            SpectralNormalization(layers.Dense(1, activation=tf.keras.layers.LeakyReLU()))
        ])
    
    def call(self, x):
        return self.discriminator(x)
    
    #disc-loss（損失関数）
    def disc_loss(self, fake_data_flat, real_data_flat):
        fake_output = self.discriminator(fake_data_flat, training=True)
        real_output = self.discriminator(real_data_flat, training=True)
        loss_D = -tf.reduce_mean(real_output) + tf.reduce_mean(fake_output)
        return loss_D

    # 特徴抽出用の関数
    def extract_features(self, x):
        features = []
        for layer in self.discriminator.layers[:-1]: # 最後の層を除くすべての層
            x = layer(x)
            features.append(x)
        return features

#--------------------------------------Generator----------------------------------------
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # パディングされた値を無視するようにマスキングレイヤーを追加
        self.bilstm1 = layers.Bidirectional(layers.LSTM(32, return_sequences=True, activation='tanh'), input_shape=(35, 32))
        self.bilstm2 = layers.Bidirectional(layers.LSTM(32, return_sequences=True, activation='tanh'))
        self.bilstm3 = layers.Bidirectional(layers.LSTM(32, return_sequences=True, activation='tanh'))
        self.bilstm4 = layers.Bidirectional(layers.LSTM(32, return_sequences=True, activation='tanh'))
        self.dense = layers.Dense(3, activation='tanh')
    
    def call(self, x):
        x = self.bilstm1(x)
        x = self.bilstm2(x)
        x = self.bilstm3(x)
        x = self.bilstm4(x)
        return self.dense(x)
    
    #gen-loss（損失関数）
    def gen_loss(self, discriminator, encoder, fake_data_flat, real_data_flat, z, z_generated):
        fake_output = discriminator(fake_data_flat, training=True)
        real_output = discriminator(real_data_flat, training=True)
        loss_G = - discriminator.disc_loss(fake_data_flat, real_data_flat)\
                 + lambda_feat * self.L_feat(discriminator, fake_data_flat, real_data_flat) \
                 + lambda_rec * self.L_rec(fake_output, real_output) \
                 + lambda_lat * encoder.L_lat(z, z_generated) \
                 + lambda_KLD * encoder.L_KLD(real_data_flat)
        return loss_G
    
    # 特徴マッチング損失計算
    def L_feat(self, discriminator, fake_output, real_output):
        loss = 0
        fake_features = discriminator.extract_features(fake_output)
        real_features = discriminator.extract_features(real_output)
        for fake_feature, real_feature in zip(fake_features, real_features):
            loss += tf.reduce_mean(tf.abs(fake_feature - real_feature))
        return loss
    
    def L_rec(self, fake_output, real_output):
        return tf.reduce_mean(tf.abs(fake_output - real_output))

    

#--------------------------------------訓練部分----------------------------------------
    
# Discriminatorの更新回数
DISC_UPDATES = 5

# GPUを選択
GPU = 1
#訓練用のパラメータ設定
BATCH_SIZE = 512
#学習率
learning_rate = 0.0002
#ハイパパラメータ
lambda_feat = 1
lambda_rec = 5
lambda_lat = 0.5
lambda_KLD = 0.05

# 訓練ステップの定義
# @tf.function
def train_step(real_data, generator, discriminator, encoder, generator_optimizer, discriminator_optimizer, z, gen_input):
    # print('gen_input:', gen_input.shape)
    real_data_flat = tf.reshape(real_data, [real_data.shape[0], -1])
    
    # Discriminatorの更新
    for _ in range(DISC_UPDATES):
        with tf.GradientTape() as disc_tape:
            fake_data = generator(gen_input, training=True)
            #fake_dataの形状を確認
            # print('fake_data:', fake_data.shape)
            # print('real_data:', real_data.shape)
            fake_data_flat = tf.reshape(fake_data, [fake_data.shape[0], -1])

            # Discriminatorの損失を計算
            disc_loss = discriminator.disc_loss(fake_data_flat, real_data_flat)

            # 勾配を計算し、オプティマイザを使用してDiscriminatorの重みを更新
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    # Generatorの更新
    with tf.GradientTape() as gen_tape:
        # ジェネレータを使用して偽のジェスチャデータを生成
        fake_data = generator(gen_input, training=True)
        fake_data_flat = tf.reshape(fake_data, [fake_data.shape[0], -1])

        # 生成されたジェスチャから潜在コードを再エンコード
        mu_generated, log_var_generated = encoder(fake_data)
        z_generated = encoder.reparameterize(mu_generated, log_var_generated)

        # Generatorの損失を計算
        gen_loss = generator.gen_loss(discriminator, encoder, fake_data_flat, real_data_flat, z, z_generated)

    # Generatorの勾配を計算し、オプティマイザを使用してGeneratorの重みを更新
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    # エンコーダは凍結されているため、勾配計算や更新は行わない
    print('trained')


# mk_train_inputs.pyにて作成されたデータの読み込み
def load_processed_data(load_dir='processed_data'):
    all_real_data = np.load(os.path.join(load_dir, 'all_real_data.npy'))
    all_z = np.load(os.path.join(load_dir, 'all_z.npy'))
    all_generator_inputs = np.load(os.path.join(load_dir, 'all_generator_inputs.npy'))
    return all_real_data, all_z, all_generator_inputs

#--------------------------------------実行部分----------------------------------------
def main():

    # with tf.device('/gpu:{}'.format(GPU)):
    # オプティマイザの設定
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    generator = Generator()
    discriminator = Discriminator()
    encoder = VariationalEncoder()
    # -----------------------Generatorに与えるデータセットを作成-----------------------
    
    # 保存したデータを読み込む
    all_real_data, all_z, all_generator_inputs = load_processed_data()
    # ここで all_real_data はジェスチャーデータのリストです
    # # リスト内の各アイテムを正しい形状にリシェイプする
    # all_real_data_reshaped = [tf.reshape(gesture, (BATCH_SIZE, -1)) for gesture in all_real_data]
    # それからデータセットを作成します
    train_dataset = tf.data.Dataset.from_tensor_slices((
        all_real_data,
        all_generator_inputs,
        all_z
    )).batch(BATCH_SIZE)
    #--------------------------------------訓練部分----------------------------------------
    # 要調整
    EPOCHS = 10
    # 訓練ループ
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for real_data, gen_input, z in train_dataset:
            train_step(real_data, generator, discriminator, encoder, generator_optimizer, discriminator_optimizer, z, gen_input)
    
    # Generatorを保存
    generator.save('/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/Generator')
    # Discriminatorを保存
    discriminator.save('/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/Discriminator')
    # # 仮想のジェネレータ入力データを生成（実際のデータの形状に合わせてください）
    # dummy_gen_input = tf.random.normal([512, 128, 32])
    # # ジェネレータからの出力を取得
    # dummy_gen_output = generator(dummy_gen_input, training=False)
    # # 出力の形状を確認
    # print('Generator output shape:', dummy_gen_output.shape)


if __name__ == "__main__":
    with tf.device('/gpu:{}'.format(GPU)):
        main()  