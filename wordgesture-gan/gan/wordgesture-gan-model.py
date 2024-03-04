import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow_addons.layers import SpectralNormalization
import torch

# Encoder
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


    def L_lat(self, real_data, generator):
        """
        潜在符号化損失 L_lat を計算

        :param real_data: 実データ
        :param generator: Generatorクラスのインスタンス
        :return: L1ノルムに基づく潜在符号化損失のスカラー値
        """
        # 実データから潜在変数の平均と対数分散をエンコード
        mu, log_var = self.call(real_data)
        # Reparameterization trickを使用して潜在変数をサンプリング
        z = self.reparameterize(mu, log_var)
        # Generatorを使用してサンプリングした潜在変数からデータを生成
        generated_data = generator(z)
        # 生成されたデータを再エンコードして潜在変数を取得
        mu_generated, _ = self.call(generated_data)
        # 元の潜在変数と再エンコードされた潜在変数の差のL1ノルムを計算
        return tf.reduce_mean(tf.abs(z - mu_generated))

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trickを使って、muとlog_varから潜在変数zをサンプリング
        """
        eps = tf.random.normal(shape=tf.shape(mu))
        return eps * tf.exp(log_var * .5) + mu
    
    # KL divergence loss（カルバック・ライブラー発散損失）
    def L_KLD(self, real_data):
        """
        カルバック・ライブラー発散損失 L_KLD を計算

        :param mu: 変分エンコーダの出力した平均ベクトル
        :param log_var: 変分エンコーダの出力した対数分散ベクトル
        :return: KL divergence損失のスカラー値
        """
        mu, log_var = self.call(real_data)
        return -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))



# Discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = tf.keras.Sequential([
            SpectralNormalization(192, activation=tf.keras.layers.LeakyReLU()),
            SpectralNormalization(96, activation=tf.keras.layers.LeakyReLU()),
            SpectralNormalization(48, activation=tf.keras.layers.LeakyReLU()),
            SpectralNormalization(24, activation=tf.keras.layers.LeakyReLU()),
            SpectralNormalization(1, activation=tf.keras.layers.LeakyReLU())
        ])
    
    def call(self, x):
        return self.discriminator(x)
    
    #disc-loss（損失関数）
    def disc_loss(self, real_output, fake_output):
        loss_D = -tf.reduce_mean(self.discriminator(real_output)) + tf.reduce_mean(self.discriminator(fake_output))
        return loss_D

    # 特徴抽出用の関数
    def extract_features(self, x):
        features = []
        for layer in self.discriminator.layers[:-1]: # 最後の層を除くすべての層
            x = layer(x)
            features.append(x)
        return features

# Generator
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.bilstm1 = layers.Bidirectional(layers.LSTM(32, return_sequences=True, activation='tanh'), input_shape=(35, 32))
        self.bilstm2 = layers.Bidirectional(layers.LSTM(32, return_sequences=True, activation='tanh'))
        self.bilstm3 = layers.Bidirectional(layers.LSTM(32, return_sequences=True, activation='tanh'))
        self.bilstm4 = layers.Bidirectional(layers.LSTM(32, activation='tanh'))
        self.dense = layers.Dense(3, activation='tanh')
    
    def call(self, x):
        x = self.bilstm1(x)
        x = self.bilstm2(x)
        x = self.bilstm3(x)
        x = self.bilstm4(x)
        return self.dense(x)
    
    #gen-loss（損失関数）
    def gen_loss(self, fake_output, real_output, lambda_feat, lambda_rec, lambda_lat, lambda_KLD):
        fake_features = Discriminator.extract_features(fake_output)
        real_features = Discriminator.extract_features(real_output)
        loss_G = - Discriminator.disc_loss(fake_output, real_output)\
                 + lambda_feat * self.L_feat(fake_features, real_features) \
                 + lambda_rec * self.L_rec(fake_output, real_output) \
                 + lambda_lat * VariationalEncoder.L_lat(real_output, self()) \
                 + lambda_KLD * VariationalEncoder.L_KLD(real_output)
        return loss_G
    
    # 特徴マッチング損失計算
    def L_feat(self, fake_features, real_features):
        loss = 0
        for fake_feature, real_feature in zip(fake_features, real_features):
            loss += tf.reduce_mean(tf.abs(fake_feature - real_feature))
        return loss
    
    def L_rec(self, fake_output, real_output):
        return tf.reduce_mean(tf.abs(fake_output - real_output))


#------------------訓練部分--------------------

# 訓練ステップの定義
@tf.function
def train_step(real_data):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generatorを使用して偽のデータを生成
        fake_data = generator(tf.random.normal([BATCH_SIZE, 100]))

        # Discriminatorを使用して本物と偽物のデータを評価
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(fake_data, training=True)

        # 損失を計算
        gen_loss = generator.gen_loss(fake_output, real_output, lambda_feat, lambda_rec, lambda_lat, lambda_KLD)
        disc_loss = discriminator.disc_loss(real_output, fake_output)

    # 勾配を計算し、オプティマイザを使用してモデルの重みを更新
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# 訓練ループ
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)



#------------------実行部分--------------------
# GPUを選択
GPU = 1

BATCH_SIZE = 512
learning_rate = 0.0002
#パラメータ設定
lambda_feat = 1
lambda_rec = 5
lambda_lat = 0.5
lambda_KLD = 0.05

with tf.device('/gpu:{}'.format(GPU)):
    # オプティマイザの設定
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)

    generator = Generator()
    discriminator = Discriminator()
    encoder = VariationalEncoder()
    # データセットの準備と訓練の実行
    # dataset = 
    # 要調整
    # epochs = 50
    # train(dataset, epochs)



    