import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import SpectralNormalization
import torch
import pandas as pd

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


    def L_lat(self, z, z_generated):
        """
        潜在符号化損失 L_lat を計算

        :param z: 元の潜在変数
        :param z_generated: 再エンコードされた潜在変数
        """
        # 元の潜在変数と再エンコードされた潜在変数の差のL1ノルムを計算
        return tf.reduce_mean(tf.abs(z - z_generated))
    
    # KL divergence loss（カルバック・ライブラー発散損失）
    def L_KLD(self, real_output):
        """
        カルバック・ライブラー発散損失 L_KLD を計算

        :param mu: 変分エンコーダの出力した平均ベクトル
        :param log_var: 変分エンコーダの出力した対数分散ベクトル
        :return: KL divergence損失のスカラー値
        """
        mu, log_var = self.call(real_output)
        return -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))



# Discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        #正規化のためにSpectralNormalizationを使用
        self.discriminator = tf.keras.Sequential([
            SpectralNormalization(layers.Dense(192, activation='leaky_relu', input_shape=(384,))),
            SpectralNormalization(layers.Dense(96, activation='leaky_relu')),
            SpectralNormalization(layers.Dense(48, activation='leaky_relu')),
            SpectralNormalization(layers.Dense(24, activation='leaky_relu')),
            SpectralNormalization(layers.Dense(1, activation='leaky_relu'))
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
    def gen_loss(self, fake_output, real_output, mu, var_log, z, z_generated):
        fake_features = Discriminator.extract_features(fake_output)
        real_features = Discriminator.extract_features(real_output)
        loss_G = - Discriminator.disc_loss(fake_output, real_output)\
                 + lambda_feat * self.L_feat(fake_features, real_features) \
                 + lambda_rec * self.L_rec(fake_output, real_output) \
                 + lambda_lat * VariationalEncoder.L_lat(z, z_generated) \
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
def get_generator_input(word_prototype, z):
    """
    Word Prototypeとガウシアンノイズzを組み合わせてジェネレータの入力を得る。
    :param word_prototype: テキストから得られるワードプロトタイプ。
    :param z: ヴァリエーショナルエンコーダから得られるガウシアンノイズ。
    :return: ジェネレータの入力として使うための結合されたテンソル。
    """
    # ガウシアンノイズzの形状をプロトタイプの形状に合わせて拡張する。
    z_expanded = tf.repeat(tf.reshape(z, [1, -1]), repeats=word_prototype.shape[0], axis=0)
    # 拡張したノイズとワードプロトタイプを結合する。
    generator_input = tf.concat([word_prototype, z_expanded], axis=1)
    return generator_input
    
# Discriminatorの更新回数
DISC_UPDATES = 5

# 訓練ステップの定義
@tf.function
def train_step(real_data, generator, discriminator, encoder, generator_optimizer, discriminator_optimizer, gen_input):
    # Discriminatorの更新
    for _ in range(DISC_UPDATES):
        with tf.GradientTape() as disc_tape:
            # 実データから潜在コードを生成
            mu, log_var = encoder(real_data)
            z = encoder.reparameterize(mu, log_var)
            fake_data = generator(z)
            
            # Discriminatorを使用して本物と偽物のデータを評価
            real_output = discriminator(real_data, training=True)
            fake_output = discriminator(fake_data, training=True)

            # Discriminatorの損失を計算
            disc_loss = discriminator.disc_loss(real_output, fake_output)

            # 勾配を計算し、オプティマイザを使用してDiscriminatorの重みを更新
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    # Generatorの更新
    with tf.GradientTape() as gen_tape:
        # ジェネレータを使用して偽のデータを生成
        fake_data = generator(gen_input)
        
        # Discriminatorを使用して偽物のデータを評価
        fake_output = discriminator(fake_data, training=True)

        
        # 生成されたジェスチャーから潜在コードを再エンコード
        mu_generated, log_var_generated = encoder(fake_data)
        z_generated = encoder.reparameterize(mu_generated, log_var_generated)

        # Generatorの損失を計算
        gen_loss = generator.gen_loss(fake_output, real_output, z, z_generated)


    # Generatorの勾配を計算し、オプティマイザを使用してGeneratorの重みを更新
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    
    # エンコーダは凍結されているため、勾配計算や更新は行わない


#--------------------------------------実行部分----------------------------------------
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


    #--------------word_prototypeの準備--------------
    # 各word_prototype CSVを読み込む
    with open('train_data_wordlist.txt', 'r') as f:
        train_words_list = f.read().splitlines()
    word_prototypes = {}
    for word in train_words_list:  # words_listは訓練する単語のリスト
        word_prototypes[word] = pd.read_csv(f'/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/prototype/prototype_csv/{word}.csv')


    # ジェネレータへの入力として使用するために、numpy配列に変換する
    word_prototype = word_prototype_df[['x_pos', 'y_pos']].to_numpy()

    # ワードプロトタイプの形状を確認（この例では (128, 2) であるべき）
    print(word_prototype.shape)

    # TensorFlowデータセットを作成する
    # ここではword_prototypeがすでに適切な形状（例: (128, 2)）を持っていると仮定している
    # 実際の単語ごとのデータセットを持っている場合は、それぞれの単語についてこのプロセスを繰り返す
    word_prototype_dataset = tf.data.Dataset.from_tensor_slices(word_prototype)

    # データセットの各要素に適用する処理を定義（必要に応じて）
    def preprocess_prototype(prototype):
        # ここで任意の前処理を行う。例えば、座標の正規化、形状の変更、次元の追加など
        # ここでは単純に形状を (128, 2) から (128, 3) に変更するダミーの処理を行っている
        prototype = tf.concat([prototype, tf.zeros((128, 1))], axis=-1)  # z_posを追加
        return prototype

    # データセットに前処理を適用
    word_prototype_dataset = word_prototype_dataset.map(preprocess_prototype)

    # バッチ処理を行う
    # バッチサイズはプログラムで指定した512などにする
    word_prototype_dataset = word_prototype_dataset.batch(BATCH_SIZE)

    # データセットが正しくバッチ処理されているかを確認
    for batch in word_prototype_dataset.take(1):
        print(batch.shape)  # 出力は (512, 128, 3) などの形状になる


    # -----------------------ユーザの描いたジェスチャデータセットの準備-----------------------
    # ユーザの描いたジェスチャデータセットを読み込む
    dataset = 
    # 仮のデータを用いて関数をテストする。
    # word_prototype = tf.random.normal([128, 3])  # 128ステップのワードプロトタイプ
    # z = tf.random.normal([32])                   # 32次元のガウシアンノイズ
        
    # 実データから潜在コードを生成
    mu, log_var = encoder(dataset, training=False)  # エンコーダは推論モードで動作する
    z = encoder.reparameterize(mu, log_var)

    # -----------------------準備を経て，Generatorに与えるデータセットを作成-----------------------
    # ジェネレータの入力を取得する。
    gen_gen_input = get_generator_input(word_prototype, z)

    print(gen_input.shape)  # get_gen_inputは、(128, 35)の次元を持つ

    #--------------------------------------訓練部分----------------------------------------
    # 要調整
    epochs = 50

    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch, generator, discriminator, encoder, generator_optimizer, discriminator_optimizer)
        print('Epoch {} finished'.format(epoch))
    
    # Generatorを保存
    generator.save('/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/Generator')
    # Discriminatorを保存
    discriminator.save('/home/rsato/.vscode-server/data/User/globalStorage/wordgesture-gan/Discriminator')




    