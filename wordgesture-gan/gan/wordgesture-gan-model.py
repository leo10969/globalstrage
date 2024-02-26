import tensorflow as tf
from tensorflow.keras import layers, Model

# GPUを選択
GPU = 1

#Encoder
class VariationalEncoder(tf.keras.Model):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(192, activation='leaky_relu'),
            layers.Dense(96, activation='leaky_relu'),
            layers.Dense(48, activation='leaky_relu'),
            layers.Dense(32, activation='leaky_relu')
        ])
        self.mu = layers.Dense(32)
        self.log_var = layers.Dense(32)
    
    def call(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

#Generator
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # 最初の層にのみinput_shapeを指定
        self.bilstm1 = layers.Bidirectional(layers.LSTM(32, return_sequences=True), input_shape=(35, 32))
        self.bilstm2 = layers.Bidirectional(layers.LSTM(32, return_sequences=True))
        self.bilstm3 = layers.Bidirectional(layers.LSTM(32, return_sequences=True))
        self.bilstm4 = layers.Bidirectional(layers.LSTM(32))
        self.dense = layers.Dense(3, activation='tanh')
    
    def call(self, x):
        x = self.bilstm1(x)
        x = self.bilstm2(x)
        x = self.bilstm3(x)
        x = self.bilstm4(x)
        return self.dense(x)


#Discriminator
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = tf.keras.Sequential([
            layers.Dense(192, activation='leaky_relu'),
            layers.Dense(96, activation='leaky_relu'),
            layers.Dense(48, activation='leaky_relu'),
            layers.Dense(24, activation='leaky_relu'),
            layers.Dense(1, activation='leaky_relu')
        ])
    
    def call(self, x):
        return self.discriminator(x)

with tf.device('/gpu:{}'.format(GPU)):
    generator = Generator()
    discriminator = Discriminator()
    encoder = VariationalEncoder()