import tensorflow as tf
from keras import layers, models, optimizers, losses

# setting
time_len = 1
sample_len, channels_num = int(128 * time_len), 64
lr = 5e-3

class Channel_Attentioin(layers.Layer):
    def __init__(self, ratio=2, **kwargs):
        super(Channel_Attentioin, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.se_reduce = layers.Dense(input_shape[-1] // self.ratio, activation=tf.nn.leaky_relu)
        self.se_expand = layers.Dense(input_shape[-1], activation='sigmoid')
    
    def call(self, inputs):
        # Squeeze: Global Average Pooling
        squeeze = tf.reduce_mean(inputs, axis=-2, keepdims=True)
        
        # Excitation: Fully Connected layers
        excitation = self.se_reduce(squeeze)
        excitation = self.se_expand(excitation)
        
        # Scale
        scale = inputs * excitation
        return scale


def create_model(sample_len, channels_num, lr, is_attention=False):
    model = models.Sequential()
    model.add(layers.Input(shape=(sample_len, channels_num)))

    model.add(layers.Conv1D(filters=16, kernel_size=3, activation=tf.nn.leaky_relu, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=16, kernel_size=3, activation=tf.nn.leaky_relu, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation=tf.nn.leaky_relu, padding='same'))
    model.add(layers.BatchNormalization())
    
    # Add channel attention layer here
    if is_attention == True:
        model.add(Channel_Attentioin())

    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(
        optimizer=optimizers.legacy.Adam(lr),
        loss=losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    model.summary()
    return model


if __name__ == '__main__':
    model = create_model(channels_num=64, sample_len=128, lr=5e-4, is_attention=False)

        
def main():
    model = create_model()
    random_data = tf.ones((16, sample_len, channels_num, lr))
    model(random_data)
    model.summary()
    del model