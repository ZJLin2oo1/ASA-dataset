import keras
import tensorflow as tf
from keras import Sequential, optimizers, losses

# setting
time_len = 1
sample_len, channels_num = int(128 * time_len), 64
lr = 5e-3

def create_model(sample_len=128, channels_num=64, lr=5e-3, is_attention=False):
    # set the model
    model = Sequential()

    model.add(keras.layers.Input(shape=(sample_len, channels_num)))
    model.add(keras.layers.Reshape((sample_len, channels_num, 1)))
    model.add(keras.layers.Conv2D(filters=5, kernel_size=(17, channels_num), activation='relu'))
    model.add(keras.layers.Reshape(target_shape=(sample_len - 16, 5)))
    model.add(keras.layers.GlobalAveragePooling1D())

    # fully connected classifier
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(5, activation='sigmoid'))
    model.add(keras.layers.Dense(2, activation='softmax'))

    # set the optimizers
    model.compile(
        optimizer=optimizers.legacy.Adam(lr),
        loss=losses.CategoricalCrossentropy(),
        metrics=['accuracy'],
    )
    model.summary()
    return model

def main():
    model = create_model()
    random_data = tf.ones((16, sample_len, channels_num, lr))
    model(random_data)
    model.summary()
    del model

if __name__ == '__main__':
    main()