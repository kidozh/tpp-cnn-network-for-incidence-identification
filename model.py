from keras.layers import *
import keras.backend as K
from keras.models import Model, Sequential
from keras.optimizer_v2.adam import Adam


class SpatialPyramidPooling1D(Layer):
    """Spatial pyramid pooling layer for 1D inputs.
    Refer to Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_list: list of int
            List of pooling regions to use. The length of the list is the number of pooling regions,
            each int in the list is the number of regions in that pool. For example [1,2,4] would be 3
            regions with 1, 2x2 and 4x4 max pools, so 21 outputs per feature map
    # Input shape
        3D tensor with shape:
        `(samples, length, channels)`
    # Output shape
        2D tensor with shape:
        `(samples, channels * sum([i for i in pool_list])`
    """

    def __init__(self, pool_list, **kwargs):
        self.nb_channels = None
        self.pool_list = pool_list
        self.num_outputs_per_channel = sum([i for i in pool_list])
        super(SpatialPyramidPooling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[2]

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.nb_channels * self.num_outputs_per_channel

    def get_config(self):
        config = {'pool_list': self.pool_list}
        base_config = super(SpatialPyramidPooling1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):

        input_shape = K.shape(inputs)
        num_time = input_shape[1]
        time_length = [K.cast(num_time, 'float32') / i for i in self.pool_list]

        outputs = []

        for pool_num, num_pool_regions in enumerate(self.pool_list):
            for i in range(num_pool_regions):
                t1 = i * time_length[pool_num]
                t2 = (i + 1) * time_length[pool_num]
                t1 = K.cast(K.round(t1), 'int32')
                t2 = K.cast(K.round(t2), 'int32')
                # the new segment of signal
                new_shape = [input_shape[0], t2 - t1, input_shape[2]]
                t_crop = inputs[:, t1:t2, :]
                tm = K.reshape(t_crop, new_shape)
                pooled_val = K.max(tm, axis=(1))
                outputs.append(pooled_val)
        # then concatenate it
        outputs = K.concatenate(outputs)

        return outputs


def build_spp_sequential_model():
    model = Sequential()
    model.add(Convolution1D(32, 3, padding='same', input_shape=(None, 5)))
    model.add(Activation('relu'))
    model.add(Convolution1D(32, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Convolution1D(64, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution1D(64, 3))
    model.add(Activation('relu'))
    model.add(SpatialPyramidPooling1D([1, 2, 4]))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                  metrics=['categorical_crossentropy', 'acc'])
    return model

def build_tt_spp_sequential_model():
    model = Sequential()
    model.add(Convolution1D(32, 3, padding='same', input_shape=(None, 2)))
    model.add(Activation('relu'))
    model.add(Convolution1D(32, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Convolution1D(64, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution1D(64, 3))
    model.add(Activation('relu'))
    model.add(SpatialPyramidPooling1D([1, 2, 4]))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(),
                  metrics=['categorical_crossentropy', 'acc'])
    return model

class SppResNetModel:

    def first_block(self, tensor_input, filters, kernel_size=5, pooling_size=1, dropout=0.5):
        out = Conv1D(filters, 3, strides=1, padding='same', use_bias=False)(tensor_input)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Dropout(dropout)(out)
        out = Conv1D(filters, kernel_size, strides=1, padding='same', use_bias=False)(out)

        pooling = MaxPooling1D(pooling_size, strides=1, padding='same')(tensor_input)

        out = add([out, pooling])
        return out

    def resnet_block(self, input, filters: list, kernel_size=3, pooling_size=2, dropout=0.0, change_dim=False):
        filter1, filter2, filter3 = filters
        out = Conv1D(filter1, kernel_size, padding='same', strides=1, use_bias=False)(input)
        out = BatchNormalization()(out)
        out = LeakyReLU(alpha=0.1)(out)
        out = Dropout(dropout)(out)
        out = Conv1D(filter2, kernel_size, padding='same', strides=1, use_bias=False)(out)
        out = BatchNormalization()(out)
        out = LeakyReLU(alpha=0.1)(out)
        out = Dropout(dropout)(out)
        if change_dim:
            out = Conv1D(filter3, kernel_size, padding='same', strides=2, use_bias=False)(out)
        else:
            out = Conv1D(filter3, kernel_size, padding='same', use_bias=False)(out)
        out = BatchNormalization()(out)
        out = LeakyReLU(alpha=0.1)(out)
        if change_dim:
            shortcut = Conv1D(filter3, kernel_size, padding='same', use_bias=False, strides=2)(input)
            shortcut = BatchNormalization()(shortcut)
            shortcut = LeakyReLU(alpha=0.1)(shortcut)
            pooling = shortcut
            # pooling = MaxPooling1D(pooling_size, padding='same', strides=2)(input)
        else:
            # shortcut = Conv1D(filter3, kernel_size, padding='same', use_bias=False)(input)
            # shortcut = BatchNormalization()(shortcut)
            # shortcut = LeakyReLU(alpha=0.1)(shortcut)
            pooling = input
            # pooling = MaxPooling1D(pooling_size, padding='same', strides=1)(input)

        out = add([out, pooling])
        return out

    def build_resnet(self, input_dim: int, output_dim: int, resnet_block_num=20, dropout=0.5):
        inp = Input(shape=(None, input_dim))
        # change them according to channel

        out = Conv1D(32, 3, strides=1, padding="same", use_bias=False)(inp)
        out = BatchNormalization()(out)
        out = LeakyReLU(alpha=0.1)(out)

        out = self.first_block(out, 32, dropout=dropout)

        CHANGE_LAYER = 5
        for blockIndex in range(resnet_block_num):
            filter = 32 * 2 ** (blockIndex // CHANGE_LAYER)
            # only allow 128 filters
            filter = min(filter, 64)
            if blockIndex >= 10:
                kernel_size = 3
            else:
                kernel_size = 3
            # print("filter", blockIndex, filter)
            if blockIndex % CHANGE_LAYER == 0:
                out = self.resnet_block(out, [filter, filter, filter], kernel_size=3, change_dim=True, dropout=dropout)
            else:
                out = self.resnet_block(out, [filter, filter, filter], kernel_size=kernel_size, change_dim=False, dropout=dropout)
        # replace dense with spatial pyramid pooling to get unified result
        out = SpatialPyramidPooling1D([1, 2, 4])(out)
        # out = Flatten()(out)
        out = Dense(output_dim, activation="softmax")(out)

        model = Model(inputs=[inp], outputs=[out])

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4),
                      metrics=['categorical_crossentropy', 'acc'])
        return model

if __name__ == "__main__":
    sampling_duration = 0.128
    sampling_rate = 1000
    resnet_block_num = 15
    dropout = 0.2
    resnet = SppResNetModel()
    model = resnet.build_resnet(5, 5,
                                resnet_block_num=resnet_block_num,
                                dropout=dropout)

    print(model.summary())
