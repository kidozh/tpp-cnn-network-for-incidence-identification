import getopt
import sys

from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from data import *
from model import *

if __name__ == "__main__":
    options, unrecognizedArgs = getopt.getopt(sys.argv[1:], "b:p:v:e:c:s:f:k:",
                                              ["block=", "dropout=", "evaluate", "epoch=", "channel=", "batch=",
                                               "freq=", "phase="])
    print(options, unrecognizedArgs)
    block = 10
    dropout = 0.8
    train = True
    epoch = 10000
    sampling_freq = 1000

    for name, value in options:
        if name in ("-b", "--block"):
            block = int(value)
        elif name in ("-p", "--dropout"):
            dropout = float(value)
        elif name in ("-v", "--evaluate"):
            train = False
        elif name in ("-e", "--epoch"):
            epoch = int(value)
        elif name in ("-c", "--channel"):
            channel = int(value)

    data = Data()

    sample_num_per_stage = 500
    sampling_rate = 1000
    log_dir = 'logs/ResNet_SPP_f_%d_b%d_d%.2f_PS_%d/' % (
        sampling_rate, block, dropout, sample_num_per_stage)

    resnet_block_num = block
    resnet = SppResNetModel()
    model = resnet.build_resnet(5, 5,
                                resnet_block_num=resnet_block_num,
                                dropout=dropout)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_acc{val_acc:.3f}.h5',
                                 monitor='val_acc', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=80, verbose=1)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=100, verbose=1)

    model.fit(multiple_shape_data_generator([1, 2],
                                            [sampling_rate],
                                            [0.128, 0.256],
                                            sample_num_per_stage=sample_num_per_stage),
              steps_per_epoch=27 * 2 * 2,
              validation_data=multiple_shape_data_generator([3],
                                                            [sampling_rate],
                                                            [0.128, 0.256],
                                                            sample_num_per_stage=sample_num_per_stage),
              validation_steps=26 * 2,
              epochs=epoch,
              callbacks=[logging, checkpoint, reduce_lr, early_stopping],
              verbose=2,
              )
