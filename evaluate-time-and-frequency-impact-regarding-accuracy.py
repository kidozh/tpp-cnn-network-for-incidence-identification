import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

from model import build_spp_sequential_model
from data import multiple_shape_data_generator

plt.rcParams['font.sans-serif'] = ['Arial']

SAMPLE_RATE = 1000
SAMPLE_TIME = 0.1
SAMPLE_NUM_PER_STAGE = 500
TOTAL_HOLES = 26

model = build_spp_sequential_model()
model.load_weights("./logs/Sequential_SPP_f_1000_d0.80_PS_500/ep249-loss0.002-val_acc0.996.h5")

print("The model weights are successfully loaded")

# freq_list = [100, 125, 200, 250, 400, 500, 800, 1000, 1250, 2000]
freq_list = [i*100 for i in range(1, 21)]
time_list = [0.05*i for i in range(1, 11)]
for time in time_list:
    for freq in freq_list:
        if time * freq < 10:
            continue
        cnt = 0
        data_list = []
        label_list = []
        for data, label in multiple_shape_data_generator([3], [freq], [time],
                                                         sample_num_per_stage=SAMPLE_NUM_PER_STAGE):
            if cnt >= TOTAL_HOLES:
                break
            cnt += 1
            data_list.append(data)
            label_list.append(label)

        signal_array = np.concatenate(data_list, axis=0)
        label_array = np.concatenate(label_list, axis=0)

        # print("Load data successfully with shape", freq, signal_array.shape)

        # analysis it
        pred_array = model.predict(signal_array, verbose=0)

        pred_integer = np.argmax(pred_array, axis=1)
        label_integer = np.argmax(label_array, axis=1)
        print(time, "\t", freq, "\t", accuracy_score(label_integer, pred_integer))
