import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from model import build_spp_sequential_model
from data import multiple_shape_data_generator
plt.rcParams['font.sans-serif'] = ['Arial']

SAMPLE_RATE = 1000
SAMPLE_TIME = 0.128
SAMPLE_NUM_PER_STAGE = 500
TOTAL_HOLES = 26

model = build_spp_sequential_model()
model.load_weights("./logs/Sequential_SPP_f_1000_d0.80_PS_500/ep249-loss0.002-val_acc0.996.h5")

print("The model weights are successfully loaded")

cnt = 0
data_list = []
label_list = []

for data, label in multiple_shape_data_generator([3], [SAMPLE_RATE], [SAMPLE_TIME],
                                                 sample_num_per_stage=SAMPLE_NUM_PER_STAGE):
    if cnt >= TOTAL_HOLES:
        break
    cnt += 1
    data_list.append(data)
    label_list.append(label)

signal_array = np.concatenate(data_list, axis=0)
label_array = np.concatenate(label_list, axis=0)

print("Load data successfully with shape", signal_array.shape)

# analysis it
pred_array = model.predict(signal_array)

pred_integer = np.argmax(pred_array, axis=1)
label_integer = np.argmax(label_array, axis=1)

print("Classification report on t", SAMPLE_TIME, "f", SAMPLE_RATE)
print(classification_report(label_integer, pred_integer, digits=4))

drilling_stage_cm = confusion_matrix(label_integer, pred_integer, normalize="true")

print(drilling_stage_cm)

CLASSFICATION_TICKS = ["Engagement", "CFRP", "Transition", "Al", "Disengagement"]

f, ax = plt.subplots()
heatmap = sns.heatmap(drilling_stage_cm, annot=True, ax=ax, cmap="YlGnBu", xticklabels=CLASSFICATION_TICKS,
                          yticklabels=CLASSFICATION_TICKS)  # 画热力图
heatmap.set_xticklabels(CLASSFICATION_TICKS, size=8)
heatmap.set_yticklabels(CLASSFICATION_TICKS, size=8, rotation=90)
# ax.set_title('Normalized confusion matrix')  # 标题
ax.set_xlabel('Prediction')  # x轴
ax.set_ylabel('True')  #
plt.show()
