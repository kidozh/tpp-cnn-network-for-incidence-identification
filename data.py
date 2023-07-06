import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import scipy.signal
from scipy.fftpack import rfft, irfft, fftfreq, fft, ifft

from utils import RAW_SAMPLE_RATE

parent_dir = r"C:\PythonProject\Philip-experiment"

SEGMENT_DATA_DIR = os.path.join(parent_dir, "data_auto_num")

XLSX_PATH = os.path.join(parent_dir, "./philip-data.xlsx")

BASE_SPINDLE_RATE = 4000 / 60


def get_cfrp_thickness(tool_index: int, hole_index: int) -> float:
    """
    get the thickness of CFRP
    :param tool_index: tool index
    :param hole_index: hole index
    :return: thickness of CFRP (mm)
    """
    data = pd.read_excel(XLSX_PATH, "CFRP")
    assert tool_index in [1, 2, 3]
    assert hole_index in [i for i in range(1, 28)]
    tool_label = "T%d" % tool_index
    return data[tool_label][hole_index - 1]


def get_al_thickness(tool_index: int, hole_index: int) -> float:
    """
    get the thickness of Al
    :param tool_index: tool index
    :param hole_index: hole index
    :return: thickness of Al (mm)
    """
    data = pd.read_excel(XLSX_PATH, "Al")

    assert tool_index in [1, 2, 3]
    assert hole_index in [i for i in range(1, 28)]
    tool_label = "T%d" % tool_index
    return data[tool_label][hole_index - 1]


def get_drilling_start_second(tool_index: int, hole_index: int) -> float:
    """
    get the start second of drilling process
    :param tool_index: tool index
    :param hole_index: hole index
    :return: start second (s)
    """
    data = pd.read_excel(XLSX_PATH, "start_sec")

    assert tool_index in [1, 2, 3]
    assert hole_index in [i for i in range(1, 28)]
    tool_label = "T%d" % tool_index

    return data[tool_label][hole_index - 1]


def get_drilling_end_second(tool_index: int, hole_index: int) -> float:
    """
    get the end second of drilling process
    :param tool_index: tool index
    :param hole_index: hole index
    :return: start second (s)
    """
    data = pd.read_excel(XLSX_PATH, "end_sec")

    assert tool_index in [1, 2, 3]
    assert hole_index in [i for i in range(1, 28)]
    tool_label = "T%d" % tool_index

    return data[tool_label][hole_index - 1]


def get_tool_wear(tool_index: int, hole_index: int) -> float:
    """
    get the end second of drilling process
    :param tool_index: tool index
    :param hole_index: hole index
    :return: start second (s)
    """
    XLSX_PATH = os.path.join(parent_dir, "./philip-data-new.xlsx")
    data = pd.read_excel(XLSX_PATH, "tool-wear-hole")

    assert tool_index in [1, 2, 3]
    assert hole_index in [i for i in range(1, 28)]
    tool_label = "T%d" % tool_index

    return data[tool_label][hole_index - 1]


def get_tool_wear_corrected(tool_index: int, hole_index: int) -> float:
    """
    get the end second of drilling process
    :param tool_index: tool index
    :param hole_index: hole index
    :return: start second (s)
    """
    XLSX_PATH = os.path.join(parent_dir, "./philip-data-new.xlsx")
    data = pd.read_excel(XLSX_PATH, "Tool_wear_rows_real")

    assert tool_index in [1, 2, 3]
    assert hole_index in [i for i in range(1, 28)]
    tool_label = "T%d" % tool_index

    return data[tool_label][(hole_index - 1) // 3]


def get_drilling_duration(thickness: float, feed_rate=200) -> float:
    """
    get the drilling duration
    :param thickness: thickness of one layer
    :param feed_rate: feed rate (mm/min)
    :return: the traverse second (s)
    """
    return thickness / (feed_rate / 60)


def get_further_drill_time(tool_index: int) -> float:
    # no further milling time
    if tool_index == 1:
        return 0
    else:
        return get_drilling_duration(2.00)


CONE_TRAVERSE_SECOND = get_drilling_duration(2.403)


class Data:

    def get_segment_data_path(self, tool_index: int, hole_index: int) -> str:
        assert tool_index in [1, 2, 3]
        assert hole_index in [i for i in range(1, 28)]

        return os.path.join(SEGMENT_DATA_DIR, "T%dH%d.csv" % (tool_index + 1, hole_index))

    def get_segment_data(self, tool_index: int, hole_index: int) -> pd.DataFrame:
        return pd.read_csv(self.get_segment_data_path(tool_index, hole_index))

    def get_keyframe_sec(self, tool_index: int, hole_index: int) -> tuple:
        assert tool_index in [1, 2, 3]
        assert hole_index in [i for i in range(1, 28)]

        cfrp_thickness = get_cfrp_thickness(tool_index, hole_index)
        al_thickness = get_al_thickness(tool_index, hole_index)
        cfrp_travel_sec = get_drilling_duration(cfrp_thickness)
        al_travel_sec = get_drilling_duration(al_thickness)
        # time point for each incidence
        # start_sec - engagement_end => engagement
        # engagement_end - cfrp_end => CFRP
        # cfrp_end - transition_end => Transition
        # transition_end - Al_end => Al
        # Al_end - final_end => Disengagement
        engagement_end = 0 + CONE_TRAVERSE_SECOND
        cfrp_end = 0 + cfrp_travel_sec
        transition_end = cfrp_end + CONE_TRAVERSE_SECOND
        al_end = cfrp_end + al_travel_sec
        return engagement_end, cfrp_end, transition_end, al_end

    def get_signal_segment(self, tool_index: int, hole_index: int) -> tuple:
        dataframe = self.get_segment_data(tool_index, hole_index)
        multi_channel_signal = dataframe.to_numpy()[:, 1:6]
        engagement_end_sec, cfrp_end_sec, transition_end_sec, al_end_sec = self.get_keyframe_sec(tool_index, hole_index)
        engagement_end_index = int(engagement_end_sec * RAW_SAMPLE_RATE)
        cfrp_end_index = int(cfrp_end_sec * RAW_SAMPLE_RATE)
        transition_end_index = int(transition_end_sec * RAW_SAMPLE_RATE)
        al_end_index = int(al_end_sec * RAW_SAMPLE_RATE)
        engagement_signal = multi_channel_signal[:engagement_end_index, :]
        cfrp_signal = multi_channel_signal[engagement_end_index:cfrp_end_index, :]
        transition_signal = multi_channel_signal[cfrp_end_index:transition_end_index, :]
        al_signal = multi_channel_signal[transition_end_index: al_end_index, :]
        disengagement_signal = multi_channel_signal[al_end_index:, :]

        return engagement_signal, cfrp_signal, transition_signal, al_signal, disengagement_signal


def get_random_samples(signal: np.ndarray, sample_num_per_stage:int=500, sampling_rate:int=1000, sampling_duration:float=0.128):
    template_sampling_sequence = np.arange(0, int(sampling_duration * RAW_SAMPLE_RATE),
                                           step=RAW_SAMPLE_RATE // sampling_rate)
    # print("Template sequence", template_sampling_sequence)
    end_sec = signal.shape[0] / RAW_SAMPLE_RATE - sampling_duration

    end_idx = int(end_sec * RAW_SAMPLE_RATE) - 1
    # generate random number array
    # print(template_sampling_sequence.shape,end_sec, end_idx)
    random_start_index_array = np.random.randint(0, end_idx, size=sample_num_per_stage)
    # shift it in random size
    random_sample_array = [start_index + template_sampling_sequence for start_index in random_start_index_array]
    return signal[random_sample_array, :]


def get_random_samples_v2(signal: np.ndarray, sample_num_per_stage=500, sampling_rate=1000, sampling_duration=0.128):
    template_sampling_sequence = np.arange(0, int(sampling_duration * RAW_SAMPLE_RATE),
                                           step=RAW_SAMPLE_RATE // sampling_rate)
    template_sampling_sequence = template_sampling_sequence[:int(sampling_rate * sampling_duration)]
    # print("Template sequence", template_sampling_sequence)
    end_sec = signal.shape[0] / RAW_SAMPLE_RATE - sampling_duration
    end_idx = int(end_sec * RAW_SAMPLE_RATE) - 1
    # generate random number array
    # print(template_sampling_sequence.shape,end_sec, end_idx)
    random_start_index_array = np.random.randint(0, end_idx, size=sample_num_per_stage)
    # shift it in random size
    random_sample_array = [start_index + template_sampling_sequence for start_index in random_start_index_array]
    return signal[random_sample_array, :]


def multiple_shape_data_generator(tool_index_list: list, sample_rate_list: list, sample_duration_list: list, sample_num_per_stage:int=1000):

    data = Data()
    while True:
        for tool_index in tool_index_list:
            for hole_index in range(1, 28):

                if tool_index == 3 and hole_index == 22:
                    continue
                for sample_rate in sample_rate_list:
                    for sample_duration in sample_duration_list:
                        signal_array = np.zeros([sample_num_per_stage * 5, int(sample_rate * sample_duration), 5])
                        label_array = np.zeros([sample_num_per_stage * 5, 5])
                        if tool_index == 3 and hole_index == 22:
                            continue
                        signal_segment_list = data.get_signal_segment(
                            tool_index, hole_index)
                        for index, signal in enumerate(signal_segment_list):
                            signal_sample = get_random_samples(signal[...],
                                                               sample_num_per_stage=sample_num_per_stage,
                                                               sampling_rate=sample_rate,
                                                               sampling_duration=sample_duration)
                            signal_array[index * sample_num_per_stage:(index + 1) * sample_num_per_stage, :,
                            :] = signal_sample[:,:int(sample_rate * sample_duration),:]
                            label_array[index * sample_num_per_stage:(index + 1) * sample_num_per_stage, index] = 1
                        # print(signal_array.shape, label_array.shape)
                        yield signal_array, label_array

def multiple_shape_tt_data_generator(tool_index_list: list, sample_rate_list: list, sample_duration_list: list, sample_num_per_stage:int=1000):

    data = Data()
    while True:
        for tool_index in tool_index_list:
            for hole_index in range(1, 28):

                if tool_index == 3 and hole_index == 22:
                    continue
                for sample_rate in sample_rate_list:
                    for sample_duration in sample_duration_list:
                        signal_array = np.zeros([sample_num_per_stage * 5, int(sample_rate * sample_duration), 2])
                        label_array = np.zeros([sample_num_per_stage * 5, 5])
                        if tool_index == 3 and hole_index == 22:
                            continue
                        signal_segment_list = data.get_signal_segment(
                            tool_index, hole_index)
                        for index, signal in enumerate(signal_segment_list):
                            signal_sample = get_random_samples(signal[..., :2],
                                                               sample_num_per_stage=sample_num_per_stage,
                                                               sampling_rate=sample_rate,
                                                               sampling_duration=sample_duration)
                            signal_array[index * sample_num_per_stage:(index + 1) * sample_num_per_stage, :,
                            :] = signal_sample[:,:int(sample_rate * sample_duration),:]
                            label_array[index * sample_num_per_stage:(index + 1) * sample_num_per_stage, index] = 1
                        # print(signal_array.shape, label_array.shape)
                        yield signal_array, label_array


if __name__ == "__main__":
    data = Data()
    for signal, label in multiple_shape_data_generator([1, 2],
                                            [1000],
                                            [0.128],
                                            sample_num_per_stage=1000):
        print(signal.shape, label.shape)
