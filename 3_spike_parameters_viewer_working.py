#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated on Fri Apr  4 15:36:35 2025

@author: gs075
"""

import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyabf
import os
import re

# Path to the CSV file where spike parameters are already calculated
spike_parameters_csv_path = root = '/Users/gs075/Desktop/test/2_all_parameters.csv'

sample_rate = 10000
highlight_window = 10  # ms
timescale_start = 0.25  # s
timescale_end = 1.25    # s
display_window = 60
LJP_CORRECTION_MV = 14.681  # Subtract from voltage to correct for liquid junction potential

spike_params_df = pd.read_csv(spike_parameters_csv_path)
unique_recordings = spike_params_df['recording'].unique()
df = spike_params_df
display_window = int(display_window * sample_rate / 1000)

def clean_timestamps(timestamp):
    if isinstance(timestamp, str):
        timestamp = [
            float(re.sub(r'np\.float64\((.*?)\)', r'\1', t).strip()) if t.strip() != '' else None
            for t in timestamp.strip('[]').split(',')
        ]
        timestamp = [t for t in timestamp if t is not None]
    elif isinstance(timestamp, float):
        timestamp = [timestamp]
    return timestamp

def calculate_steady_state_voltage(time, voltage, timescale_start, timescale_end, sample_rate):
    start_idx = np.abs(time - timescale_start).argmin()
    end_idx = np.abs(time - timescale_end).argmin()
    return np.mean(voltage[start_idx:end_idx])

spike_params_df['timestamp'] = spike_params_df['timestamp'].apply(clean_timestamps)

current_recording_index = 0
current_sweep_index = 0
current_peak_index = 0
last_steady_state_voltage = None

fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 0.5])
ax_full = fig.add_subplot(gs[0])
ax_zoom = fig.add_subplot(gs[1])

def update_plot():
    global current_recording_index, current_sweep_index, current_peak_index, last_steady_state_voltage

    recording = unique_recordings[current_recording_index]
    record_data = df[df['recording'] == recording]
    if current_sweep_index >= len(record_data):
        print("Sweep index out of range")
        return

    sweep_row = record_data.iloc[current_sweep_index]
    sweep_number = sweep_row['sweep_number']
    peak_times = sweep_row['timestamp']

    if len(peak_times) == 0:
        print(f"No peaks for recording {recording}, sweep {sweep_number}. Skipping this sweep.")
        return

    peak_time = float(peak_times[current_peak_index])

    abf = pyabf.ABF(os.path.join('/Users/gs075/Desktop/steps_analyses', recording))
    abf.setSweep(sweep_number)
    time = abf.sweepX
    voltage = abf.sweepY - LJP_CORRECTION_MV

    length = len(time)
    mf = 0.015625
    offset = int(length * mf)
    time = np.concatenate([np.zeros(offset, dtype=np.float64), time])[:-offset]

    ax_full.clear()
    ax_full.plot(time, voltage, label=f"Recording: {recording}, Sweep: {sweep_number}")
    ax_full.set_title(f"Full Sweep - Recording: {recording}, Sweep: {sweep_number}")
    ax_full.set_xlabel('Time (s)')
    ax_full.set_ylabel('Voltage (mV)')

    peak_idx = np.abs(time - peak_time).argmin()
    window_size = int(highlight_window * abf.sampleRate / 1000)
    start_idx = max(0, peak_idx - window_size)
    end_idx = min(len(time), peak_idx + window_size)
    ax_full.plot(time[start_idx:end_idx], voltage[start_idx:end_idx], color='red', linewidth=2)

    if last_steady_state_voltage is None or last_steady_state_voltage['sweep'] != sweep_number:
        steady_state_voltage = calculate_steady_state_voltage(time, voltage, timescale_start, timescale_end, abf.sampleRate)
        last_steady_state_voltage = {'sweep': sweep_number, 'voltage': steady_state_voltage}
    else:
        steady_state_voltage = last_steady_state_voltage['voltage']

    ax_full.axhline(y=steady_state_voltage, color='black', linestyle='--')
    ax_full.axvline(x=timescale_start, color='grey', linestyle='--')
    ax_full.axvline(x=timescale_end, color='grey', linestyle='--')
    ax_full.set_ylim(min(voltage), max(voltage))

    ax_zoom.clear()
    ax_zoom.plot(time[start_idx:end_idx], voltage[start_idx:end_idx], color='red', linewidth=2)
    ax_zoom.set_xlim(time[start_idx], time[end_idx])
    ax_zoom.set_ylim(np.min(voltage[start_idx:end_idx]), np.max(voltage[start_idx:end_idx]))

    spike_params = spike_params_df[(spike_params_df['recording'] == recording) & 
                                   (spike_params_df['sweep_number'] == sweep_number)]

    if not spike_params.empty:
        closest_peak_idx = np.abs(spike_params['ap_peak_time'] - peak_time).argmin()
        ap_params = spike_params.iloc[closest_peak_idx]

        threshold = ap_params['ap_threshold']
        ap_peak_voltage = ap_params['ap_peak_voltage']
        ahp_amplitude = ap_params['ahp_amplitude']
        ahp_time = ap_params['ahp_peak_time']
        half_voltage = ap_params['half_voltage']
        half_width = ap_params['half_width']
        first_crossing = ap_params['first_crossing']
        second_crossing = ap_params['second_crossing']
        ahp_peak = ap_params['ahp_peak']
        ap_count = ap_params['ap_count']

        ax_zoom.axhline(y=threshold, color='green', linestyle='--')
        ax_zoom.scatter(peak_time, ap_peak_voltage, color='blue', zorder=5)
        ax_zoom.scatter(ahp_time, ahp_peak, color='purple', zorder=5)
        ax_zoom.plot([ahp_time, ahp_time], [threshold, ahp_peak], color='grey', zorder=5)
        ax_zoom.plot([first_crossing, second_crossing], [half_voltage, half_voltage],
                     color='black', linestyle='-', linewidth=2, zorder=10)
        ax_zoom.axhline(y=half_voltage, color='orange', linestyle='--')
        ax_zoom.axvline(x=first_crossing, color='brown', linestyle='--')
        ax_zoom.axvline(x=second_crossing, color='brown', linestyle='--')
        ax_zoom.set_title(f"Recording: {recording}, Sweep: {sweep_number}, AP #{ap_count}")

    fig.canvas.draw_idle()  

def on_key(event):
    global current_recording_index, current_sweep_index, current_peak_index
    recording = unique_recordings[current_recording_index]
    record_data = df[df['recording'] == recording]
    peak_times = record_data.iloc[current_sweep_index]['timestamp']

    if event.key == 'right':
        current_peak_index += 1
        if current_peak_index >= len(peak_times):
            current_peak_index = 0
            current_sweep_index += 1
            if current_sweep_index >= len(record_data):
                current_sweep_index = 0
                current_recording_index += 1
                if current_recording_index >= len(unique_recordings):
                    current_recording_index = 0

    elif event.key == 'left':
        current_peak_index -= 1
        if current_peak_index < 0:
            current_sweep_index -= 1
            if current_sweep_index < 0:
                current_recording_index -= 1
                if current_recording_index < 0:
                    current_recording_index = len(unique_recordings) - 1
                record_data = df[df['recording'] == unique_recordings[current_recording_index]]
                current_sweep_index = len(record_data) - 1

            record_data = df[df['recording'] == unique_recordings[current_recording_index]]
            peak_times = record_data.iloc[current_sweep_index]['timestamp']
            current_peak_index = len(peak_times) - 1 if len(peak_times) > 0 else 0

    update_plot()

fig.canvas.mpl_connect('key_press_event', on_key)  # ✅ Connect event
update_plot()
plt.show()
