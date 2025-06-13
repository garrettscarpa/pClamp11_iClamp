#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:15:08 2025

@author: gs075
"""
import matplotlib.pyplot as plt
import pyabf
import numpy as np
import os
import pandas as pd
from matplotlib.widgets import Slider, TextBox, Button
import ast

# Folder where recordings are stored
root = '/Users/gs075/Desktop/final_analysis_HF/'
protocol = 'Steps_300'

min_isi = .3  # ms
threshold = 0  # default threshold
is_unblinded = False
sweep_threshold = None
is_slider_update_internal = False
is_updating_plot = False

abf_files = []
excluded_dirs = {'excluded'}
for root_dir, subdirs, files in os.walk(root):
    subdirs[:] = [d for d in subdirs if d not in excluded_dirs]
    for file in files:
        if file.endswith(f"{protocol}.abf"):
            abf_files.append(os.path.join(root_dir, file))

current_sweep = 0
current_recording_index = 0
sweep_thresholds = {}

window_ms = 5
window_samples = None

temporal_range_start = 0.25
temporal_range_end = 1.26

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
action_potential_data = []

def calculate_steady_state_voltage(time, voltage, timescale_start, timescale_end, sample_rate):
    start_idx = np.abs(time - timescale_start).argmin()
    end_idx = np.abs(time - timescale_end).argmin()
    return np.mean(voltage[start_idx:end_idx])

def update_threshold(val):
    global last_recording_threshold, is_updating_plot, is_slider_update_internal
    if is_updating_plot or is_slider_update_internal:
        return
    last_recording_threshold = threshold_slider.val
    key = (current_recording_index, current_sweep)
    sweep_thresholds[key] = last_recording_threshold
    update_plot(current_recording_index, current_sweep)


def update_temporal_range(val):
    global temporal_range_start, temporal_range_end
    try:
        temporal_range_start = float(start_time_textbox.text)
        temporal_range_end = float(end_time_textbox.text)
    except ValueError:
        print("Invalid temporal range")
    update_plot(current_recording_index, current_sweep)

def determine_y_limits_for_recording(recording_index):
    abf_file = abf_files[recording_index]
    abf = pyabf.ABF(abf_file)
    min_v, max_v, min_c, max_c = float('inf'), float('-inf'), float('inf'), float('-inf')
    for sweep_number in range(abf.sweepCount):
        abf.setSweep(sweep_number)
        min_v = min(min_v, np.min(abf.sweepY))
        max_v = max(max_v, np.max(abf.sweepY))
        min_c = min(min_c, np.min(abf.sweepC))
        max_c = max(max_c, np.max(abf.sweepC))
    return (min_v, max_v), (min_c, max_c)

def update_plot(recording_index, sweep_number):
    global current_recording_index, current_sweep, window_samples
    global temporal_range_start, temporal_range_end
    global last_recording_threshold, is_updating_plot
    global action_potential_data
    global sweep_threshold, is_slider_update_internal
    
    current_recording_index = recording_index
    current_sweep = sweep_number
    is_updating_plot = True

    abf_file = abf_files[recording_index]
    abf = pyabf.ABF(abf_file)
    abf.setSweep(sweep_number)

    # Removed the early return that caused recursion
    # Instead, proceed with threshold logic here

    key = (current_recording_index, current_sweep)

    if key not in sweep_thresholds:
        latency_samples = int(min_isi * (abf.sampleRate / 1000))
        window_samples = int(window_ms * (abf.sampleRate / 1000))
    
        raw_time = abf.sweepX
        mf = 0.015625
        offset = int(len(raw_time) * mf)
        time = np.concatenate([np.zeros(offset), raw_time])[:-offset]
    
        voltage = abf.sweepY
        command = abf.sweepC
    
        time_filter = (time >= temporal_range_start) & (time <= temporal_range_end)
        time_filtered = time[time_filter]
        voltage_filtered = voltage[time_filter]
        command_filtered = command[time_filter]
    
        ssv = calculate_steady_state_voltage(time_filtered, voltage_filtered,
                                             temporal_range_start, temporal_range_end, abf.sampleRate)
    
        current_injection = float(np.median(command_filtered))
    
        if current_injection >= 0:
            sweep_threshold = ssv + 20
        else:
            sweep_threshold = 0
    
        sweep_thresholds[key] = sweep_threshold
    else:
        sweep_threshold = sweep_thresholds[key]

    # Avoid feedback loop from triggering on_changed
    # Removed code attempting to manipulate observers (not supported in matplotlib Slider)

    # Set the threshold slider to match the current threshold for this sweep,
    # only if slider val differs
    if sweep_threshold is not None and -100 <= sweep_threshold <= 100:
        if threshold_slider.val != sweep_threshold:
            is_slider_update_internal = True
            fig.canvas.draw_idle()  # Ensure slider geometry ready
            threshold_slider.set_val(sweep_threshold)
            is_slider_update_internal = False

    latency_samples = int(min_isi * (abf.sampleRate / 1000))
    window_samples = int(window_ms * (abf.sampleRate / 1000))

    raw_time = abf.sweepX
    mf = 0.015625
    offset = int(len(raw_time) * mf)
    time = np.concatenate([np.zeros(offset), raw_time])[:-offset]

    voltage = abf.sweepY
    command = abf.sweepC

    if temporal_range_end is None:
        temporal_range_end = time[-1]

    time_filter = (time >= temporal_range_start) & (time <= temporal_range_end)
    time_filtered = time[time_filter]
    voltage_filtered = voltage[time_filter]
    command_filtered = command[time_filter]

    ssv = calculate_steady_state_voltage(time_filtered, voltage_filtered,
                                         temporal_range_start, temporal_range_end, abf.sampleRate)

    detected_peaks = []
    last_crossing = -latency_samples
    crossings = np.where(voltage_filtered > sweep_threshold)[0]

    for idx in crossings:
        if idx - last_crossing > latency_samples:
            win_start = max(0, idx - window_samples)
            win_end = min(len(voltage_filtered), idx + window_samples)
            peak_idx = np.argmax(voltage_filtered[win_start:win_end]) + win_start
            detected_peaks.append((time_filtered[peak_idx], voltage_filtered[peak_idx]))
            last_crossing = idx
    
    # === Store Spike Data ===
    timestamps = [pt for pt, _ in detected_peaks]
    rel_path = os.path.relpath(abf_file, root)  # tumor/2025_05_14_0037_Steps_300.abf
    condition = rel_path.split(os.sep)[0]
    entry = {
        'recording': abf_file,
        'sweep_number': sweep_number,
        'current_injection': float(np.median(command)),  # median of step
        'threshold': sweep_threshold,
        'spike_count': len(timestamps),
        'condition': condition,
        'timestamps': timestamps,
        'ssv': ssv
    }

    # Avoid duplicates
    action_potential_data[:] = [
        row for row in action_potential_data
        if not (row['recording'] == abf_file and row['sweep_number'] == sweep_number)
    ]
    action_potential_data.append(entry)

    # === Plotting ===
    axes[0].cla()
    axes[1].cla()

    axes[0].plot(time_filtered, voltage_filtered, color='purple')
    display = abf_file if is_unblinded else f"Recording {recording_index + 1} of {len(abf_files)}"
    axes[0].set_title(f'Signal - {display} | Sweep {sweep_number}')
    axes[0].set_ylabel('Voltage (mV)')

    axes[1].plot(time_filtered, command_filtered, color='k')
    axes[1].set_title('Command')
    axes[1].set_xlabel('Time (s)')

    (min_v, max_v), (min_c, max_c) = determine_y_limits_for_recording(recording_index)
    axes[0].set_ylim(min_v - 20, max_v + 20)
    axes[1].set_ylim(min_c - 20, max_c + 20)

    axes[0].axhline(y=ssv, color='b', linestyle='-', label=f'SSV = {ssv:.2f} mV')
    axes[0].axhline(y=sweep_threshold, color='r', linestyle='--', label=f'Threshold = {sweep_threshold} mV')

    for pt, pv in detected_peaks:
        axes[0].scatter(pt, pv, color='orange')

    axes[0].legend()

    plt.draw()
    is_updating_plot = False


def save_and_export(event):
    df = pd.DataFrame(action_potential_data)
    for idx, row in df.iterrows():
        unique_ts = list(sorted(set(row['timestamps'])))
        df.at[idx, 'spike_count'] = max(0, row['spike_count'] - (len(row['timestamps']) - len(unique_ts)))
        df.at[idx, 'timestamps'] = unique_ts
    df.to_csv(os.path.join(root, '1_thresholded_data.csv'), index=False)
    print("Data saved!")

def toggle_unblind(event):
    global is_unblinded
    is_unblinded = not is_unblinded
    update_plot(current_recording_index, current_sweep)

def on_key(event):
    global current_sweep, current_recording_index
    abf = pyabf.ABF(abf_files[current_recording_index])
    if event.key == 'right':
        if current_sweep < abf.sweepCount - 1:
            current_sweep += 1
        else:
            if current_recording_index < len(abf_files) - 1:
                current_recording_index += 1
                current_sweep = 0
    elif event.key == 'left':
        if current_sweep > 0:
            current_sweep -= 1
        else:
            if current_recording_index > 0:
                current_recording_index -= 1
                abf = pyabf.ABF(abf_files[current_recording_index])
                current_sweep = abf.sweepCount - 1
    update_plot(current_recording_index, current_sweep)

button_ax = plt.axes([0.45, 0.04, 0.15, 0.05])
save_button = Button(button_ax, 'Save & Export')
save_button.on_clicked(save_and_export)

unblind_ax = plt.axes([0.65, 0.04, 0.15, 0.05])
unblind_button = Button(unblind_ax, 'Unblind')
unblind_button.on_clicked(toggle_unblind)

fig.canvas.mpl_connect('key_press_event', on_key)

ax_threshold = plt.axes([0.15, 0.005, 0.7, 0.03])
threshold_slider = Slider(ax_threshold, 'Threshold (mV)', -100, 100, valinit=threshold, valstep=1)
threshold_slider.on_changed(update_threshold)

ax_start_time = plt.axes([0.15, 0.15, 0.7, 0.03])
start_time_textbox = TextBox(ax_start_time, 'Start Time (s)', initial=str(temporal_range_start))
start_time_textbox.on_submit(update_temporal_range)

ax_end_time = plt.axes([0.15, 0.10, 0.7, 0.03])
end_time_textbox = TextBox(ax_end_time, 'End Time (s)', initial=str(temporal_range_end))
end_time_textbox.on_submit(update_temporal_range)

plt.subplots_adjust(bottom=0.25)
update_plot(current_recording_index, current_sweep)
plt.show()
