#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:15:08 2025

@author: gs075
"""

import matplotlib.pyplot as plt
import pyabf
import re
import numpy as np
import os
import pandas as pd
from matplotlib.widgets import Slider, TextBox, Button

# --- User parameters ---
root = '/Users/gs075/Desktop/400pA_RSP_Hannah_Farnsworth'
protocols_of_interest = ['membrane_test', 'Steps_300', 'Steps_400']
min_isi = 1  # ms
threshold = 0  # default slider threshold
LJP_CORRECTION_MV = 14.681
window_ms = 2
temporal_range_start = 0.25
temporal_range_end = 1.25
excluded_dirs = {'excluded'}

# --- Global state ---
current_recording_index = 0
current_sweep = 0
sweep_thresholds = {}
action_potential_data = []
is_unblinded = False
is_slider_update_internal = False
is_updating_plot = False

# --- Helper classes & functions ---

class MergedCellData:
    def __init__(self, steps_300_path, steps_400_path):
        self.y_300, self.c_300, self.sample_rate, self.sweepX = self.load_abf_sweeps(steps_300_path)
        self.y_400, self.c_400, sample_rate_400, sweepX_400 = self.load_abf_sweeps(steps_400_path)
        assert self.sample_rate == sample_rate_400, "Sample rates do not match!"
        assert np.allclose(self.sweepX, sweepX_400), "Sweep times do not match!"
        self.sweeps_y = np.concatenate((self.y_300, self.y_400), axis=0)
        self.sweeps_c = np.concatenate((self.c_300, self.c_400), axis=0)
        self.sweepCount = self.sweeps_y.shape[0]

    def load_abf_sweeps(self, abf_path):
        abf = pyabf.ABF(abf_path)
        sweeps_y = []
        sweeps_c = []
        for sweep in range(abf.sweepCount):
            abf.setSweep(sweep)
            sweeps_y.append(abf.sweepY) 
            sweeps_c.append(abf.sweepC)
        return np.array(sweeps_y), np.array(sweeps_c), abf.sampleRate, abf.sweepX

    def get_sweep(self, sweep_number):
        return self.sweepX, self.sweeps_y[sweep_number], self.sweeps_c[sweep_number]

def calculate_steady_state_voltage(time, voltage, timescale_start, timescale_end, sample_rate):
    start_idx = np.abs(time - timescale_start).argmin()
    end_idx = np.abs(time - timescale_end).argmin()
    return np.mean(voltage[start_idx:end_idx])

def determine_y_limits_for_recording(merged_cell):
    min_v = np.min(merged_cell.sweeps_y)
    max_v = np.max(merged_cell.sweeps_y)
    min_c = np.min(merged_cell.sweeps_c)
    max_c = np.max(merged_cell.sweeps_c)
    return (min_v, max_v), (min_c, max_c)

# --- File discovery & grouping ---
def find_and_group_files(root, protocols_of_interest, excluded_dirs):
    all_files = []
    for root_dir, subdirs, files in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in excluded_dirs]
        for file in files:
            if file.endswith('.abf'):
                if any(p in file for p in protocols_of_interest):
                    all_files.append(os.path.join(root_dir, file))
    all_files.sort()

    # Group every 3 files (membrane_test, Steps_300, Steps_400) per cell
    assert len(all_files) % 3 == 0, f"Expected multiples of 3 files, got {len(all_files)} files."

    cells = []
    for i in range(0, len(all_files), 3):
        chunk = all_files[i:i+3]
        # Sort chunk by protocol order
        chunk_sorted = sorted(chunk, key=lambda x: protocols_of_interest.index(
            next(p for p in protocols_of_interest if p in x)
        ))
        cells.append(chunk_sorted)
    return cells

# Load grouped files and create merged data objects
cells_files = find_and_group_files(root, protocols_of_interest, excluded_dirs)

merged_cells_data = []
for cell_files in cells_files:
    membrane_test_file, steps_300_file, steps_400_file = cell_files
    merged_cell = MergedCellData(steps_300_file, steps_400_file)
    merged_cells_data.append({
        'membrane_test_file': membrane_test_file,
        'merged_data': merged_cell
    })

# --- Load existing thresholds if any ---
threshold_data_file = os.path.join(root, '1_thresholded_data.csv')
if os.path.exists(threshold_data_file):
    df_existing = pd.read_csv(threshold_data_file)
    for idx, row in df_existing.iterrows():
        try:
            recording = row['recording']
            sweep_number = int(row['sweep_number'])
            threshold_val = float(row['threshold'])

            # Find matching cell index by membrane_test_file or merged_data key
            # We'll match by membrane_test_file path here (exact match)
            for i, cell in enumerate(merged_cells_data):
                if os.path.normpath(cell['membrane_test_file']) == os.path.normpath(recording):
                    key = (i, sweep_number)
                    sweep_thresholds[key] = threshold_val
                    break
        except Exception as e:
            print(f"Error loading threshold for row {idx}: {e}")

# --- Plotting setup ---
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

def update_plot(recording_index, sweep_number):
    global current_recording_index, current_sweep, is_updating_plot, is_slider_update_internal
    global sweep_thresholds, action_potential_data

    is_updating_plot = True
    current_recording_index = recording_index
    current_sweep = sweep_number

    merged_cell = merged_cells_data[recording_index]['merged_data']
    membrane_test_file = merged_cells_data[recording_index]['membrane_test_file']

    # Retrieve sweep data
    time, voltage_raw, command = merged_cell.get_sweep(sweep_number)
    voltage = voltage_raw - LJP_CORRECTION_MV

    key = (recording_index, sweep_number)

    # Threshold calculation (same logic as original)
    if key not in sweep_thresholds:
        latency_samples = int(min_isi * (merged_cell.sample_rate / 1000))
        window_samples = int(window_ms * (merged_cell.sample_rate / 1000))

        mf = 0.015625
        offset = int(len(time) * mf)
        # Time adjustment like in original
        time_adj = np.concatenate([np.zeros(offset), time])[:-offset]

        time_filter = (time_adj >= temporal_range_start) & (time_adj <= temporal_range_end)
        time_filtered = time_adj[time_filter]
        voltage_filtered = voltage[time_filter]
        command_filtered = command[time_filter]

        ssv = calculate_steady_state_voltage(time_filtered, voltage_filtered,
                                             temporal_range_start, temporal_range_end, merged_cell.sample_rate)

        current_injection = float(np.median(command_filtered))

        if current_injection >= 0:
            sweep_threshold = ssv + 20
        else:
            sweep_threshold = 0
        sweep_thresholds[key] = sweep_threshold
    else:
        sweep_threshold = sweep_thresholds[key]

    # Update slider position (avoid feedback loop)
    if sweep_threshold is not None and -100 <= sweep_threshold <= 100:
        if threshold_slider.val != sweep_threshold:
            is_slider_update_internal = True
            threshold_slider.set_val(sweep_threshold)
            is_slider_update_internal = False

    latency_samples = int(min_isi * (merged_cell.sample_rate / 1000))
    window_samples = int(window_ms * (merged_cell.sample_rate / 1000))

    mf = 0.015625
    offset = int(len(time) * mf)
    time_adj = np.concatenate([np.zeros(offset), time])[:-offset]

    time_filter = (time_adj >= temporal_range_start) & (time_adj <= temporal_range_end)
    time_filtered = time_adj[time_filter]
    voltage_filtered = voltage[time_filter]
    command_filtered = command[time_filter]

    ssv = calculate_steady_state_voltage(time_filtered, voltage_filtered,
                                         temporal_range_start, temporal_range_end, merged_cell.sample_rate)

    # Detect spikes based on threshold crossing and min ISI
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

    timestamps = [pt for pt, _ in detected_peaks]
    rel_path = os.path.relpath(membrane_test_file, root)
    condition = rel_path.split(os.sep)[0]

    entry = {
        'recording': membrane_test_file,
        'sweep_number': sweep_number,
        'current_injection': float(np.median(command_filtered)),
        'threshold': sweep_threshold,
        'spike_count': len(timestamps),
        'condition': condition,
        'timestamps': timestamps,
        'ssv': ssv
    }

    # Remove duplicates and add new entry
    action_potential_data[:] = [
        row for row in action_potential_data
        if not (row['recording'] == membrane_test_file and row['sweep_number'] == sweep_number)
    ]
    action_potential_data.append(entry)

    # Plot voltage and command
    axes[0].cla()
    axes[1].cla()

    rec = rel_path.split('/', 1)[1].rsplit('_', 2)[0] + " | " + f"Condition: {condition}"
    display_name = rec if is_unblinded else f"Cell {recording_index + 1} of {len(merged_cells_data)}"
    axes[0].plot(time_filtered, voltage_filtered, color='purple')
    axes[0].set_title(f'{display_name} | Sweep {sweep_number}')
    axes[0].set_ylabel('Voltage (mV)')

    axes[1].plot(time_filtered, command_filtered, color='k')
    axes[1].set_title('Command')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Current (pA)')

    (min_v, max_v), (min_c, max_c) = determine_y_limits_for_recording(merged_cell)
    axes[0].set_ylim(min_v - 20, max_v + 20)
    axes[1].set_ylim(min_c - 20, max_c + 20)

    axes[0].axhline(y=ssv, color='b', linestyle='-', alpha = 0.4, label=f'SSV = {ssv:.2f} mV')
    axes[0].axhline(y=sweep_threshold, color='g', alpha = 0.6, linestyle='--', label=f'Threshold = {sweep_threshold} mV')

    for pt, pv in detected_peaks:
        axes[0].scatter(pt, pv, color='crimson')

    plt.draw()

    is_updating_plot = False

def update_threshold(val):
    global sweep_thresholds, current_recording_index, current_sweep, is_updating_plot, is_slider_update_internal
    if is_updating_plot or is_slider_update_internal:
        return
    key = (current_recording_index, current_sweep)
    sweep_thresholds[key] = threshold_slider.val
    update_plot(current_recording_index, current_sweep)

def update_temporal_range(val):
    global temporal_range_start, temporal_range_end
    try:
        temporal_range_start = float(start_time_textbox.text)
        temporal_range_end = float(end_time_textbox.text)
    except ValueError:
        print("Invalid temporal range inputs")
        return
    update_plot(current_recording_index, current_sweep)

def next_sweep(event):
    global current_recording_index, current_sweep
    merged_cell = merged_cells_data[current_recording_index]['merged_data']
    if current_sweep < merged_cell.sweepCount - 1:
        current_sweep += 1
    else:
        if current_recording_index < len(merged_cells_data) - 1:
            current_recording_index += 1
            merged_cell = merged_cells_data[current_recording_index]['merged_data']
            current_sweep = 0
        else:
            print("You are at the last sweep of the last cell.")
            return
    update_plot(current_recording_index, current_sweep)

def prev_sweep(event):
    global current_recording_index, current_sweep
    if current_sweep > 0:
        current_sweep -= 1
    else:
        if current_recording_index > 0:
            current_recording_index -= 1
            merged_cell = merged_cells_data[current_recording_index]['merged_data']
            current_sweep = merged_cell.sweepCount - 1
        else:
            print("You are at the first sweep of the first cell.")
            return
    update_plot(current_recording_index, current_sweep)


def next_recording(event):
    global current_recording_index, merged_cells_data, current_sweep
    if current_recording_index + 1 < len(merged_cells_data):
        current_recording_index += 1
        current_sweep = 0
    update_plot(current_recording_index, current_sweep)

def prev_recording(event):
    global current_recording_index, merged_cells_data, current_sweep
    if current_recording_index - 1 >= 0:
        current_recording_index -= 1
        current_sweep = 0
    update_plot(current_recording_index, current_sweep)

def toggle_blinding(event):
    global is_unblinded
    is_unblinded = not is_unblinded
    update_plot(current_recording_index, current_sweep)

def save_thresholds(event):
    global action_potential_data

    # Create a copy of the data for modification
    modified_data = []
    for entry in action_potential_data:
        new_entry = entry.copy()
        x = new_entry['recording']
        new_entry['recording'] = x.rsplit('/', 1)[-1].rsplit('_', 2)[0]
        modified_data.append(new_entry)

    df = pd.DataFrame(modified_data)

    def construct_path(row):
        base_dir = root
        condition = row['condition']
        rec_base = row['recording']  # e.g., '2025_05_06_0006'
        current_injection = row['current_injection']
        
        # Determine protocol type based on current injection
        protocol = 'Steps_300' if current_injection <= 300 else 'Steps_400'
        
        # Extract date and numeric part
        match = re.match(r"(.+?)_(\d{4})$", rec_base)
        if not match:
            print(f"[Warning] Unexpected recording format: {rec_base}")
            return None
        
        date_part, num_str = match.groups()
        num = int(num_str)
        
        # Start searching from the *next* recording number
        for i in range(1, 10):  # Look up to 9 recordings ahead
            new_num = num + i
            new_rec_id = f"{date_part}_{new_num:04d}_{protocol}.abf"
            candidate_path = os.path.join(base_dir, condition, new_rec_id)
            if os.path.exists(candidate_path):
                return candidate_path
        
        print(f"[Warning] ABF file not found for {rec_base}, protocol {protocol}")
        return None


    df['path'] = df.apply(construct_path, axis=1)

    df.to_csv(threshold_data_file, index=False)
    print(f"Saved threshold data to {threshold_data_file}")



def on_key(event):
    if event.key == 'left':
        prev_sweep(None)
    elif event.key == 'right':
        next_sweep(None)

fig.canvas.mpl_connect('key_press_event', on_key)

# --- UI setup ---

plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.9, hspace=0.4)

axcolor = 'lightgoldenrodyellow'

# Slider for threshold
axthreshold = plt.axes([0.2, 0.2, 0.6, 0.02], facecolor=axcolor)
threshold_slider = Slider(axthreshold, 'Threshold', -100, 100, valinit=threshold)
threshold_slider.on_changed(update_threshold)

# TextBoxes for temporal range
axstart = plt.axes([0.3, 0.12, 0.13, 0.04])
start_time_textbox = TextBox(axstart, 'Start Time (s):', initial=str(temporal_range_start))
start_time_textbox.on_submit(update_temporal_range)

axend = plt.axes([0.7, 0.12, 0.13, 0.04])
end_time_textbox = TextBox(axend, 'End Time (s):', initial=str(temporal_range_end))
end_time_textbox.on_submit(update_temporal_range)

# Buttons
axprevsweep = plt.axes([0.05, 0.02, 0.13, 0.05])
bprevsweep = Button(axprevsweep, 'Prev Sweep')
bprevsweep.on_clicked(prev_sweep)

axnextsweep = plt.axes([0.18, 0.02, 0.13, 0.05])
bnextsweep = Button(axnextsweep, 'Next Sweep')
bnextsweep.on_clicked(next_sweep)

axprevrec = plt.axes([0.34, 0.02, 0.11, 0.05])
bprevrec = Button(axprevrec, 'Prev Cell')
bprevrec.on_clicked(prev_recording)

axnextrec = plt.axes([0.45, 0.02, 0.11, 0.05])
bnextrec = Button(axnextrec, 'Next Cell')
bnextrec.on_clicked(next_recording)

axblind = plt.axes([0.6, 0.02, 0.15, 0.05])
bblind = Button(axblind, 'Toggle Blinding')
bblind.on_clicked(toggle_blinding)

axsave = plt.axes([0.8, 0.02, 0.15, 0.05])
bsave = Button(axsave, 'Save')
bsave.on_clicked(save_thresholds)

# Initialize first plot
update_plot(current_recording_index, current_sweep)

plt.show()
