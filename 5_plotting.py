#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 17:46:19 2025

@author: gs075
"""

import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from scipy import stats
from scipy.interpolate import interp1d  # Importing interp1d for interpolation
import pyabf

# --- Settings ---
ROOT = '/Volumes/BWH-HVDATA/Individual Folders/Garrett/PatchClamp/Analyses/RSP_Hannah_Farnsworth'
  # Root directory for data
redefine_path = True  # Set to True if working on laptop for path adjustment
ORDERED_CONDITIONS = ['control', 'tumor']  # Conditions to compare
COLOR_MAP = {'control': 'dimgrey', 'tumor': 'tomato'}  # Colors for plotting
DOT_COLOR, IR_STEP, IO_RANGE = 'black', -20, [0, 300]  # Parameters for plotting and analysis
WINDOW_SEC = 0.005  # ±5 ms for AP window (Action Potential)
COMMON_TIME = np.linspace(-WINDOW_SEC, WINDOW_SEC, 1000)  # Time range for AP waveforms
artist_to_data = {}
selected_point_artist = None
selected_annotation = None

# --- Load Updated Data ---
thresholded = pd.read_csv(f"{ROOT}/1_thresholded_data.csv")
all_spikes = pd.read_csv(f"{ROOT}/2_all_parameters.csv")  # Optional: spike-level data
ap_params_df = pd.read_csv(f"{ROOT}/2_averaged_parameters.csv")  # Averaged AP params per recording
results = pd.read_csv(f"{ROOT}/2_passive_membrane_features.csv")  # RMP and Input Resistance
results['condition'] = results['recording'].apply(lambda path: os.path.normpath(path).split(os.sep)[-2]) #remove after adjusting spike_params_calculator
filtered_data = pd.read_csv(f"{ROOT}/3_filtered_data.csv")

# --- Standardize column names for passive properties ---
results = results.rename(columns={
    'RMP_mV': 'RMP',
    'Input_Resistance_MOhm': 'IR (megaohm)'
})

# --- Filter valid recordings based on filtered_data ---
valid_recordings = set(filtered_data['recording'].unique())
thresholded = thresholded[thresholded['recording'].isin(valid_recordings)]
ap_params_df = ap_params_df[ap_params_df['recording'].isin(valid_recordings)]
results = results[results['recording'].isin(valid_recordings)]


# --- Helper Functions for Statistics and Plotting ---
# T-test for comparing two conditions
def t_test_by_condition(df, col):
    a = df[df['condition'] == ORDERED_CONDITIONS[0]][col]
    b = df[df['condition'] == ORDERED_CONDITIONS[1]][col]
    return stats.ttest_ind(a, b).pvalue

def plot_data(ax, df, y, title):
    p = t_test_by_condition(df, y)
    sns.boxplot(x='condition', y=y, data=df, hue='condition', palette=COLOR_MAP,
                order=ORDERED_CONDITIONS, showfliers=False, ax=ax)

    swarm = sns.swarmplot(x='condition', y=y, data=df, hue='condition', palette='dark:black',  
                          order=ORDERED_CONDITIONS, size=3, alpha=0.7, ax=ax)

    # Register each point collection in the swarm plot
    for line in swarm.collections:
        line.set_picker(True)
        artist_to_data[line] = (df, y)  # <-- register the artist with its data

    ax.set_title(f'{title} (p = {p:.4f})')
    for cond in ORDERED_CONDITIONS:
        n = df[df['condition'] == cond]['recording'].nunique()
        ax.plot([], [], marker='s', label=f"{cond} (n={n})", color=COLOR_MAP[cond])
    ax.set_xlabel("")  # Removes the 'condition' label on the x-axis
    return swarm

# Modified version of plot_ap_param function without individual legends
def plot_ap_param(ax, df, y, title):
    p = t_test_by_condition(df, y)

    sns.boxplot(x='condition', y=y, data=df, hue='condition', palette=COLOR_MAP,
                order=ORDERED_CONDITIONS, showfliers=False, ax=ax)

    swarm = sns.swarmplot(x='condition', y=y, data=df, hue='condition', palette='dark:black',  
                          order=ORDERED_CONDITIONS, size=3, alpha=0.7, ax=ax)

    # Register each point collection for picking
    for coll in swarm.collections:
        coll.set_picker(True)
        artist_to_data[coll] = (df, y)

    ax.set_title(f'{title} (p = {p:.4f})' if p >= 0.0001 else f'{title} (p < 0.0001)')
    for cond in ORDERED_CONDITIONS:
        n = df[df['condition'] == cond]['recording'].nunique()
        ax.plot([], [], marker='s', label=f"{cond} (n={n})", color=COLOR_MAP[cond])
    ax.set_xlabel("")

def onpick(event):
    global selected_point_artist, selected_annotation

    if isinstance(event.artist, PathCollection):
        ind = event.ind
        if len(ind) == 0:
            return

        offset = event.artist.get_offsets()[ind[0]]
        x_clicked, y_clicked = offset

        y_clicked = round(float(y_clicked), 4)
        x_categories = ORDERED_CONDITIONS

        try:
            x_index = int(round(x_clicked))
            x_clicked = x_categories[x_index]
        except (IndexError, ValueError):
            return

        # Clear previous highlight/annotation
        if selected_point_artist:
            selected_point_artist.remove()
            selected_point_artist = None
        if selected_annotation:
            selected_annotation.remove()
            selected_annotation = None

        # Find the axis this was clicked in
        ax = event.artist.axes
                
        # Loop through all dataframes to find the matching row
        for df in [results, rheos, ap_params_df, capacitance_data]:
            possible_rows = df[df['condition'] == x_clicked]
            for _, row in possible_rows.iterrows():
                for param in ['RMP', 'IR (megaohm)', 'rheobase', 'value',
                              'ap_peak_voltage', 'ahp_peak', 'half_width',
                              'ap_amplitude', 'ahp_amplitude', 'ap_threshold']:
        
                    if param in row and pd.notnull(row[param]):
                        val = round(float(row[param]), 4)
                        if val == y_clicked:
                            # Found the matching datapoint
                            recording = row.get('recording', row.get('filename', ''))
                            trimmed = recording.split('steps_analyses/')[-1] if 'steps_analyses/' in recording else recording
                            label = f"{trimmed}\n{param} = {val} ({x_clicked})"
        
                            # Highlight the selected point with the correct x/y coordinates
                            selected_point_artist = ax.plot(x_clicked, y_clicked, 'o', color='red', markersize=6, zorder=10)[0]
        
                            # Annotate it with the label
                            selected_annotation = ax.annotate(
                                label,
                                (x_clicked, y_clicked),
                                textcoords="offset points",
                                xytext=(10, 10),  # Adjust this for label placement
                                ha='left',
                                fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=0.8),  # solid white background
                                arrowprops=dict(arrowstyle="->", color='black', lw=0.8),
                                zorder=1000  # ensures it's on top
                            )
        
                            event.canvas.draw_idle()
                            print(f"Clicked on datapoint: Recording = {trimmed}, Condition = {x_clicked}, Parameter = {param}")
                            return


# --- Step 2: Rheobase Calculation ---
# Rheobase is the minimal current injection needed to generate a spike
rheos = filtered_data[filtered_data['current_injection'] > 0]
rheos = rheos[rheos['spike_count'] > 0].groupby(['recording', 'condition'])['current_injection'].min().reset_index(name='rheobase')

# --- Step 3: Input/Output Curve ---
# Filter data for current injections within the defined range (IO_RANGE)
io_df = filtered_data[(filtered_data['current_injection'] >= IO_RANGE[0]) & 
                      (filtered_data['current_injection'] <= IO_RANGE[1])]

# --- Step 4: Average first 1–3 APs per recording at rheobase ---
# For each recording, get the first sweep and average the first 3 APs
min_sweep_rows = []
for rec in all_spikes['recording'].unique():
    rec_rows = all_spikes[(all_spikes['recording'] == rec) & (all_spikes['current_injection'] > 0)]
    if not rec_rows.empty:
        min_sweep = rec_rows['sweep_number'].min()
        min_rows = rec_rows[rec_rows['sweep_number'] == min_sweep]
        min_sweep_rows.append(min_rows)

# Concatenate the rows with minimum sweep number
rheo_df = pd.concat(min_sweep_rows, ignore_index=True)

# Filter rows with action potentials (APs) and average features of top 3 APs at rheobase
filtered_ap_rows = []
for (rec, sweep), group in rheo_df[rheo_df['ap_count'] > 0].groupby(['recording', 'sweep_number']):
    min_inj = group['current_injection'].min()
    if min_inj != group['current_injection'].iloc[0]:
        continue
    top_ap = group.sort_values('ap_peak_time').head(3)
    avg_row = top_ap.iloc[0].copy()
    for col in ['ap_peak_voltage', 'ahp_peak', 'half_width', 'ap_amplitude', 'ahp_amplitude', 'ap_threshold']:
        avg_row[col] = top_ap[col].mean()  # Average the features for the selected APs
    filtered_ap_rows.append(avg_row)

# Convert the averaged action potential parameters into a DataFrame
ap_params_df = pd.DataFrame(filtered_ap_rows)
ap_params_df = ap_params_df[ap_params_df['recording'].isin(valid_recordings)]


# --- Step 5: Load traces from ABF files and extract AP windows ---
# Function to load data from ABF files
def load_trace(recording, sweep, abf_dir=f"{ROOT}/abfs"):
    if redefine_path:
        old_user = 'garrett'
        new_user = 'gs075'
        recording = recording.replace(f'/Users/{old_user}/', f'/Users/{new_user}/')

    path = os.path.join(abf_dir, recording)
    abf = pyabf.ABF(path)
    abf.setSweep(sweep)
    time = abf.sweepX
    voltage = abf.sweepY

    offset = int(len(time) * 0.015625)
    time = np.concatenate([np.zeros(offset), time])[:-offset]
    return time, voltage

# Extract the action potential (AP) window around the peak time
def extract_ap_window(time, voltage, peak_time, window=WINDOW_SEC):
    mask = (time >= peak_time - window) & (time <= peak_time + window)
    return time[mask] - peak_time, voltage[mask]

# --- Step 6: Collect and interpolate AP waveforms ---
# Collect and interpolate action potential waveforms for each condition
waveforms = {cond: [] for cond in ORDERED_CONDITIONS}
grouped = ap_params_df.groupby(['recording', 'sweep_number'])

for (rec, sweep), group in grouped:
    cond = group['condition'].iloc[0]
    peaks = group.sort_values('ap_peak_time')['ap_peak_time'].values[:3]
    try:
        t, v = load_trace(rec, int(sweep))
        interp_waves = []
        for peak in peaks:
            rt, ap_v = extract_ap_window(t, v, peak)
            if len(rt) > 1 and len(rt) == len(ap_v):
                f = interp1d(rt, ap_v, kind='linear', bounds_error=False, fill_value='extrapolate')
                interp_v = f(COMMON_TIME)
                interp_waves.append(interp_v)
        if interp_waves:
            mean_wave = np.mean(np.vstack(interp_waves), axis=0)
            waveforms[cond].append(mean_wave)
    except Exception as e:
        print(f"Error loading {rec}, sweep {sweep}: {e}")

# --- Step 7: Plot average AP waveforms in subplots ---
# Set up subplots for various analyses
fig, axs = plt.subplots(4, 3, figsize=(18, 18))
axs = axs.flatten()


# Plot data for various parameters and enable picking
swarm1 = plot_data(axs[0], results, 'RMP', 'Resting Membrane Potential')
swarm2 = plot_data(axs[1], results, 'IR (megaohm)', 'Input Resistance')
p_rheo = t_test_by_condition(rheos, 'rheobase')

sns.boxplot(x='condition', y='rheobase', data=rheos, hue='condition', palette=COLOR_MAP,
            order=ORDERED_CONDITIONS, showfliers=False, ax=axs[2])
swarm_rheo = sns.swarmplot(x='condition', y='rheobase', data=rheos, hue='condition', 
                           palette='dark:black', order=ORDERED_CONDITIONS, size=3, alpha=0.7, ax=axs[2])
for coll in swarm_rheo.collections:
    coll.set_picker(True)
    artist_to_data[coll] = (rheos, 'rheobase')

# Set formatted title for rheobase plot
axs[2].set_title(f'Rheobase (p = {p_rheo:.4f})' if p_rheo >= 0.0001 else 'Rheobase (p < 0.0001)')

# Plot AP parameter data
plot_ap_param(axs[3], ap_params_df, 'ap_peak_voltage', 'Action Potential Peak Voltage')
plot_ap_param(axs[4], ap_params_df, 'ahp_peak', 'Afterhyperpolarization Peak')
plot_ap_param(axs[5], ap_params_df, 'half_width', 'Action Potential Half Width')
plot_ap_param(axs[6], ap_params_df, 'ap_amplitude', 'Action Potential Amplitude')
plot_ap_param(axs[7], ap_params_df, 'ahp_amplitude', 'Afterhyperpolarpolarization Amplitude')
plot_ap_param(axs[8], ap_params_df, 'ap_threshold', 'Action Potential Threshold')

# --- Customize labels and title for waveforms plot ---
ax_waveforms = axs[10]
for cond, waveforms_cond in waveforms.items():
    mean_wave = np.mean(np.vstack(waveforms_cond), axis=0)
    sem_wave = np.std(np.vstack(waveforms_cond), axis=0) / np.sqrt(len(waveforms_cond))
    sample_size = len(waveforms_cond)
    ax_waveforms.plot(COMMON_TIME, mean_wave, label=f"{cond} (n={sample_size})", color=COLOR_MAP[cond])
    ax_waveforms.fill_between(COMMON_TIME, mean_wave - sem_wave, mean_wave + sem_wave, 
                               color=COLOR_MAP[cond], alpha=0.3)

# Customize labels and title for waveforms plot
ax_waveforms.set_title("AP Waveforms by Condition")
ax_waveforms.set_xlabel("Time (s)")
ax_waveforms.set_ylabel("Adjusted Vm")

# --- Input/Output Curve ---
ax10 = axs[11]  # Position of subplot 10
for cond in ORDERED_CONDITIONS:
    c_data = io_df[io_df['condition'] == cond]
    grouped = c_data.groupby('current_injection')['spike_count']
    avg, sem = grouped.mean(), grouped.sem()
    n = c_data['recording'].nunique()
    ax10.plot(avg.index, avg, label=f"{cond} (n={n})", color=COLOR_MAP[cond])
    ax10.fill_between(avg.index, avg - sem, avg + sem, color=COLOR_MAP[cond], alpha=0.2)
ax10.set_xlabel('Current Injection (pA)')
ax10.set_ylabel('Average Spike Count')
ax10.set_title('Input/Output Curve')

# --- Step 10: Plot Capacitance ---
# Collect capacitance data and perform statistical test
def parse_abf_metadata(file_paths):
    parsed_data = []
    for path in file_paths:
        abf = pyabf.ABF(path)
        lines = abf.headerText.splitlines()
        condition = os.path.basename(os.path.dirname(path))
        for line in lines:
            if any(key in line for key in ["fTelegraphMembraneCap", "fTelegraphAccessResistance"]):
                dtype = "Membrane_Capacitance" if "MembraneCap" in line else "Access_Resistance"
                match = re.search(r"\[([\d.eE+-]+)", line)
                value = float(match.group(1)) if match else None
                parsed_data.append({
                    'filename': os.path.basename(path),
                    'condition': condition,
                    'data_type': dtype,
                    'value': value,
                    'protocol': abf.protocol
                })
    return parsed_data

# --- Step 9: Collect closest ABF file paths ---
def find_closest_abf_files(filtered_df, root_dir):
    trimmed_ids = ['_'.join(os.path.basename(rec).split('_')[:4]) for rec in filtered_df['recording'].unique()]
    abf_files = []
    for root_dir, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith('membrane_test.abf'):
                parts = fname.split('_')
                if len(parts) >= 4:
                    try:
                        abf_files.append({
                            "file_path": os.path.join(root_dir, fname),
                            "filename": fname,
                            "date": '_'.join(parts[:3]),
                            "counter": int(parts[3]),
                            "full_id": '_'.join(parts[:4])
                        })
                    except ValueError:
                        continue

    abf_df = pd.DataFrame(abf_files)
    closest_paths = []
    for tid in trimmed_ids:
        parts = tid.split('_')
        if len(parts) < 4: continue
        try:
            date, count = '_'.join(parts[:3]), int(parts[3])
            subset = abf_df[(abf_df['date'] == date) & (abf_df['counter'] < count)]
            if not subset.empty:
                closest = subset.iloc[(count - subset['counter']).abs().argmin()]
                closest_paths.append(closest['file_path'])
        except ValueError:
            continue
    return closest_paths

# Collect closest ABF file paths based on filtered data
closest_paths = find_closest_abf_files(filtered_data, ROOT)

# Parse the ABF metadata for capacitance and resistance
metadata = parse_abf_metadata(closest_paths)
df_metadata = pd.DataFrame(metadata)

# Filter for membrane capacitance data
capacitance_data = df_metadata[df_metadata['data_type'] == 'Membrane_Capacitance']

# --- Step 10: Plot capacitance on subplot 10 (axs[9]) ---
# Perform t-test for significance between conditions
p_val = stats.ttest_ind(capacitance_data[capacitance_data['condition'] == 'control']['value'],
                        capacitance_data[capacitance_data['condition'] == 'tumor']['value'])[1]

# Plot Capacitance (boxplot and stripplot)
sns.boxplot(x='condition', y='value', data=capacitance_data, hue='condition', palette=COLOR_MAP,
            order=ORDERED_CONDITIONS, showfliers=False, ax=axs[9])
strip = sns.stripplot(x='condition', y='value', data=capacitance_data, hue='condition',
                      color=DOT_COLOR, jitter=True, size=3, alpha=0.7, ax=axs[9])
for coll in strip.collections:
    coll.set_picker(True)
    artist_to_data[coll] = (capacitance_data, 'value')


# Set formatted title with p-value for capacitance plot
axs[9].set_title(f"Capacitance (p = {p_val:.4f})" if p_val >= 0.0001 else "Capacitance (p < 0.0001)")

# Axis labels for capacitance plot
axs[9].set_ylabel("Capacitance (pF)")
axs[9].set_xlabel("")  # Remove default x-axis label


# --- Step 9: Add a Universal Legend ---
# Creating a custom legend at the bottom
handles, labels = [], []
for cond in ORDERED_CONDITIONS:
    n = capacitance_data[capacitance_data['condition'] == cond]['filename'].nunique()
    handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MAP[cond], markersize=5)
    handles.append(handle)
    labels.append(f"{cond} (n={n})")

fig.legend(handles, labels, loc='center', ncol=2, bbox_to_anchor=(0.5, 0.01))


# Connect pick event to the handler
fig.canvas.mpl_connect('pick_event', onpick)


# Show the figure with all subplots
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.2)  # hspace for vertical padding, wspace for horizontal padding
plt.show()
