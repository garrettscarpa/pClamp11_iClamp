# === START OF UPDATED BLOCK ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib.collections import PathCollection
import pyabf
import re
import os
from scipy.interpolate import interp1d

# === CONFIGURATION ===
root = '/Users/gs075/Desktop/400pA_RSP_Hannah_Farnsworth'

sample_rate = 10000
highlight_window = 20  # ms
timescale_start = 0.25  # s
timescale_end = 1.25    # s
LJP_CORRECTION_MV = 14.681

# === DATA PREP ===
modified_csv = os.path.join(root, '3_modified_spikes.csv')
original_csv = os.path.join(root, '2_all_parameters.csv')

spike_params_df = pd.read_csv(original_csv)
spike_params_df = spike_params_df[spike_params_df['current_injection'] > 0]

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

spike_params_df['timestamp'] = spike_params_df['timestamp'].apply(clean_timestamps)


# === Load spikes ===
if os.path.exists(modified_csv):
    df = pd.read_csv(modified_csv)
else:
    df = (
        spike_params_df
        .sort_values(by=['recording', 'ap_peak_time'])
        .groupby('recording')
        .head(3)
        .reset_index(drop=True)
    )

def calculate_steady_state_voltage(time, voltage, t_start, t_end, rate):
    start_idx = np.abs(time - t_start).argmin()
    end_idx = np.abs(time - t_end).argmin()
    return np.mean(voltage[start_idx:end_idx])

def on_slider_change(val):
    global highlight_window
    highlight_window = int(val)
    update_plot()

current_ap_index = 0
last_steady_state_voltage = None
add_spike_mode = False
dragging_line = None
time = None
voltage = None
draggable_lines = {}

fig = plt.figure(figsize=(12, 14))
gs = fig.add_gridspec(3, 1, height_ratios=[0.45, 0.45, 0.05])
ax_full = fig.add_subplot(gs[0])
ax_zoom = fig.add_subplot(gs[1])
fig.subplots_adjust(hspace=0.5)

slider_ax = fig.add_axes([0.25, 0.13, 0.5, 0.03])
highlight_slider = Slider(
    ax=slider_ax,
    label='Highlight Window (ms)',
    valmin=1,
    valmax=300,
    valinit=highlight_window,
    valstep=1
)
highlight_slider.on_changed(on_slider_change)



delete_ax = plt.axes([0.1, 0.05, 0.1, 0.04])
add_ax = plt.axes([0.22, 0.05, 0.1, 0.04])
save_ax = plt.axes([0.34, 0.05, 0.1, 0.04])

delete_button = Button(delete_ax, 'Delete Spike')
add_button = Button(add_ax, 'Add Spike')
save_button = Button(save_ax, 'Save CSV')
# Text box to show Add Spike Mode status
status_text = fig.text(0.75, 0.06, 'Add Spike: OFF', fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black'))
# Warning text for >3 spikes
warning_text = fig.text(0.75, 0.93, '', fontsize=10, color='red', bbox=dict(facecolor='white', edgecolor='red'))


def update_plot():
    global last_steady_state_voltage, time, voltage
    draggable_lines.clear()

    if current_ap_index >= len(df):
        print("AP index out of range")
        return

    sweep_row = df.iloc[current_ap_index]
    recording = sweep_row['recording']
    sweep_number = sweep_row['sweep_number']
    local_sweep = sweep_row['local_sweep']
    path = sweep_row['path']
    peak_time = sweep_row['ap_peak_time']

    abf = pyabf.ABF(path)
    abf.setSweep(local_sweep)
    time = abf.sweepX
    voltage = abf.sweepY - LJP_CORRECTION_MV

    length = len(time)
    offset = int(length * 0.015625)
    time = np.concatenate([np.zeros(offset), time])[:-offset]

    globals()['time'] = time
    globals()['voltage'] = voltage


    ax_full.clear()
    ax_full.plot(time, voltage)
    ax_full.set_title(f"Spike {current_ap_index+1} of {len(df)} - Full Sweep - Recording: {recording}, Sweep: {sweep_number}")
    ax_full.set_xlabel('Time (s)')
    ax_full.set_ylabel('Voltage (mV)')

    peak_idx = np.abs(time - peak_time).argmin()
    window_size = int(highlight_window * abf.sampleRate / 1000)
    start_idx = max(0, peak_idx - window_size)
    end_idx = min(len(time), peak_idx + window_size)
    ax_full.plot(time[start_idx:end_idx], voltage[start_idx:end_idx], color='red', linewidth=.7)

    if last_steady_state_voltage is None or last_steady_state_voltage['sweep'] != sweep_number:
        steady = calculate_steady_state_voltage(time, voltage, timescale_start, timescale_end, abf.sampleRate)
        last_steady_state_voltage = {'sweep': sweep_number, 'voltage': steady}
    else:
        steady = last_steady_state_voltage['voltage']

    ax_full.axhline(y=steady, color='black', linestyle='--')
    ax_full.axvline(x=timescale_start, color='grey', linestyle='--')
    ax_full.axvline(x=timescale_end, color='grey', linestyle='--')

    ax_zoom.clear()
    ax_zoom.plot(time[start_idx:end_idx], voltage[start_idx:end_idx], color='red', linewidth=.7)
    
    # Set xlim centered on peak_time ± half highlight window
    window_seconds = highlight_window / 1000  # convert ms to seconds
    xmin = max(time[0], peak_time - window_seconds / 2)
    xmax = min(time[-1], peak_time + window_seconds / 2)
    ax_zoom.set_xlim(xmin, xmax)

    ax_zoom.set_ylim(np.min(voltage[start_idx:end_idx]), np.max(voltage[start_idx:end_idx]))

    spike_params = df[(df['recording'] == recording) & 
                      (df['sweep_number'] == sweep_number)]
    
    # === Show warning if >3 spikes in current trace ===
    if len(spike_params) > 3:
        warning_text.set_text(f"⚠ Warning: {len(spike_params)} spikes in this trace")
    else:
        warning_text.set_text('')
    

    if not spike_params.empty:
        closest_idx = np.abs(spike_params['ap_peak_time'] - peak_time).argmin()
        ap = spike_params.iloc[closest_idx]

        draggable_lines['ap_threshold'] = ax_zoom.axhline(y=ap['ap_threshold'], color='green', linestyle='--', picker=5, linewidth = .5)
        ax_zoom.scatter(peak_time, ap['ap_peak_voltage'], color='blue', zorder=5)
        ax_zoom.plot([ap['first_crossing'], ap['second_crossing']], [ap['half_voltage'], ap['half_voltage']], color='black', linewidth=1)
        ax_zoom.axhline(y=ap['half_voltage'], color='orange', linestyle='--', linewidth = .5)

        draggable_lines['first_crossing'] = ax_zoom.axvline(x=ap['first_crossing'], color='brown', linestyle='--', picker=5, linewidth = .5)
        draggable_lines['second_crossing'] = ax_zoom.axvline(x=ap['second_crossing'], color='brown', linestyle='--', picker=5, linewidth = .5)

        draggable_lines['ahp_line'] = ax_zoom.plot(
            [ap['ahp_peak_time'], ap['ahp_peak_time']],
            [ap['ap_threshold'], ap['ahp_peak']],
            color='grey',
            linewidth=0.5,
            picker=5  # Enable picking for dragging
        )[0]
        
        draggable_lines['ahp_peak'] = ax_zoom.scatter(
            ap['ahp_peak_time'], ap['ahp_peak'],
            color='purple',
            picker=None,  # Disable picking, no dragging here
            zorder=5
        )

        

        ap_number = spike_params.index.get_loc(sweep_row.name) + 1
        ax_zoom.set_title(f"Recording: {recording}, Sweep: {sweep_number}, AP #{ap_number}")

    fig.canvas.draw_idle()

def on_key(event):
    global current_ap_index
    if event.key == 'right':
        current_ap_index = (current_ap_index + 1) % len(df)
    elif event.key == 'left':
        current_ap_index = (current_ap_index - 1) % len(df)
    update_plot()

def delete_spike(event):
    global current_ap_index
    if len(df) > 0:
        df.drop(df.index[current_ap_index], inplace=True)
        df.reset_index(drop=True, inplace=True)
        current_ap_index = max(0, current_ap_index - 1)
        update_plot()

def toggle_add_spike(event):
    global add_spike_mode
    add_spike_mode = not add_spike_mode

    # Update label text and color
    if add_spike_mode:
        status_text.set_text('Add Spike: ON')
        status_text.set_color('green')
    else:
        status_text.set_text('Add Spike: OFF')
        status_text.set_color('black')

    fig.canvas.draw_idle()

def on_press(event):
    global dragging_line
    if event.inaxes != ax_zoom:
        return

    for key, artist in draggable_lines.items():
        if hasattr(artist, 'contains'):
            contains, _ = artist.contains(event)
            if contains and key in ['ahp_peak', 'first_crossing', 'second_crossing', 'ap_threshold']:
                dragging_line = key
                fig.canvas.draw_idle()
                break

def on_release(event):
    global dragging_line
    dragging_line = None

def on_motion(event):
    global dragging_line, time, voltage

    if dragging_line is None or event.inaxes != ax_zoom:
        return

    sweep_row = df.iloc[current_ap_index]
    idx = sweep_row.name

    if dragging_line in ['first_crossing', 'second_crossing']:
        new_x = event.xdata
        if new_x is None:
            return
        df.at[idx, dragging_line] = new_x
        draggable_lines[dragging_line].set_xdata([new_x, new_x])

    elif dragging_line == 'ap_threshold':
        new_y = event.ydata
        if new_y is None:
            return

        df.at[idx, 'ap_threshold'] = new_y
        draggable_lines[dragging_line].set_ydata([new_y, new_y])
        # Update AHP amplitude if ahp_peak exists
        ahp_y = df.at[idx, 'ahp_peak']
        if pd.notnull(ahp_y):
            df.at[idx, 'ahp_amplitude'] = new_y - ahp_y


        # Also update half_voltage accordingly:
        peak_voltage = df.at[idx, 'ap_peak_voltage']
        half_voltage = new_y + 0.5 * (peak_voltage - new_y)
        df.at[idx, 'half_voltage'] = half_voltage

        # Update half_voltage orange dashed line
        for line in ax_zoom.lines:
            if line.get_linestyle() == '--' and line.get_color() == 'orange':
                line.set_ydata([half_voltage, half_voltage])
                break

        # Update ahp_line vertical line y-start
        if 'ahp_line' in draggable_lines:
            ahp_peak_y = df.at[idx, 'ahp_peak']
            new_x = df.at[idx, 'ahp_peak_time']
            draggable_lines['ahp_line'].set_ydata([new_y, ahp_peak_y])
            draggable_lines['ahp_line'].set_xdata([new_x, new_x])

    elif dragging_line == 'ahp_peak':
        new_x = event.xdata
        if new_x is None:
            return

        # Snap new_x to within time limits
        if new_x < time[0]:
            new_x = time[0]
        elif new_x > time[-1]:
            new_x = time[-1]

        closest_idx = np.abs(time - new_x).argmin()
        new_y = voltage[closest_idx]

        df.at[idx, 'ahp_peak_time'] = time[closest_idx]
        df.at[idx, 'ahp_peak'] = new_y
        df.at[idx, 'ahp_amplitude'] = df.at[idx, 'ap_threshold'] - new_y

        draggable_lines['ahp_peak'].set_offsets([[time[closest_idx], new_y]])

        ap_threshold = df.at[idx, 'ap_threshold']

        draggable_lines['ahp_line'].set_xdata([time[closest_idx], time[closest_idx]])
        draggable_lines['ahp_line'].set_ydata([ap_threshold, new_y])

    fig.canvas.draw_idle()


def on_click(event):
    global add_spike_mode, df, current_ap_index

    if not add_spike_mode or event.inaxes != ax_full:
        return

    sweep_row = df.iloc[current_ap_index]
    recording = sweep_row['recording']
    sweep_number = sweep_row['sweep_number']
    local_sweep = sweep_row['local_sweep']
    path = sweep_row['path']

    abf = pyabf.ABF(path)
    abf.setSweep(local_sweep)
    time = abf.sweepX
    voltage = abf.sweepY - LJP_CORRECTION_MV
    offset = int(len(time) * 0.015625)
    time = np.concatenate([np.zeros(offset), time])[:-offset]
    interp_func = interp1d(time, voltage, bounds_error=False, fill_value="extrapolate")

    detection_window_ms = 10
    half_window = detection_window_ms / 1000
    start_time = event.xdata - half_window
    end_time = event.xdata + half_window

    if start_time < time[0] or end_time > time[-1]:
        print("Click too close to edge of trace.")
        return

    start_idx = np.searchsorted(time, start_time)
    end_idx = np.searchsorted(time, end_time)
    time_slice = time[start_idx:end_idx]
    voltage_slice = voltage[start_idx:end_idx]
    peak_idx = np.argmax(voltage_slice)
    new_peak_time = float(time_slice[peak_idx])
    new_peak_voltage = voltage_slice[peak_idx]

    ap_threshold = new_peak_voltage - 20
    half_voltage = ap_threshold + 0.5 * (new_peak_voltage - ap_threshold)
    ahp_time = new_peak_time + 0.01
    ahp_voltage = float(interp_func(ahp_time))
    first_crossing = new_peak_time - 0.001
    second_crossing = new_peak_time + 0.001

    # --- Initialize new spike with all columns as NaN ---
    new_spike = {col: np.nan for col in df.columns}

     # --- Fill in known fields ---
    new_spike.update({
        'recording': recording,
        'sweep_number': sweep_number,
        'local_sweep': local_sweep,
        'path': path,
        'ap_peak_time': float(new_peak_time),
        'ap_peak_voltage': float(new_peak_voltage),
        'ap_threshold': float(ap_threshold),
        'half_voltage': float(half_voltage),
        'first_crossing': float(first_crossing),
        'second_crossing': float(second_crossing),
        'ahp_peak_time': float(ahp_time),
        'ahp_peak': float(ahp_voltage),
    })
    
    # Set timestamp to match ap_peak_time
    new_spike['timestamp'] = float(new_peak_time)

    df.loc[len(df)] = new_spike
    df.sort_values(by=['recording', 'sweep_number', 'ap_peak_time'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    matching_rows = df[
        (df['recording'] == recording) &
        (df['sweep_number'] == sweep_number) &
        (df['ap_peak_time'] == new_peak_time)
    ]
    if not matching_rows.empty:
        current_ap_index = matching_rows.index[0]
    else:
        current_ap_index = 0

    update_plot()




def save_csv(event):
    # Merge back metadata from original spike_params_df
    metadata_cols = ['recording', 'sweep_number', 'current_injection', 'spike_count',
                     'condition', 'timestamp', 'half_width', 'ap_amplitude', 
                     'ahp_amplitude', 'sweep_threshold', 'ap_count']

    # Deduplicate original data to avoid merge explosion
    metadata_unique = spike_params_df[metadata_cols].drop_duplicates(subset=['recording', 'sweep_number'])

    # Merge missing metadata into df (left join on recording & sweep_number)
    updated_df = df.merge(metadata_unique, on=['recording', 'sweep_number'], how='left', suffixes=('', '_meta'))

    # Fill only missing values in df from the merged metadata
    for col in metadata_cols:
        if col in ['recording', 'sweep_number']:
            continue  # These were merge keys, so they won't have '_meta' suffix
        meta_col = col + '_meta'
        if meta_col in updated_df.columns:
            for col in metadata_cols:
                if col in ['recording', 'sweep_number']:
                    continue  # Skip merge keys
                meta_col = col + '_meta'
                if meta_col in updated_df.columns:
                    # Clean up known problem types (e.g., empty strings)
                    cleaned_meta = updated_df[meta_col].replace('', np.nan)
                    df[col] = df[col].combine_first(cleaned_meta)
   
    # Recalculate half_voltage before saving
    df['half_voltage'] = df.apply(
        lambda row: row['ap_threshold'] + 0.5 * (row['ap_peak_voltage'] - row['ap_threshold'])
        if pd.notnull(row['ap_threshold']) and pd.notnull(row['ap_peak_voltage']) else np.nan,
        axis=1
    )

    # Save final version
    df['half_width'] = (df['second_crossing'] - df['first_crossing']) * 1000  # convert to ms

    df['ap_peak_time'] = df['ap_peak_time'].apply(
    lambda x: x[0] if isinstance(x, list) else float(x))

    output_path = os.path.join(root, "3_modified_spikes.csv")
    df.to_csv(output_path, index=False)

    averaged_rows = []
    for rec in df['recording'].unique():
        rec_df = df[df['recording'] == rec].sort_values('ap_peak_time').head(3)
        if not rec_df.empty:
            avg_row = rec_df.iloc[0].copy()
            for col in ['ap_peak_voltage', 'ahp_peak', 'half_width', 'ap_amplitude', 'ahp_amplitude', 'ap_threshold']:
                avg_row[col] = rec_df[col].mean()
            avg_row['n_aps_averaged'] = len(rec_df)
            averaged_rows.append(avg_row)
    
    if averaged_rows:
        avg_df = pd.DataFrame(averaged_rows)
        avg_output_path = os.path.join(root, "3_modified_averaged_parameters.csv")
        avg_df.to_csv(avg_output_path, index=False)
        print(f"Averaged parameters saved to {avg_output_path}")
        print(f"Saved to {output_path}")


update_plot()
delete_button.on_clicked(delete_spike)
add_button.on_clicked(toggle_add_spike)
save_button.on_clicked(save_csv)
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
