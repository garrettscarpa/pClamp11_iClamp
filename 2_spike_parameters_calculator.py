import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pyabf
import os
import re
from tqdm import tqdm  # Import tqdm for progress bar
from scipy.interpolate import interp1d

###ADD CONDITION TO ALL_PARAMETERS.CSV

# Path to the CSV file
file_path = '/Users/gs075/Desktop/test'


detection_window = 20
highlight_window = 30       # for zoomed-in pop-up plot; in milliseconds
display_window = 20
timescale_start = 0.25      # Start of timescale for steady state calculation in seconds (e.g., 250 ms)
timescale_end = 1.25        # End of timescale for steady state calculation in seconds (e.g., 1250 ms)
offset = 0.015625
LJP_CORRECTION_MV = 14.681  # LJP Correction (in mV) — this value will be subtracted from all voltage measurements


# Load the action potential data
df_all = pd.read_csv(file_path + "/1_thresholded_data.csv")

# Preprocessing
df = df_all[df_all['spike_count'] > 0]  # Filter out rows with zero spike_count
unique_recordings = df['recording'].unique()

# Convert timestamps from float64 to floating point numbers
def clean_timestamps(timestamps):
    if isinstance(timestamps, str):
        timestamps = [
            float(re.sub(r'np\.float64\((.*?)\)', r'\1', t).strip()) if t.strip() != '' else None
            for t in timestamps.strip('[]').split(',')
        ]
    
    # Remove None values
    timestamps = [t for t in timestamps if t is not None]
    
    return timestamps

df['timestamps'] = df['timestamps'].apply(clean_timestamps)

# Create 'peaks' dataframe with new row for each spike
new_rows = []
for _, row in df.iterrows():
    recording = row['recording']
    sweep_number = row['sweep_number']
    current_injection = row['current_injection']
    spike_count = row['spike_count']
    threshold = row['threshold']
    condition = row['condition']
    timestamps = row['timestamps']

    for timestamp in timestamps:
        new_row = {
            'recording': recording,
            'sweep_number': sweep_number,
            'current_injection': current_injection,
            'spike_count': spike_count,
            'threshold': threshold,
            'condition': condition,
            'timestamp': timestamp
        }
        new_rows.append(new_row)

peaks = pd.DataFrame(new_rows)

# Initialize some variables for cycling through the recordings and sweeps
current_recording_index = 0
current_sweep_index = 0
current_peak_index = 0

# Create a new dataframe for the data
data = pd.DataFrame(columns=[
    'recording', 'sweep_number', 'current_injection', 'spike_count', 'threshold', 'condition',
    'timestamp', 'AP_peak_voltage', 'AP_peak_time', 'Threshold', 'AHP_peak', 'AHP_peak_time', 
    'Half_Voltage', 'Half_Width', 'AP_Amplitude', 'AHP_Amplitude', 'first_crossing_idx', 'second_crossing_idx'
])


def calculate_ap_parameters(time, voltage, ap_peak_time, threshold, abf_sample_rate, detection_window):
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    import numpy as np

    ap_peak_time = ap_peak_time - time[0]  # Adjust for offset

    window_size = int(detection_window * abf_sample_rate / 1000)
    ap_peak_idx = np.abs(time - ap_peak_time).argmin()
    start_idx = max(0, ap_peak_idx - window_size)
    end_idx = min(len(time), ap_peak_idx + window_size)

    ap_peak_voltage = np.max(voltage[start_idx:end_idx])
    ap_peak_time_index = np.argmax(voltage[start_idx:end_idx]) + start_idx
    ap_peak_time = time[ap_peak_time_index]

    threshold_voltage = np.percentile(voltage[start_idx:ap_peak_idx], 95)

    ahp_peak_window_size = int(detection_window * abf_sample_rate / 1000)
    ahp_peak_start_idx = ap_peak_time_index
    ahp_peak_end_idx = min(len(time), ahp_peak_start_idx + ahp_peak_window_size)
    potential_ahp_peak = np.min(voltage[ahp_peak_start_idx:ahp_peak_end_idx])
    ahp_peak_time_index = np.argmin(voltage[ahp_peak_start_idx:ahp_peak_end_idx]) + ahp_peak_start_idx
    ahp_peak_time = time[ahp_peak_time_index]

    half_voltage = threshold_voltage + 0.5 * (ap_peak_voltage - threshold_voltage)

    voltage_slice = voltage[start_idx:end_idx]
    time_slice = time[start_idx:end_idx]
    crossings = np.where(np.isclose(voltage_slice, half_voltage, atol=8))[0]

    if len(crossings) >= 2:
        unique_voltage, unique_indices = np.unique(voltage_slice, return_index=True)
        unique_time = time_slice[unique_indices]

        if len(unique_voltage) >= 2:
            interp_voltage_to_time = interp1d(unique_voltage, unique_time, kind='linear', fill_value="extrapolate")

            first_crossing_voltage = voltage[crossings[0] + start_idx]
            second_crossing_voltage = voltage[crossings[1] + start_idx]

            first_crossing = interp_voltage_to_time(first_crossing_voltage)
            second_crossing = interp_voltage_to_time(second_crossing_voltage)

            first_crossing_ms = first_crossing * 1000
            second_crossing_ms = second_crossing * 1000
            half_width = second_crossing_ms - first_crossing_ms
        else:
            print(f"[Warning] Not enough unique voltages for interpolation at AP time {ap_peak_time:.4f}s")
            first_crossing = second_crossing = half_width = np.nan
    else:
        print(f"[Warning] <2 half-voltage crossings at AP time {ap_peak_time:.4f}s")
        first_crossing = second_crossing = half_width = np.nan

    ap_amplitude = ap_peak_voltage - threshold_voltage
    ahp_amplitude = threshold_voltage - potential_ahp_peak


    return (
        ap_peak_voltage, ap_peak_time, threshold_voltage, potential_ahp_peak, ahp_peak_time, half_voltage, half_width, ap_amplitude, ahp_amplitude, first_crossing, second_crossing)


# Initialize a dictionary to hold counters for each sweep
sweep_counters = {}

# Process each peak (recording, sweep_number, etc.)
for idx, row in tqdm(peaks.iterrows(), total=peaks.shape[0], desc="Processing Peaks", unit="peak"):
    recording = row['recording']
    sweep_number = row['sweep_number']
    ap_peak_time = row['timestamp']
    abf = pyabf.ABF(row['recording'])
    abf.setSweep(row['sweep_number'])

    # Get the voltage trace and time
    time = abf.sweepX
    voltage = abf.sweepY - LJP_CORRECTION_MV
    
    # Apply the offset to the time array before any calculations
    length = len(time)  # in samples
    mf = 0.015625  # offset fraction (assuming this is correct as per your setup)
    offset = int(length * mf)
    time = np.concatenate([np.zeros(offset, dtype=np.float64), time])  # Apply offset
    time = time[:-offset]  # Remove the excess offset if necessary
    
    # Calculate the action potential parameters
    ap_peak_voltage, ap_peak_time, threshold_voltage, potential_ahp_peak, ahp_peak_time, half_voltage, half_width, ap_amplitude, ahp_amplitude, first_crossing, second_crossing = calculate_ap_parameters(
        time, voltage, ap_peak_time, row['threshold'], abf.sampleRate, detection_window)

    # Create a temporary DataFrame for the current row
    new_row = pd.DataFrame([{
        'recording': row['recording'],
        'sweep_number': row['sweep_number'],
        'current_injection': row['current_injection'],
        'spike_count': row['spike_count'],
        'Sweep_threshold': row['threshold'],
        'condition': row['condition'],
        'timestamp': row['timestamp'],
        'AP_peak_voltage': ap_peak_voltage,
        'AP_peak_time': ap_peak_time,
        'AP_Threshold': threshold_voltage,
        'AHP_peak': potential_ahp_peak,
        'AHP_peak_time': ahp_peak_time,
        'Half_Voltage': half_voltage,
        'first_crossing': first_crossing,
        'second_crossing': second_crossing,
        'Half_Width': half_width,
        'AP_Amplitude': ap_amplitude,
        'AHP_Amplitude': ahp_amplitude
    }])

    sweep_key = (recording, sweep_number)
    
    if sweep_key not in sweep_counters:
        sweep_counters[sweep_key] = 1
    else:
        sweep_counters[sweep_key] += 1
    
    new_row['AP_count'] = sweep_counters[sweep_key]
    

    # Concatenate the new row to the 'data' DataFrame
    if not new_row.empty and not new_row.isna().all(axis=1).any():
        data = pd.concat([data, new_row], ignore_index=True)
    data = data.drop(columns=['threshold', 'Threshold', 'first_crossing_idx', 'second_crossing_idx'], errors='ignore')

# Convert all column titles to lowercase
data.columns = data.columns.str.lower()

# Group by recording, get lowest sweep number with spikes (rheobase)
rheobase_rows = []
for rec in data['recording'].unique():
    rec_data = data[(data['recording'] == rec) & (data['current_injection'] > 0)]
    if not rec_data.empty:
        min_sweep = rec_data['sweep_number'].min()
        spike_rows = rec_data[(rec_data['sweep_number'] == min_sweep) & (rec_data['ap_count'] > 0)]
        if not spike_rows.empty:
            top3 = spike_rows.sort_values('ap_peak_time').head(3)
            avg_row = top3.iloc[0].copy()
            for col in ['ap_peak_voltage', 'ahp_peak', 'half_width', 'ap_amplitude', 'ahp_amplitude', 'ap_threshold']:
                avg_row[col] = top3[col].mean()
            avg_row['n_aps_averaged'] = len(top3)
            rheobase_rows.append(avg_row)

# Save to new CSV
if rheobase_rows:
    avg_df = pd.DataFrame(rheobase_rows)
    avg_df.to_csv(os.path.join(file_path, '2_averaged_parameters.csv'), index=False)


data.to_csv(os.path.join(file_path, '2_all_parameters.csv'), index=False)


### === NEW SECTION: Calculate RMP and Input Resistance per recording === ###
passive_features = []

print("\nCalculating RMP and Input Resistance for each recording using 'ssv' column...")

for recording in tqdm(unique_recordings, desc="Processing Recordings", unit="recording"):
    
    rec_sweeps = df_all[df_all['recording'] == recording][['sweep_number', 'current_injection', 'ssv']]
    # Extract ssv values at 0 and -20 pA
    ssv_at_0 = rec_sweeps.loc[rec_sweeps['current_injection'] == 0, 'ssv']
    ssv_at_minus_20 = rec_sweeps.loc[rec_sweeps['current_injection'] == -20, 'ssv']

    if not ssv_at_0.empty:
        rmp = ssv_at_0.values[0]
    else:
        rmp = np.nan

    if not ssv_at_0.empty and not ssv_at_minus_20.empty:
        v0 = ssv_at_0.values[0] / 1000  # convert mV to V
        v20 = ssv_at_minus_20.values[0] / 1000  # convert mV to V
        delta_v = v20 - v0
        delta_i = (-20e-12)  # -20 pA in Amps

        input_resistance = delta_v / delta_i  # Ohms
        input_resistance = input_resistance / 1e6  # Convert to MΩ
    else:
        input_resistance = np.nan

    passive_features.append({
        'recording': recording,
        'RMP_mV': rmp,
        'Input_Resistance_MOhm': input_resistance
    })

passive_df = pd.DataFrame(passive_features)
passive_df.to_csv(os.path.join(file_path, '2_passive_membrane_features.csv'), index=False)

print("RMP and Input Resistance calculations saved to '2_passive_membrane_features.csv'")

