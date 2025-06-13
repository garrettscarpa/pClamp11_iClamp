import os
import pandas as pd
import numpy as np

###############################################################################
# User-defined configuration
parent_folder = '/Users/gs075/Desktop/final_analysis_HF/'
freq_range = [61, 500]  # Frequency range (spike count)
rmp_threshold = -50
ap_amplitude_threshold = 40  # in mV
ap_peak_voltage_threshold = 10  # in mV (new exclusion)
input_resistance_threshold = 50  # in MΩ (new exclusion)
###############################################################################

# Load the thresholded AP data
thresholded_path = os.path.join(parent_folder, '1_thresholded_data.csv')
thresholded_data = pd.read_csv(thresholded_path)

# Load the averaged parameters data (for AP amplitude, AP peak voltage, etc.)
averaged_path = os.path.join(parent_folder, '2_averaged_parameters.csv')
averaged_data = pd.read_csv(averaged_path)

# Load passive membrane features (RMP and input resistance)
passive_features_path = os.path.join(parent_folder, '2_passive_membrane_features.csv')
passive_features = pd.read_csv(passive_features_path)

# Merge passive features into thresholded data based on recording
# Assuming 'recording' is the key in all datasets
thresholded_data = thresholded_data.merge(
    passive_features[['recording', 'RMP_mV', 'Input_Resistance_MOhm']],
    on='recording',
    how='left'
)

filtered_data = []
excluded_records = []

# Detect condition folders
conditions = [folder for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]

# Process each condition
for condition in conditions:
    condition_data = thresholded_data[thresholded_data['condition'] == condition]

    # Filter by frequency range
    valid_recordings = []
    for recording in np.unique(condition_data['recording']):
        recording_data = condition_data[condition_data['recording'] == recording]
        max_spike_count = recording_data['spike_count'].max()
        
        if freq_range[0] <= max_spike_count <= freq_range[1]:
            valid_recordings.append(recording)
        else:
            if max_spike_count < freq_range[0]:
                reason = f"<{freq_range[0]} Hz"
            elif max_spike_count > freq_range[1]:
                reason = f">{freq_range[1]} Hz"
            else:
                reason = f"within {freq_range[0]}-{freq_range[1]} Hz"

            excluded_records.append({
                'recording': recording,
                'reason': f'Frequency {reason}',
                'max_firing_rate': max_spike_count
            })

    filtered_condition_data = condition_data[condition_data['recording'].isin(valid_recordings)]

    # Filter based on RMP (use merged rmp column)
    rmp_valid_recordings = []
    for recording in np.unique(filtered_condition_data['recording']):
        # Get RMP from passive_features merge (not from zero current sweep in thresholded_data)
        rmp_series = filtered_condition_data.loc[filtered_condition_data['recording'] == recording, 'RMP_mV']
        if not rmp_series.isnull().all():
            rmp = rmp_series.iloc[0]
            if rmp <= rmp_threshold:
                rmp_valid_recordings.append(recording)
            else:
                excluded_records.append({
                    'recording': recording,
                    'reason': 'RMP > -50 mV',
                    'RMP': rmp
                })
        else:
            excluded_records.append({
                'recording': recording,
                'reason': 'No RMP measurement',
                'RMP': np.nan
            })

    rmp_filtered_data = filtered_condition_data[filtered_condition_data['recording'].isin(rmp_valid_recordings)]
    
    # Filter based on Input Resistance
    ir_valid_recordings = []
    for recording in np.unique(rmp_filtered_data['recording']):
        ir_series = rmp_filtered_data.loc[rmp_filtered_data['recording'] == recording, 'Input_Resistance_MOhm']
        if not ir_series.isnull().all():
            ir = ir_series.iloc[0]
            if ir >= input_resistance_threshold:
                ir_valid_recordings.append(recording)
            else:
                excluded_records.append({
                    'recording': recording,
                    'reason': 'IR < 50',
                    'IR': ir
                })
        else:
            excluded_records.append({
                'recording': recording,
                'reason': 'No IR measurement',
                'IR': np.nan
            })

    ir_filtered_data = rmp_filtered_data[rmp_filtered_data['recording'].isin(ir_valid_recordings)]


    # Filter by AP amplitude, AP peak voltage, and Input Resistance (from averaged_data)
    ap_valid_recordings = []
    for recording in np.unique(ir_filtered_data['recording']):
        ap_row = averaged_data[averaged_data['recording'] == recording]
        if not ap_row.empty:
            ap_amp = ap_row['ap_amplitude'].values[0]
            ap_peak = ap_row['ap_peak_voltage'].values[0]

            if ap_amp < ap_amplitude_threshold:
                excluded_records.append({
                    'recording': recording,
                    'reason': 'AP amplitude < 40 mV',
                    'ap_amplitude': ap_amp
                })
                continue

            if ap_peak < ap_peak_voltage_threshold:
                excluded_records.append({
                    'recording': recording,
                    'reason': 'AP peak voltage < 10 mV',
                    'ap_peak_voltage': ap_peak
                })
                continue

            ap_valid_recordings.append(recording)
        else:
            excluded_records.append({
                'recording': recording,
                'reason': 'No AP amplitude data',
                'ap_amplitude': np.nan
            })

    final_filtered_data = ir_filtered_data[ir_filtered_data['recording'].isin(ap_valid_recordings)]
    filtered_data.append(final_filtered_data)


# Combine all filtered data and save
filtered_data_df = pd.concat(filtered_data, ignore_index=True)
filtered_data_df.to_csv(os.path.join(parent_folder, '3_filtered_data.csv'), index=False)

# Save exclusion log
excluded_df = pd.DataFrame(excluded_records)
excluded_df.to_csv(os.path.join(parent_folder, 'excluded_data.csv'), index=False)
