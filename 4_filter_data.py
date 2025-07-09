import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# User-defined configuration
parent_folder = '/Volumes/BWH-HVDATA/Individual Folders/Garrett/PatchClamp/Analyses/RSP_Hannah_Farnsworth'
freq_range = [2, 600]  # Frequency range (spike count)
input_resistance_range = [10, 200]  # in MΩ, [min, max]
rmp_range = [-100, -50]  # acceptable RMP range in mV
half_width_range = [0.0, 4.0]  # in ms, adjust as needed
ap_amplitude_threshold = 0  # in mV
ap_peak_voltage_threshold = -100  # in mV (new exclusion)
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
            if rmp_range[0] <= rmp <= rmp_range[1]:
                rmp_valid_recordings.append(recording)
            else:
                excluded_records.append({
                    'recording': recording,
                    'reason': f'RMP outside {rmp_range[0]} to {rmp_range[1]} mV',
                    'RMP': rmp
                })

        else:
            excluded_records.append({
                'recording': recording,
                'reason': 'No RMP measurement',
                'RMP': np.nan
            })

    rmp_filtered_data = filtered_condition_data[filtered_condition_data['recording'].isin(rmp_valid_recordings)]
    
    # Filter based on Input Resistance within a range
    ir_valid_recordings = []
    for recording in np.unique(rmp_filtered_data['recording']):
        ir_series = rmp_filtered_data.loc[rmp_filtered_data['recording'] == recording, 'Input_Resistance_MOhm']
        if not ir_series.isnull().all():
            ir = ir_series.iloc[0]
            if input_resistance_range[0] <= ir <= input_resistance_range[1]:
                ir_valid_recordings.append(recording)
            else:
                excluded_records.append({
                    'recording': recording,
                    'reason': f'IR outside {input_resistance_range[0]}–{input_resistance_range[1]} MΩ',
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
            
        # Filter based on half_width from averaged_data
    hw_valid_recordings = []
    for recording in ap_valid_recordings:
        ap_row = averaged_data[averaged_data['recording'] == recording]
        if not ap_row.empty:
            hw = ap_row['half_width'].values[0]  # assuming 'half_width' is a column in averaged_data
            if half_width_range[0] <= hw <= half_width_range[1]:
                hw_valid_recordings.append(recording)
            else:
                excluded_records.append({
                    'recording': recording,
                    'reason': f'Half-width outside {half_width_range[0]}–{half_width_range[1]} ms',
                    'half_width': hw
                })
        else:
            excluded_records.append({
                'recording': recording,
                'reason': 'No half-width data',
                'half_width': np.nan
            })
    
    final_filtered_data = ir_filtered_data[ir_filtered_data['recording'].isin(hw_valid_recordings)]
    filtered_data.append(final_filtered_data)


# Combine all filtered data and save
filtered_data_df = pd.concat(filtered_data, ignore_index=True)
filtered_data_df.to_csv(os.path.join(parent_folder, '3_filtered_data.csv'), index=False)

# Save exclusion log
excluded_df = pd.DataFrame(excluded_records)
excluded_df.to_csv(os.path.join(parent_folder, 'excluded_data.csv'), index=False)


# Step 1: Assign unique mouse ID from recording path
filtered_data_df['mouse_id'] = filtered_data_df['recording'].apply(
    lambda x: "/".join(x.split('/')[-1].split('_')[:3])  # e.g., '2025_05_14'
)

# Step 2: Compute max firing rate per recording
max_firing_df = filtered_data_df.groupby(['recording', 'mouse_id'])['spike_count'].max().reset_index()

# Step 3: Bin the spike counts
num_bins = 40
bins = np.linspace(0, max_firing_df['spike_count'].max(), num_bins)
max_firing_df['bin'] = np.digitize(max_firing_df['spike_count'], bins) - 1  # Get bin index
print(max_firing_df[['spike_count', 'bin']])

# Step 4: Create bin x mouse matrix
bin_mouse_matrix = max_firing_df.groupby(['bin', 'mouse_id']).size().unstack(fill_value=0)

# Step 5: Set up color map
cmap = plt.get_cmap('tab20b')
mouse_ids = bin_mouse_matrix.columns
color_map = {mouse: cmap(i % 20) for i, mouse in enumerate(mouse_ids)}

# Step 6: Plot
# Plot stacked histogram using bin_mouse_matrix
num_bins = len(bins) - 1
bin_centers = bins[:-1] + np.diff(bins)[0] / 2
bottom = np.zeros(num_bins)

# Ensure matrix is complete
bin_mouse_matrix = bin_mouse_matrix.reindex(index=range(num_bins), columns=mouse_ids, fill_value=0)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for mouse_id in mouse_ids:
    counts = bin_mouse_matrix[mouse_id].values
    ax.bar(
        bin_centers,
        counts,
        bottom=bottom,
        width=np.diff(bins)[0] * 0.9,
        color=color_map[mouse_id],
        label=mouse_id,
        align='center'
    )
    bottom += counts

ax.set_title('Histogram of Max Firing Rates Colored by Mouse ID')
ax.set_xlabel('Max Firing Rate (Hz)')
ax.set_ylabel('Number of Cells')
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(title='Recording Date', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save
histogram_path = os.path.join(parent_folder, 'max_firing_rate_by_mouse.png')
plt.savefig(histogram_path, dpi=300)
plt.show()


# Step 7: Save
histogram_path = os.path.join(parent_folder, 'max_firing_rate_by_mouse.png')
plt.savefig(histogram_path, dpi=300)
plt.show()