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
ROOT = '/Users/garrett/Desktop/Analyses/Patch Clamp'

FIG_TITLE = "Sample Data"
ORDERED_CONDITIONS = ['Condition 1', 'Condition 2']  # Conditions to compare
COLOR_MAP = {'Condition 1': 'dimgrey', 'Condition 2': 'tomato'}  # Colors for plotting
DOT_COLOR, IR_STEP, IO_RANGE = 'black', -20, [0, 300]  # Parameters for plotting and analysis
WINDOW_SEC = 0.005  # ±5 ms for AP window (Action Potential)
COMMON_TIME = np.linspace(-WINDOW_SEC, WINDOW_SEC, 1000)  # Time range for AP waveforms
artist_to_data = {}
selected_point_artist = None
selected_annotation = None
redefine_path = True  # Set to True if working on laptop for path adjustment
LJP_CORRECTION_MV = 14.681  # mV, subtract from recorded voltages

# --- Build ABF file index (ONLY ONCE) ---
all_files = []
for root, _, files in os.walk(ROOT):
    for f in files:
        if f.endswith(".abf"):
            all_files.append(os.path.join(root, f))

print(f"Indexed {len(all_files)} ABF files")


# --- Smart matcher for Steps_300 files ---
def extract_index(name):
    m = re.search(r'_(\d{4})', name)
    return int(m.group(1)) if m else None


def find_best_abf(recording):
    base = os.path.basename(recording)
    date = "_".join(base.split("_")[:3])
    rec_idx = extract_index(base)

    candidates = []

    for f in all_files:
        # ONLY use Steps_300 files
        if "Steps_300.abf" not in f:
            continue

        if date not in f:
            continue

        idx = extract_index(f)
        if idx is not None:
            candidates.append((abs(idx - rec_idx), f))

    if not candidates:
        raise FileNotFoundError(f"No Steps_300 file for {recording}")

    best = sorted(candidates, key=lambda x: x[0])[0]

    # DEBUG (keep this!)
    print(f"{recording} → {os.path.basename(best[1])} (Δ={best[0]})")

    return best[1]


# --- Load Updated Data ---
thresholded = pd.read_csv(f"{ROOT}/1_thresholded_data.csv")
all_spikes = pd.read_csv(f"{ROOT}/3_modified_spikes.csv")  # Optional: spike-level data
ap_params_df = pd.read_csv(f"{ROOT}/3_modified_averaged_parameters.csv")  # Averaged AP params per recording
results = pd.read_csv(f"{ROOT}/2_passive_membrane_features.csv")  # RMP and Input Resistance

# Normalize 'recording' to remove paths if needed, and merge based on just the base recording ID
def extract_base_recording(rec):
    return os.path.basename(rec).replace('.abf', '')  # remove path and file extension

# Add base recording ID to all_spikes and results
all_spikes['base_recording'] = all_spikes['recording'].apply(extract_base_recording)
results['base_recording'] = results['recording'].apply(extract_base_recording)

# Drop duplicates and merge
condition_map = all_spikes[['base_recording', 'condition']].drop_duplicates()
results = results.merge(condition_map, on='base_recording', how='left')

filtered_data = pd.read_csv(f"{ROOT}/4_filtered_data.csv")
filtered_data['local_sweep'] = filtered_data.groupby(['recording', 'path'])['sweep_number'].transform(lambda x: x - x.min())


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

# =========================
# SAMPLE SIZE HELPERS (cell n, animal N)
# =========================

def extract_animal_id(rec):
    """
    Animal ID = first 3 underscore-delimited tokens of filename
    Matches logic used in improved script
    """
    return '_'.join(os.path.basename(rec).split('_')[:3])

for df in [results, ap_params_df, filtered_data]:
    if 'recording' in df.columns:
        df['animal_id'] = df['recording'].apply(extract_animal_id)

# --- Helper Functions for Statistics and Plotting ---
def t_test_by_condition(df, col):
    a = df[df['condition'] == ORDERED_CONDITIONS[0]][col]
    b = df[df['condition'] == ORDERED_CONDITIONS[1]][col]
    return stats.ttest_ind(a, b, equal_var=False).pvalue

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
        subset = df[df['condition'] == cond]
        n = subset['recording'].nunique()
        N = subset['animal_id'].nunique()
        
        ax.plot([], [], marker='s',
                label=f"{cond} (n={n}, N={N})",
                color=COLOR_MAP[cond])

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
        subset = df[df['condition'] == cond]
        n = subset['recording'].nunique()
        N = subset['animal_id'].nunique()
        
        ax.plot([], [], marker='s',
                label=f"{cond} (n={n}, N={N})",
                color=COLOR_MAP[cond])

    ax.set_xlabel("")

def onpick(event):
    """
    Universal pick handler for all swarm/strip plots.
    Highlights the clicked point and shows its identity (recording, value, condition).
    Works for AP parameters, rheobase, capacitance, and 300 pA output.
    """
    global selected_point_artist, selected_annotation

    if not isinstance(event.artist, PathCollection):
        return

    # Get the index of the clicked point in the collection
    ind = event.ind
    if len(ind) == 0:
        return

    # Coordinates of the clicked point
    offset = event.artist.get_offsets()[ind[0]]
    x_clicked, y_clicked = offset
    y_clicked = round(float(y_clicked), 4)

    # Find the axis and corresponding x-tick labels
    ax = event.artist.axes
    x_ticks = [tick.get_text() for tick in ax.get_xticklabels()]
    tick_positions = ax.get_xticks()

    if not x_ticks:
        return

    # Map x_clicked to the closest x-tick label (works for jitter/hue)
    x_clicked_str = min(x_ticks, key=lambda t: abs(tick_positions[x_ticks.index(t)] - x_clicked))

    # Clear previous highlight and annotation
    if selected_point_artist:
        selected_point_artist.remove()
        selected_point_artist = None
    if selected_annotation:
        selected_annotation.remove()
        selected_annotation = None

    # Loop through all relevant DataFrames
    dataframes = [results, rheos, ap_params_df, capacitance_data, io_300_rec]
    y_cols = ['RMP', 'IR (megaohm)', 'rheobase', 'value',
              'ap_peak_voltage', 'ahp_peak', 'half_width',
              'ap_amplitude', 'ahp_amplitude', 'ap_threshold',
              'spike_count']

    match_row = None
    matched_col = None
    for df in dataframes:
        if 'condition' not in df.columns:
            continue

        possible_rows = df[df['condition'] == x_clicked_str]
        for col in y_cols:
            if col not in possible_rows.columns:
                continue
            for _, row in possible_rows.iterrows():
                val = row[col]
                if pd.notnull(val) and round(float(val), 4) == y_clicked:
                    match_row = row
                    matched_col = col
                    break
            if match_row is not None:
                break
        if match_row is not None:
            break

    if match_row is None:
        # No match found
        return

    # Get recording/filename
    recording = match_row.get('recording', match_row.get('filename', ''))
    trimmed = recording.split('steps_analyses/')[-1] if 'steps_analyses/' in recording else recording

    # Highlight the selected point
    selected_point_artist = ax.plot(x_clicked_str, y_clicked, 'o', color='red', markersize=6, zorder=10)[0]

    # Annotate it
    selected_annotation = ax.annotate(
        f"{trimmed}\n{matched_col} = {y_clicked} ({x_clicked_str})",
        (x_clicked_str, y_clicked),
        textcoords="offset points",
        xytext=(10, 10),
        ha='left',
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=0.8),
        arrowprops=dict(arrowstyle="->", color='black', lw=0.8),
        zorder=1000
    )

    # Refresh figure
    event.canvas.draw_idle()
    print(f"Clicked on datapoint: Recording = {trimmed}, Condition = {x_clicked_str}, Parameter = {matched_col}")


# --- Step 2: Rheobase Calculation ---
# Rheobase is the minimal current injection needed to generate a spike
rheos = filtered_data[filtered_data['current_injection'] > 0]
rheos = rheos[rheos['spike_count'] > 0].groupby(['recording', 'condition'])['current_injection'].min().reset_index(name='rheobase')
rheos['animal_id'] = rheos['recording'].apply(extract_animal_id)

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
ap_params_df['animal_id'] = ap_params_df['recording'].apply(extract_animal_id)


def load_trace(recording, local_sweep, path=None, abf_dir=ROOT):
    # IGNORE 'path' completely — it is unreliable
    full_path = find_best_abf(recording)

    abf = pyabf.ABF(full_path)
    abf.setSweep(local_sweep)

    time = abf.sweepX
    voltage = abf.sweepY - LJP_CORRECTION_MV

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
    path = group['path'].iloc[0]  # full path relative to abf_dir
    local_sweep = group['local_sweep'].iloc[0]

    peaks = group.sort_values('ap_peak_time')['ap_peak_time'].values[:3]
    try:
        t, v = load_trace(rec, local_sweep, path=path)
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
fig, axs = plt.subplots(4, 4, figsize=(24, 18))
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
axs[2].set_xlabel('')  # <-- add this line

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
    if len(waveforms_cond) == 0:
        print(f"No waveforms for {cond}, skipping")
        continue

    waves = np.vstack(waveforms_cond)

    mean_wave = np.mean(waves, axis=0)
    sem_wave = np.std(waves, axis=0) / np.sqrt(len(waveforms_cond))
    subset = ap_params_df[ap_params_df['condition'] == cond]
    n = subset['recording'].nunique()
    N = subset['animal_id'].nunique()
    
    ax_waveforms.plot(COMMON_TIME, mean_wave,
                      label=f"{cond} (n={n}, N={N})",
                      color=COLOR_MAP[cond])
    ax_waveforms.fill_between(COMMON_TIME, mean_wave - sem_wave, mean_wave + sem_wave, 
                               color=COLOR_MAP[cond], alpha=0.3)
for cond in ORDERED_CONDITIONS:
    subset = rheos[rheos['condition'] == cond]
    n = subset['recording'].nunique()
    N = subset['animal_id'].nunique()
    axs[2].plot([], [], marker='s',
                label=f"{cond} (n={n}, N={N})",
                color=COLOR_MAP[cond])

# Customize labels and title for waveforms plot
ax_waveforms.set_title("AP Waveforms by Condition")
ax_waveforms.set_xlabel("Time (s)")
ax_waveforms.set_ylabel("Adjusted Vm")

ax10 = axs[11]  # Subplot for I/O curve

for cond in ORDERED_CONDITIONS:
    c_data = io_df[io_df['condition'] == cond]
    
    # Step 1: Average across sweeps **per cell**
    cell_avg = c_data.groupby(['recording', 'current_injection'])['spike_count'].mean().reset_index()
    
    # Step 2: Average across cells for the condition
    grouped = cell_avg.groupby('current_injection')['spike_count']
    avg, sem = grouped.mean(), grouped.sem()
    
    # Count of unique cells (n) and animals (N)
    n = cell_avg['recording'].nunique()
    N = cell_avg['recording'].map(lambda r: extract_animal_id(r)).nunique()
    
    ax10.plot(avg.index, avg,
              label=f"{cond} (n={n}, N={N})",
              color=COLOR_MAP[cond])
    ax10.fill_between(avg.index, avg - sem, avg + sem, color=COLOR_MAP[cond], alpha=0.2)

ax10.set_xlabel('Current Injection (pA)')
ax10.set_ylabel('Spike Count')
ax10.set_title('Input/Output Curve')
# --- Step 10: Peak Output Frequency per Recording ---

# Compute "peak frequency" = sweep with highest average output per recording
peak_freq_df = []

for rec, group in io_df.groupby('recording'):
    cond = group['condition'].iloc[0]
    # Average spike count per sweep
    sweep_avg = group.groupby('sweep_number')['spike_count'].mean()
    if not sweep_avg.empty:
        max_avg = sweep_avg.max()  # take sweep with highest output
        peak_freq_df.append({
            'recording': rec,
            'condition': cond,
            'peak_frequency': max_avg
        })

peak_freq_df = pd.DataFrame(peak_freq_df)
peak_freq_df['animal_id'] = peak_freq_df['recording'].apply(extract_animal_id)

# --- Plot peak frequency / output in axs[13] ---
ax_pf = axs[13]

# Compute p-value
p_pf = t_test_by_condition(peak_freq_df, 'peak_frequency')

# Boxplot
sns.boxplot(
    x='condition',
    y='peak_frequency',
    data=peak_freq_df,
    order=ORDERED_CONDITIONS,
    palette=COLOR_MAP,
    showfliers=False,
    ax=ax_pf,
    hue='condition'
)

# Swarmplot for picking
swarm_pf = sns.swarmplot(
    x='condition',
    y='peak_frequency',
    data=peak_freq_df,
    order=ORDERED_CONDITIONS,
    color='black',
    size=3,
    alpha=0.7,
    ax=ax_pf
)

# Register for pick events
for coll in swarm_pf.collections:
    coll.set_picker(True)
    artist_to_data[coll] = (peak_freq_df, 'peak_frequency')

# Axis labels and title
ax_pf.set_xlabel('')
ax_pf.set_ylabel('Frequency (Hz)')
ax_pf.set_title(
    f'Peak Frequency (p = {p_pf:.4f})'
    if p_pf >= 0.0001 else
    'Peak Frequency (p < 0.0001)'
)

# --- Helper: Find all ABF files for capacitance ---
def find_all_abf_files(filtered_df, root_dir):
    """Find all .abf files for capacitance data."""
    trimmed_ids = ['_'.join(os.path.basename(rec).split('_')[:4]) 
                   for rec in filtered_df['recording'].unique()]
    abf_files = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith('membrane_test.abf'):
                parts = fname.split('_')
                if len(parts) >= 4:
                    try:
                        abf_files.append({
                            "file_path": os.path.join(root, fname),
                            "filename": fname,
                            "date": '_'.join(parts[:3]),
                            "counter": int(parts[3]),
                            "full_id": '_'.join(parts[:4])
                        })
                    except ValueError:
                        continue

    abf_df = pd.DataFrame(abf_files)
    
    # Filter the ABF files based on trimmed_ids (to match recording IDs)
    abf_df_filtered = abf_df[abf_df['full_id'].isin(trimmed_ids)]
    
    return abf_df_filtered['file_path'].tolist()

# --- Extract capacitance per cohort ---
def parse_abf_metadata(file_paths):
    parsed_data = []
    for path in file_paths:
        abf = pyabf.ABF(path)
        lines = abf.headerText.splitlines()
        condition = os.path.basename(os.path.dirname(path))
        for line in lines:
            if "fTelegraphMembraneCap" in line:
                match = re.search(r"\[([\d.eE+-]+)", line)
                value = float(match.group(1)) if match else None
                parsed_data.append({
                    'filename': os.path.basename(path),
                    'condition': condition,
                    'value': value,
                    'protocol': abf.protocol
                })
    return parsed_data# Find all ABF files for each cohort


all_paths = find_all_abf_files(filtered_df=filtered_data, root_dir=ROOT)

# Parse the metadata from those files
metadata = parse_abf_metadata(all_paths)


capacitance_data = pd.DataFrame(metadata)

p_cap = t_test_by_condition(capacitance_data, 'value')
# --- Plot capacitance ---
# --- Plot capacitance ---
if not capacitance_data.empty:
    sns.boxplot(
        x='condition', 
        y='value', 
        data=capacitance_data,
        order=ORDERED_CONDITIONS, 
        palette=COLOR_MAP, 
        showfliers=False, 
        ax=axs[9], 
        hue='condition'
    )
    sns.stripplot(
        x='condition', 
        y='value', 
        data=capacitance_data, 
        hue='condition',
        color='black', 
        jitter=True, 
        size=3, 
        alpha=0.7, 
        ax=axs[9]
    )
    axs[9].set_title(
        f"Capacitance (p = {p_cap:.4f})"
        if p_cap >= 0.0001 else
        "Capacitance (p < 0.0001)"
    )
    axs[9].set_ylabel("Capacitance (pF)")  # <-- Change y-axis label here
    axs[9].set_xlabel('')
else:
    axs[9].text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
    axs[9].set_axis_off()


# --- Output at 300 pA as a box-and-whisker plot ---
IO_TARGET = 300
io_300 = io_df[io_df['current_injection'] == IO_TARGET]

# Average spike count per recording
io_300_rec = (
    io_300
    .groupby(['recording', 'condition'])['spike_count']
    .mean()
    .reset_index()
)

# --- Step 8b: Boxplot + swarmplot overlay for 300 pA ---
ax_io_bar = axs[12]

sns.boxplot(
    x='condition',
    y='spike_count',
    data=io_300_rec,
    order=ORDERED_CONDITIONS,
    palette=COLOR_MAP,
    showfliers=False,  # hides outlier points
    ax=ax_io_bar
)

# Add swarmplot overlay for point selection
swarm_io = sns.swarmplot(
    x='condition',
    y='spike_count',
    data=io_300_rec,
    order=ORDERED_CONDITIONS,
    color='black',   # solid black dots
    size=3,
    alpha=0.7,
    ax=ax_io_bar
)

# Register swarm points for picking
for coll in swarm_io.collections:
    coll.set_picker(True)
    artist_to_data[coll] = (io_300_rec, 'spike_count')
p_io_300 = t_test_by_condition(io_300_rec, 'spike_count')

ax_io_bar.set_ylabel('Spike Count')
ax_io_bar.set_xlabel('')
ax_io_bar.set_title(
    f'Output at 300 pA (p = {p_io_300:.4f})'
    if p_io_300 >= 0.0001 else
    'Output at 300 pA (p < 0.0001)'
)

# --- Step 9: Add a Universal Legend ---
handles, labels = [], []
for cond in ORDERED_CONDITIONS:
    subset = filtered_data[filtered_data['condition'] == cond]
    n = subset['recording'].nunique()
    N = subset['animal_id'].nunique()

    handle = plt.Line2D([0], [0], marker='o', color='w',
                        markerfacecolor=COLOR_MAP[cond], markersize=5)
    handles.append(handle)
    labels.append(f"{cond} (n={n}, N={N})")


fig.legend(handles, labels, loc='center', ncol=2, bbox_to_anchor=(0.5, 0.02))
fig.suptitle(FIG_TITLE, fontsize = 20, fontweight = 'bold', color = 'darkgreen')

# Connect pick event to the handler
fig.canvas.mpl_connect('pick_event', onpick)


# Show the figure with all subplots
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.2, bottom = 0.07)  # hspace for vertical padding, wspace for horizontal padding
plt.show()
