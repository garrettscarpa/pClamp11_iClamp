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
from matplotlib.widgets import Slider

# Load the ABF file
root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett/Patch Clamp/Data'
file = '2025_06_10_0002_Steps_300'


# Set minimum inter-spike-interval
min_isi = 2

rec = os.path.join(root, file) + '.abf'
abf = pyabf.ABF(rec)

# Determine the length of the sweep in ms and the offset
length = len(abf.sweepX)  # in samples
mf = 0.015625

offset = int(length * mf)
time = abf.sweepX
time = np.concatenate([np.zeros(offset, dtype=np.float64), time])
time = time[:-offset]

# Initialize variables to store the global min and max for y-axes
global_ymin = float('inf')
global_ymax = float('-inf')
global_cmin = float('inf')
global_cmax = float('-inf')

# Iterate over all sweeps to find the global min and max values for both y-axes
for sweep_number in range(abf.sweepCount):
    abf.setSweep(sweep_number)
    global_ymin = min(global_ymin, min(abf.sweepY)) - 1
    global_ymax = max(global_ymax, max(abf.sweepY)) + 10
    global_cmin = min(global_cmin, min(abf.sweepC)) - 1
    global_cmax = max(global_cmax, max(abf.sweepC)) + 10

# Create a figure with two subplots: one for abf.sweepY and one for abf.sweepC
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=False, gridspec_kw={'height_ratios': [1, 1, 1]})
plt.subplots_adjust(bottom=0.12)  # Leave space for the slider

# Initialize a variable to keep track of the current sweep number
current_sweep = 0

# Initial threshold value
threshold = 0  # Set default threshold value
thresholds = [0] * abf.sweepCount  # Store threshold per sweep

latency_samples = int(min_isi * (abf.sampleRate / 1000))  # Convert latency to samples

# Action potential window parameters (±5 ms)
window_ms = 5
window_samples = int(window_ms * (abf.sampleRate / 1000))  # Convert window to samples

# Initialize lists to store current injections and spike counts
current_injections = []
spike_counts = []

# Iterate over all sweeps to calculate current injections and spike counts
for sweep_number in range(abf.sweepCount):
    abf.setSweep(sweep_number)
    
    # Step 1: Get the current value at time = 0 (this is the baseline)
    baseline_current = abf.sweepC[0]  # Current at time = 0
    
    # Step 2: Subtract the baseline from the entire command trace to get relative values
    relative_current_trace = abf.sweepC - baseline_current
    
    # Step 3: Calculate the max positive and negative deviation from 0
    max_deviation = np.max(relative_current_trace)  # Maximum positive deviation
    min_deviation = np.min(relative_current_trace)  # Maximum negative deviation
    
    # Step 4: Check if the command current contains any negative values
    if min_deviation < 0:  # If there's a negative value, consider the current injection as negative
        current_injected_range = -max(abs(max_deviation), abs(min_deviation))
    else:
        current_injected_range = max(abs(max_deviation), abs(min_deviation))
    
    # Store this current range (now considering both positive and negative deflections)
    current_injections.append(current_injected_range)

    # Detect threshold crossings and apply latency
    voltage_trace = abf.sweepY
    crossing_indices = np.where(voltage_trace > threshold)[0]
    
    detected_peaks = []
    last_crossing = -latency_samples  # Start with no crossings
    
    for idx in crossing_indices:
        if idx - last_crossing > latency_samples:
            # Define the window around the threshold crossing
            start_idx = max(0, idx - window_samples)  # Make sure we don't go negative
            end_idx = min(len(voltage_trace), idx + window_samples)  # Ensure we don't exceed the trace length
            
            # Find the peak in the window
            peak_idx = np.argmax(voltage_trace[start_idx:end_idx]) + start_idx  # Get index of max voltage
            
            # Store the detected peak
            detected_peaks.append((time[peak_idx], voltage_trace[peak_idx]))
            
            last_crossing = idx  # Update last crossing time

    # Count the number of detected spikes (peaks)
    spike_counts.append(len(detected_peaks))

    # For plotting, determine the global min and max for y-limits
    global_ymin = min(global_ymin, min(voltage_trace)) - 1
    global_ymax = max(global_ymax, max(voltage_trace)) + 10
    global_cmin = min(global_cmin, min(abf.sweepC)) - 1
    global_cmax = max(global_cmax, max(abf.sweepC)) + 10

# Now, generate the input/output curve

# Convert the spike counts to frequency (spikes per second)
sampling_rate = abf.sampleRate  # in Hz (samples per second)
time_per_sweep = len(abf.sweepX) / sampling_rate  # Time per sweep in seconds
spike_frequencies = [spike_count / time_per_sweep for spike_count in spike_counts]

def compute_spike_frequencies(threshold_val):
    spike_counts_local = []
    for sweep_number in range(abf.sweepCount):
        abf.setSweep(sweep_number)
        voltage_trace = abf.sweepY

        crossing_indices = np.where(voltage_trace > threshold_val)[0]
        last_crossing = -latency_samples
        detected_peaks = []

        for idx in crossing_indices:
            if idx - last_crossing > latency_samples:
                start_idx = max(0, idx - window_samples)
                end_idx = min(len(voltage_trace), idx + window_samples)
                peak_idx = np.argmax(voltage_trace[start_idx:end_idx]) + start_idx
                detected_peaks.append((time[peak_idx], voltage_trace[peak_idx]))
                last_crossing = idx

        spike_counts_local.append(len(detected_peaks))

    # Convert counts to frequencies (Hz)
    return [count / time_per_sweep for count in spike_counts_local]



# Function to update the plot based on the current sweep
def update_plot(sweep_number, spike_freqs):
    # Initialize last_crossing for the current sweep
    global threshold
    last_crossing = -latency_samples  # Start with no crossings
    
    abf.setSweep(sweep_number)
    threshold = thresholds[sweep_number]  # Load threshold for current sweep
    if abs(threshold_slider.val - threshold) > 1e-3:
            threshold_slider.eventson = False  # Temporarily disable event callbacks
            threshold_slider.set_val(threshold)
            threshold_slider.eventson = True   # Re-enable events

    # Clear plots
    for ax in axes:
        ax.cla()


    # Plot the new sweep data
    axes[0].plot(time, abf.sweepY, color='purple')
    axes[0].set_title(f'Signal - Sweep {sweep_number}')
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].set_ylim(-100, 100)  # Apply global y-limits
    
    # Plot the threshold line
    axes[0].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold} mV')
    
    axes[1].plot(time, abf.sweepC, color='k')  # Convert command to pA
    axes[1].set_title(f'Command')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Amplitude (pA)')
    axes[1].set_ylim(global_cmin, global_cmax)  # Apply global y-limits
    
    # Plot the input/output curve on the third axis
    axes[2].plot(current_injections, spike_freqs, marker='o', linestyle='-', color='b')
    axes[2].set_xlabel('Current Injection (pA)')
    axes[2].set_ylabel('Spike Frequency (Hz)')
    axes[2].set_title('Input/Output Curve')
    axes[2].grid(True)


    # Detect threshold crossings and apply latency
    voltage_trace = abf.sweepY
    crossing_indices = np.where(voltage_trace > threshold)[0]
    
    detected_peaks = []
    for idx in crossing_indices:
        if idx - last_crossing > latency_samples:
            # Define the window around the threshold crossing
            start_idx = max(0, idx - window_samples)  # Make sure we don't go negative
            end_idx = min(len(voltage_trace), idx + window_samples)  # Ensure we don't exceed the trace length
            
            # Find the peak in the window
            peak_idx = np.argmax(voltage_trace[start_idx:end_idx]) + start_idx  # Get index of max voltage
            peak_voltage = voltage_trace[peak_idx]
            peak_time = time[peak_idx]
            
            # Store the detected peak
            detected_peaks.append((peak_time, peak_voltage))
            
            last_crossing = idx  # Update last crossing time
    
    if len(detected_peaks) > 0:
        # Mark the detected peaks on the voltage trace
        for peak_time, peak_voltage in detected_peaks:
            axes[0].scatter(peak_time, peak_voltage, color='orange', label='Action Potential Peak')

    # Redraw the figure
    plt.draw()

# Define key press event handler to cycle through sweeps
def on_key(event):
    global current_sweep, threshold
    if event.key == 'right' and current_sweep < abf.sweepCount - 1:
        current_sweep += 1
    elif event.key == 'left' and current_sweep > 0:
        current_sweep -= 1
    else:
        return

    threshold = thresholds[current_sweep]  # Load saved threshold
    spike_freqs = compute_spike_frequencies(threshold)
    update_plot(current_sweep, spike_freqs)

def update_threshold(val):
    global threshold
    threshold = val
    thresholds[current_sweep] = val  # Save threshold for this sweep

    # Recalculate spike frequency for current sweep only
    abf.setSweep(current_sweep)
    voltage_trace = abf.sweepY
    crossing_indices = np.where(voltage_trace > threshold)[0]
    last_crossing = -latency_samples
    detected_peaks = []

    for idx in crossing_indices:
        if idx - last_crossing > latency_samples:
            start_idx = max(0, idx - window_samples)
            end_idx = min(len(voltage_trace), idx + window_samples)
            peak_idx = np.argmax(voltage_trace[start_idx:end_idx]) + start_idx
            detected_peaks.append((time[peak_idx], voltage_trace[peak_idx]))
            last_crossing = idx

    spike_frequencies[current_sweep] = len(detected_peaks) / time_per_sweep

    # Update only the current plot
    update_plot(current_sweep, spike_frequencies)

# Create a slider for adjusting the threshold
ax_threshold = plt.axes([0.15, 0.01, 0.7, 0.03])  # Position of slider
threshold_slider = Slider(ax_threshold, 'Threshold (mV)', -100, 100, valinit=threshold, valstep=1)
threshold_slider.on_changed(update_threshold)
# Set up the initial plot for sweep 0
spike_frequencies = compute_spike_frequencies(threshold)
update_plot(0, spike_frequencies)

# Connect the key press event to the handler
fig.canvas.mpl_connect('key_press_event', on_key)



# Adjust the layout to make it nicer
plt.tight_layout()
# Show the plot
plt.show()