import matplotlib.pyplot as plt
import pyabf
import os

############################### Set the ABF file ##############################
root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/PatchClamp/Data/'
recording = '2026_01_31_0010'

############################## Spritz Parameters ##############################
spritz_start = 11  # seconds
spritz_end = 21    # seconds
spritz_label = "DCZ (30 uM)"

############################## Load Trace #####################################
abf = pyabf.ABF(os.path.join(root, recording + ".abf"))

abf.setSweep(0, channel=0)  # Removed baseline subtraction
fs = abf.dataRate  # Sampling rate (Hz)

############################## Create Plot ####################################
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot raw current (channel 0)
axs[0].plot(abf.sweepX, abf.sweepY, color='maroon')
axs[0].axhline(0, color='k', ls='--')
axs[0].set_xlabel(abf.sweepLabelX)
axs[0].set_ylabel('Vm (mV)')
axs[0].set_title("Raw Membrane Potential")
axs[0].set_ylim(-100, 50)

# --- Spritz bar annotation (inside plot at y=100) ---
axs[0].hlines(
    y=25,
    xmin=spritz_start,
    xmax=spritz_end,
    colors='black',
    linewidth=6
)

axs[0].text(
    (spritz_start + spritz_end) / 2,
    30,  # slightly above the bar
    spritz_label,
    ha='center',
    va='bottom',
    fontsize=11,
    color='black'
)