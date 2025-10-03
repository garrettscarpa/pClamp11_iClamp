import matplotlib.pyplot as plt
import pyabf
import os

############################## Load the ABF file ##############################
root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/PatchClamp/Data'
recording = '2025_09_17_0033'

abf = pyabf.ABF(os.path.join(root, recording + ".abf"))

############################## Prepare Sweep ##################################
print("total sweeps =", abf.sweepCount)
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
axs[0].set_ylim(-1000, 500)

# Plot raw membrane potential (channel 1)
abf.setSweep(sweepNumber=0, channel=1)
axs[1].plot(abf.sweepX, abf.sweepY, color='olive')
axs[1].set_xlabel(abf.sweepLabelX)
axs[1].set_ylabel("Current (pA)")
axs[1].set_title("Raw Current")
axs[1].set_ylim(-100, 100)

plt.subplots_adjust(hspace=0.4)
plt.show()
