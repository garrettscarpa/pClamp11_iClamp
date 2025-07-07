import matplotlib.pyplot as plt
import pyabf
import os

############################## Load the ABF file ##############################
root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett/PatchClamp/Data'
recording = '2025_07_02_0008_Gap_Free'

abf = pyabf.ABF(os.path.join(root, recording + ".abf"))

config = 'iclamp'

LJP_CORRECTION_MV = 14.681  # LJP correction in mV — will be subtracted from recorded voltages

############################## Prepare Sweep ##################################
print("total sweeps =", abf.sweepCount)
abf.setSweep(0, channel=0)  # Removed baseline subtraction
fs = abf.dataRate  # Sampling rate (Hz)

# Apply LJP correction if in current clamp mode
if config == 'iclamp':
    corrected_vm = abf.sweepY - LJP_CORRECTION_MV
else:
    corrected_vm = abf.sweepY

############################## Create Plot ####################################
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot membrane potential (channel 0)
axs[0].plot(abf.sweepX, corrected_vm, color='maroon')
axs[0].axhline(0, color='k', ls='--')
axs[0].set_xlabel(abf.sweepLabelX)
axs[0].set_ylabel('Vm (mV)')
axs[0].set_title("Membrane Potential" + (" (LJP Corrected)" if config == 'iclamp' else ""))
axs[0].set_ylim(-200, 200)

# Plot current (channel 1)
abf.setSweep(sweepNumber=0, channel=1)
axs[1].plot(abf.sweepX, abf.sweepY, color='olive')
axs[1].set_xlabel(abf.sweepLabelX)
axs[1].set_ylabel("Current (pA)")
axs[1].set_title("Raw Current")
axs[1].set_ylim(-100, 100)

plt.subplots_adjust(hspace=0.4)
plt.show()
