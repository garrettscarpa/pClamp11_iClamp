import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pyabf
import os

############################## Load the ABF file ##############################
root = '/Volumes/BWH-HVDATA/Individual Folders/Garrett Scarpa/PatchClamp/Data'
recording = '2025_08_28_0113'

abf = pyabf.ABF(os.path.join(root, recording + ".abf"))

config = 'iclamp'
LJP_CORRECTION_MV = 0  

############################## Prepare Sweep ##################################
print("total sweeps =", abf.sweepCount)
abf.setSweep(0, channel=0)
fs = abf.dataRate  # Sampling rate (Hz)

if config == 'iclamp':
    corrected_vm = abf.sweepY - LJP_CORRECTION_MV
else:
    corrected_vm = abf.sweepY

############################## Create Plot ####################################
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Top plot: Vm
axs[0].plot(abf.sweepX, corrected_vm, color='maroon')
axs[0].axhline(0, color='k', ls='--')
axs[0].set_xlabel(abf.sweepLabelX)
axs[0].set_ylabel('Vm (mV)')
axs[0].set_title("Membrane Potential")

axs[0].set_xlim(1.040, 1.060)
axs[0].set_ylim(-4, 0.2)

# Bottom plot: Current
abf.setSweep(sweepNumber=0, channel=1)
axs[1].plot(abf.sweepX, abf.sweepY, color='olive')
axs[1].set_xlabel(abf.sweepLabelX)
axs[1].set_ylabel("Current (pA)")
axs[1].set_title("Raw Current")
axs[1].set_xlim(1.040, 1.060)
axs[1].set_ylim(-1.0, 0.2)

############################## Sliders ####################################
# Add axes for sliders (outside main plots)
axcolor = 'lightgoldenrodyellow'
ax_x1 = plt.axes([0.15, 0.03, 0.65, 0.02], facecolor=axcolor)
ax_x2 = plt.axes([0.15, 0.06, 0.65, 0.02], facecolor=axcolor)
ax_y1 = plt.axes([0.83, 0.25, 0.02, 0.5], facecolor=axcolor)
ax_y2 = plt.axes([0.87, 0.25, 0.02, 0.5], facecolor=axcolor)

# Define sliders
s_x1 = Slider(ax_x1, 'X1', 1.040, 1.060, valinit=1.045)
s_x2 = Slider(ax_x2, 'X2', 1.040, 1.060, valinit=1.055)
s_y1 = Slider(ax_y1, 'Y1', -4, 0.2, valinit=-2.0, orientation='vertical')
s_y2 = Slider(ax_y2, 'Y2', -4, 0.2, valinit=-1.0, orientation='vertical')

# Add draggable lines
vline1 = axs[0].axvline(s_x1.val, color='blue', linestyle='--')
vline2 = axs[0].axvline(s_x2.val, color='red', linestyle='--')
hline1 = axs[0].axhline(s_y1.val, color='blue', linestyle='--')
hline2 = axs[0].axhline(s_y2.val, color='red', linestyle='--')

# Text box to show differences
text = axs[0].text(0.02, 0.95, "", transform=axs[0].transAxes, 
                   va='top', ha='left', fontsize=12, bbox=dict(facecolor='w'))

def update(val):
    # Update line positions
    vline1.set_xdata([s_x1.val, s_x1.val])
    vline2.set_xdata([s_x2.val, s_x2.val])
    hline1.set_ydata([s_y1.val, s_y1.val])
    hline2.set_ydata([s_y2.val, s_y2.val])
    
    # Update text with differences
    dx = abs(s_x2.val - s_x1.val)
    dy = abs(s_y2.val - s_y1.val)
    text.set_text(f"ΔX = {dx:.6f} s\nΔY = {dy:.3f} mV")
    fig.canvas.draw_idle()

# Connect sliders to update
for slider in [s_x1, s_x2, s_y1, s_y2]:
    slider.on_changed(update)

update(None)  # Initialize

plt.subplots_adjust(hspace=0.4, bottom=0.12, right=0.8)
plt.show()
