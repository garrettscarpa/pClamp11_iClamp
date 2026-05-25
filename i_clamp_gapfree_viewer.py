import matplotlib.pyplot as plt
import pyabf
import os

############################## Load the ABF file #############################
root = '/Users/garrett/Desktop/Analyses/Patch Clamp/Condition 2'
recording = '2025_05_06_0012_membrane_test'


abf = pyabf.ABF(os.path.join(root, recording + ".abf"))

############################## Globals ########################################
sweep = 0
print("total sweeps =", abf.sweepCount)

############################## Create Plot ####################################
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# initial data
abf.setSweep(sweep, channel=0)
line_vm, = axs[0].plot(abf.sweepX, abf.sweepY, color='maroon')
axs[0].axhline(0, color='k', ls='--')
axs[0].set_xlabel(abf.sweepLabelX)
axs[0].set_ylabel('Vm (mV)')
axs[0].set_title("Raw Membrane Potential")
axs[0].set_ylim(-100, 200)

abf.setSweep(sweep, channel=1)
line_i, = axs[1].plot(abf.sweepX, abf.sweepY, color='olive')
axs[1].set_xlabel(abf.sweepLabelX)
axs[1].set_ylabel("Current (pA)")
axs[1].set_title("Raw Current")
axs[1].set_ylim(-100, 100)

plt.subplots_adjust(hspace=0.4)

############################## Update Function ################################
def update_plot():
    global sweep

    abf.setSweep(sweep, channel=0)
    line_vm.set_ydata(abf.sweepY)

    abf.setSweep(sweep, channel=1)
    line_i.set_ydata(abf.sweepY)

    fig.suptitle(f"Sweep {sweep+1} / {abf.sweepCount}")
    fig.canvas.draw_idle()

############################## Key Press Handler ##############################
def on_key(event):
    global sweep

    if event.key == "right":
        sweep = min(sweep + 1, abf.sweepCount - 1)
        update_plot()

    elif event.key == "left":
        sweep = max(sweep - 1, 0)
        update_plot()

fig.canvas.mpl_connect("key_press_event", on_key)

plt.show()