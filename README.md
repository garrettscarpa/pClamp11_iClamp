This collection of scripts is for classifying patch clamp data in the current clamp configuration. Currently, it requires 3 recordings per cell (membrane_test, Steps_300, and Steps_400). 

membrane_test is a 1 second train of membrane test pulses (5 mV, 5 ms I think). The important thing is that when you collect a v-clamp recording, it automatically logs the capacitence, which is used for plotting. 

Steps_300 is a series of 20 repeats of 1 second current steps ranging from -100 to +300 pA (20 pA steps). Each current step is flanked by a 0 pA step for 0.25 seconds. 

Steps_400 is a series of 5 repeats of 1 second current steps ranging from +320 to +400 pA.

You must independently calculate your liquid junction potential and enter it, or else enter 0. 

Run '0_filename_protocol_merger.py' to append protocol name to the recording (I use a separate folder than where raw data is kept)

Run '1_threshold_data.py' to load the current steps data for each cell. You can adjust the thresholding with the cursor. Left and right arrows cycle between sweeps, and the next/previous cell buttons cycle between cells. Save your data for automatic re-loading mid-analysis. Upon save, generates '1_thresholded_data.csv'

<img width="784" height="587" alt="Image" src="https://github.com/user-attachments/assets/ce2f0bb5-d468-49e7-a156-ed84455b1759" />

Run '2_all_parameters.py' to automatically extract spike features for each spike detected in '1_threshold_data'. Processing speed is ~60 spikes/sec on my old 2013 macbook, and ~250 spikes/sec on my newer silicone mac. 

Run '3_spike_parameters_viewer_limited.py' to display the first 1-3 spikes per cell at rheobase. These are the spikes that will be used for spike feature analysis. You can use the 'allspikes' code if you want to view each and every spike. The user can add spikes by toggling add spike mode to on, and then clicking in the upper image near the spike peak (you might need to zoom in). The user can delete a spike by clicking 'delete' when that spike is displayed. Halfwidth, threshold, and ahp markers can each be adjusted by dragging. Click save to remember your changes upon re-load. 

<img width="1130" height="677" alt="Image" src="https://github.com/user-attachments/assets/8c96745a-96ed-4337-b487-2eb218b74c9a" />

Run '4_filter_data.py' to remove any low-quality recordings bsed on parameters such as frequency, spike amplitude, peak voltage, etc. You can also use this to separate putative populations based on features such as half-width and input resistance. Upon running, it generates a histogram of cell peak frequency, with color coding based on mouse id. The code assumes that only 1 mouse is recorded per day. The code also generates 4_filtered_data.csv & excluded_data.csv, the latter of which contains recorded justification for each excluded cell. 

<img width="982" height="590" alt="Image" src="https://github.com/user-attachments/assets/ba763337-b548-4ca4-9b5d-e74b21a57bb6" />

Run 5_plotting.py to generate comparison plots with students t-tests run on column data. Clicking a datapoint reveals the recording it was taken from (or rather the membrane_test recording associated with that cell). 2 panels available for additional plots.

<img width="1899" height="971" alt="Image" src="https://github.com/user-attachments/assets/be955fee-aac7-4ff4-bbd9-9a37f21c1431" />

