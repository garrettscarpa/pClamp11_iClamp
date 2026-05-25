This collection of scripts is for classifying patch clamp data in the current clamp configuration. Currently, it requires 3 recordings per cell (membrane_test, Steps_300, and Steps_400). 

membrane_test is a 1 second train of membrane test pulses (5 mV, 5 ms I think). The important thing is that when you collect a v-clamp recording, it automatically logs the capacitence, which is used for plotting. 

Steps_300 is a series of 20 repeats of  1 second current steps ranging from -100 to +300 pA (20 pA steps). 

Steps_400 is a series of 5 repeats of 1 second current steps ranging from +320 to +400 pA.

You must independently calculate your liquid junction potential and enter it, or else enter 0. 

Run '0_filename_protocol_merger' to append protocol name to the recording (I use a separate folder than where raw data is kept)

Run '1_threshold_data' to load the current steps data for each cell. You can adjust the thresholding with the cursor. Left and right arrows cycle between sweeps, and the next/previous cell buttons cycle between cells. Save your data for automatic re-loading mid-analysis.
