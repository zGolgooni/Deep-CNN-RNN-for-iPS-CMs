# Deep-CNN-RNN-for-iPS-CMs
Implementation of the algrithm in the paper "Deep Learning Based Proarrhythmia Analysis Using Field Potentials Recorded from Human Pluripotent Stem Cells Derived Cardiomyocytes"

# Pre_requisites
Python 3.5.2
Keras 2.0.6
tensorflow 1.2.1
Numpy 1.11.1

# Format of input sample file
To test a new sample, export your recordings in .txt format.

# Trying for new samples
The trained networks are saved and currently avilable at the folder of "Trained_networks". To test a new sample, use the file "test_new_sample.py" in the "CNN_RNN" folder. You just need to give the name and directory which your .txt file is saved and also its sampling rate. In addition you can determine that you want to see the plot of your data or not.

Furthermore, example file of samples is provided in the folder of "Data". You can use this file for running the final classification algorithm too.
