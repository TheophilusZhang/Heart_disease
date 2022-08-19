#Inspecting the classification data
#
import librosa as lr
from glob import glob

# List all the wav files in the folder,
# Use glob to return a list of the .wav files in data_dir directory.
audio_files = glob(data_dir + '/*.wav')

# Read in the first audio file, 
#Import the first audio file in the list using librosa
audio, sfreq = lr.load(audio_files[0])
#create the time array by
#dividing the index of each datapoint by the sampling frequency.
time = np.arange(0, len(audio)) / sfreq

# Plot the waveform for this file, along with the time array.
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()

#smooth the data and then rectify it so that the total amount of
#  sound energy over time is more distinguishable
# A heartbeat file is available in the variable audio.

# Plot the raw audio data for calculation of the envelope.
audio.plot(figsize=(10, 5))
plt.show()

# Rectify the audio signal
audio_rectified = audio.apply(np.abs)

# Plot the result
audio_rectified.plot(figsize=(10, 5))
plt.show()

# Smooth by applying a rolling mean
audio_rectified_smooth = audio_rectified.rolling(50).mean()

# Plot the result
audio_rectified_smooth.plot(figsize=(10, 5))
plt.show()

# Calculate stats
means = np.mean(audio_rectified_smooth, axis=0)
stds = np.std(audio_rectified_smooth, axis=0)
maxs = np.max(audio_rectified_smooth, axis=0)

# Create the X and y arrays, Column stack these stats in the same order
X = np.column_stack([means, stds, maxs])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data,Use cross-validation to fit a model on each CV iteration.
from sklearn.model_selection import cross_val_score
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))

#the envelope calculation is a common technique in computing tempo and rhythm features. 
# use librosa to compute some tempo and rhythm features for heartbeat data, and fit a model once more.
# librosa functions tend to only operate on numpy arrays instead of DataFrames, 
# so access Pandas data as a Numpy array with the .values attribute.

# Create the X and y arrays,Column stack these tempo features (mean, standard deviation, and maximum) in the same order.
X = np.column_stack([means, stds, maxs, tempos_mean, tempos_std, tempos_max])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data,Score the classifier with cross-validation.
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))

#Spectrograms of heartbeat audio
#Spectral engineering is one of the most common techniques in machine learning for time series data. 
# The first step in this process is to calculate a spectrogram of sound. 
# This describes what spectral content (e.g., low and high pitches) are present in the sound over time. 

# Import the stft function
from librosa.core import stft

# Prepare the STFT,Calculate the spectral content 
HOP_LENGTH = 2**4
spec = stft(audio, hop_length=HOP_LENGTH, n_fft=2**7)

from librosa.core import amplitude_to_db
from librosa.display import specshow

# Convert the spectogram (spec) to decibels.
spec_db = amplitude_to_db(spec)

# Compare the raw audio to the spectrogram of the audio by visualization
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axs[0].plot(time, audio)
lr.display.specshow(spec_db, sr=sfreq, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH)
plt.show()

#Engineering spectral features

import librosa as lr

# Calculate the spectral centroid and bandwidth for the spectrogram
#use the spectral_bandwidth() and spectral_centroid() functions
#The spectrogram is stored in the variable spec.
bandwidths = lr.feature.spectral_bandwidth(S=spec)[0]
centroids = lr.feature.spectral_centroid(S=spec)[0]

from librosa.core import amplitude_to_db
from librosa.display import specshow

# Convert spectrogram to decibels for visualization
spec_db = amplitude_to_db(spec)

# Display these features on top of the spectrogram,plot the spectrogram over time.
fig, ax = plt.subplots(figsize=(10, 5))
ax = specshow(spec_db, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH)
ax.plot(times_spec, centroids)
ax.fill_between(times_spec, centroids - bandwidths / 2, centroids + bandwidths / 2, alpha=.5)
ax.set(ylim=[None, 6000])
plt.show()

# Create X and y arrays,Column stack all the features to create the array X
X = np.column_stack([means, stds, maxs, tempo_mean, tempo_max, tempo_std, bandwidths, centroids])
y = labels.reshape([-1, 1])

# Fit the model and score the classifier with cross-validation
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))

#focused on creating new "features" from raw data and not obtaining the best accuracy. 
# To improve the accuracy, you want to find the right features
#  that provide relevant information and also build models on much larger data.