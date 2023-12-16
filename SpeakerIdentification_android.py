import io
import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def calculate_delta(array):
   
    rows,cols = array.shape

    deltas = np.zeros((rows,20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              first =0
            else:
              first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j 
            index.append((second,first))
            j+=1
        deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return deltas


def extract_features(audio,rate):
       
    mfcc_feature = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    # plot_mfcc_features(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature,delta)) 
    return combined


def plot_mfcc_features(mfcc_features):
    # Transpose the MFCC matrix for proper visualization
    mfcc_features = mfcc_features.T

    # Create a time axis for the MFCC features
    time_axis = np.arange(0, mfcc_features.shape[1])

    # Plot the MFCC features
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc_features, cmap='viridis', aspect='auto', origin='lower', interpolation='none')
    plt.title('MFCC Features')
    plt.xlabel('Time Frame')
    plt.ylabel('MFCC Coefficient')
    plt.colorbar()
    plt.show()

def train_model(audio_data):

    dest = "/home/dimatkchnk/sem7/biometria/speaker-identification/trained_models/"
    features = np.asarray(())

    sr, audio = read(io.BytesIO(audio_data))
    print("Sample Rate: ", str(sr))
    print("Data: ", str(audio))

    vector = extract_features(audio, sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

    gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=3)
    gmm.fit(features)

    # dumping the trained gaussian model
    # picklefile = path.split("/")[-1] + ".gmm"
    pickle.dump(gmm, open(dest + 'test.wav', 'wb'))
    print('+ modeling completed for speaker: ', " with data point = ", features.shape)
    features = np.asarray(())

def test_model():

    source   = "/home/dimatkchnk/sem7/biometria/speaker-identification/testing_set"
    modelpath = "/home/dimatkchnk/sem7/biometria/speaker-identification/trained_models"
    test_file = "/home/dimatkchnk/sem7/biometria/speaker-identification/testing_set_addition.txt"
    file_paths = open(test_file,'r')

    gmm_files = [os.path.join(modelpath,fname) for fname in
                  os.listdir(modelpath) if fname.endswith('.gmm')]

    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("/")[-1].split(".gmm")[0] for fname
                  in gmm_files]
    for path in file_paths:

        path = path.strip()
        print(path)
        sr,audio = read(source + path)
        vector   = extract_features(audio,sr)

        log_likelihood = np.zeros(len(models))

        for i in range(len(models)):
            gmm = models[i]
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()

        print(log_likelihood)
        winner = np.argmax(log_likelihood)
        print(winner)
        print("\tdetected as - ", speakers[winner])
        time.sleep(1.0)

