import numpy as np
from scipy.signal import medfilt, find_peaks
from matplotlib import pyplot as plt
from scipy.io.wavfile import read
import glob
import os
import math
import scipy as sp



def ToolReadAudio(cAudioFilePath):
    [samplerate, x] = read(cAudioFilePath)

    if x.dtype == 'float32':
        audio = x
    else:
        # change range to [-1,1)
        if x.dtype == 'uint8':
            nbits = 8
        elif x.dtype == 'int16':
            nbits = 16
        elif x.dtype == 'int32':
            nbits = 32

        audio = x / float(2**(nbits - 1))

    if x.dtype == 'uint8':
        audio = audio - 1.

    return (samplerate, audio)





def compute_hann(iWindowLength):
    return 0.5 - (0.5 * np.cos(2 * np.pi / iWindowLength * np.arange(iWindowLength)))

def block_audio(x,blockSize,hopSize,fs):
    # allocate memory
    numBlocks = math.ceil(x.size / hopSize)
    xb = np.zeros([numBlocks, blockSize])
    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs

    x = np.concatenate((x, np.zeros(blockSize)),axis=0)
    for n in range(0, numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return (xb,t)


def compute_spectrogram(xb,fs):    
    numBlocks = xb.shape[0]    
    afWindow = compute_hann(xb.shape[1])    
    X = np.zeros([math.ceil(xb.shape[1]/2+1), numBlocks])    
    for n in range(0, numBlocks):        # apply window        
        tmp = abs(sp.fft(xb[n,:] * afWindow))*2/xb.shape[1]        # compute magnitude spectrum        
        X[:,n] = tmp[range(math.ceil(tmp.size/2+1))]         
        X[[0,math.ceil(tmp.size/2)],n]= X[[0,math.ceil(tmp.size/2)],n]/np.sqrt(2) #let's be pedantic about normalization    
    f = np.arange(0, X.shape[0])*fs/(xb.shape[1])    
    return (X,f)



########################### A. Tuning Frequency Estimation ###########################
def get_spectral_peaks(X):
    blockSize,numBlocks=X.shape
    peaks=np.zeros((20,numBlocks))
    for i in range(numBlocks):
        magnitude=X[:,i]
        peak=find_peaks(magnitude)
        peak_value=np.array([magnitude[i]for i in peak[0]])
        peak_index = np.argsort(peak_value)[-20:]
        peaks[:,i]=peak_index
    return peaks
        
def get_equal_temper_frequency(n1,n2):
    f0=440
    a=pow(2,1/12)
    equal_temper_frequency=[]
    for i in range(n1,n2):
        freq = f0*pow(a,i)
        equal_temper_frequency.append(freq)
    return np.array(equal_temper_frequency)

def estimate_tuning_freq(x, blockSize, hopSize, fs):
    xb,t = block_audio(x,blockSize,hopSize,fs)
    eq_tmp_f = get_equal_temper_frequency(-77,68)
    X,fInHz = compute_spectrogram(xb, fs)
    blockSize,numBlocks=X.shape
    peaks= get_spectral_peaks(X)
    tfInHz = fInHz
    dev=[]
    for i in range(numBlocks):
        peak_index= peaks[:,i]
        for j in range(20):
            peak_f = fInHz[int(peak_index[j])]
            if peak_f >= min(eq_tmp_f) and peak_f <=max(eq_tmp_f):    
                index = np.argmin([abs(i-peak_f)for i in eq_tmp_f])
                dev1 = 1200 * math.log2(peak_f/eq_tmp_f[index]);
                dev.append(dev1)
    hist,bin_edges = np.histogram(dev, bins=100)
    error_cent = bin_edges[np.argmax(hist)]
    tfInHz = pow(2,error_cent/1200)*440
    
    
    return tfInHz



########################### B.  Key Detection  ###########################

def extract_pitch_chroma(X, fs, tfInHz):
    blockSize,numBlocks = X.shape
    pitch_chroma = np.zeros((12,numBlocks))
    eq_tmp_f = get_equal_temper_frequency(-21,15)
    adjust_freq = np.zeros((36,1))
    error_cent = math.log2(tfInHz/440)*1200
    adjust_freq = [pow(2,error_cent/1200)*i for i in eq_tmp_f]
    f = np.arange(0, X.shape[0])*fs/(X.shape[0])
    for i in range(numBlocks):
        for j in range(blockSize):
            bin_freq = f[j]
            if bin_freq >= adjust_freq[0] and bin_freq <= adjust_freq[35]:
                index = np.argmin([abs(i-bin_freq)for i in adjust_freq])
                class_num = np.mod(index, 12);
                pitch_chroma[class_num,i] +=abs(X[j, i]) 
    return pitch_chroma
    

def norm(array):
    max1 = max(array)
    min1 = min(array)
    norm_array = [(array[i]-min1)/(max1-min1) for i in range(len(array))]
    return norm_array





def detect_key(x, blockSize, hopSize, fs, bTune):
    t_pc = np.array([[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
[6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]])
    keyEstimate = np.zeros((24,1))
    t_pc_major = t_pc[0]
    t_pc_major_norm = norm(t_pc_major)
    t_pc_minor = t_pc[1]
    t_pc_minor_norm = norm(t_pc_minor)
    xb,t = block_audio(x,blockSize,hopSize,fs)
    X,fInHz = compute_spectrogram(xb, fs)
    blockSize,numBlocks = X.shape
    tfInHz = estimate_tuning_freq(x, blockSize, hopSize, fs)
    if bTune:
        pitch_chroma = extract_pitch_chroma(X, fs, tfInHz)
    else:
        pitch_chroma = np.zeros((12,numBlocks))
        eq_tmp_f = get_equal_temper_frequency(-21,15)
        f = np.arange(0, X.shape[0])*fs/(xb.shape[1])
        for i in range(numBlocks):
            for j in range(blockSize):
                bin_freq = f[j]
                if bin_freq >= eq_tmp_f[0] and bin_freq <= eq_tmp_f[35]:
                    index = np.argmin([abs(i-bin_freq)for i in eq_tmp_f])
                    class_num = np.mod(index, 12)
                    pitch_chroma[class_num,i] +=abs(X[j, i]) 
    
        
    pitch_chroma_total = np.sum(pitch_chroma, axis=1)
    distance = np.zeros((24,1))
    pitch_chroma_total = np.roll(pitch_chroma_total,3)
    picth_chroma_norm = norm(pitch_chroma_total)
    for i in range(12):
        
        current_pc_major_norm = np.roll(t_pc_major_norm,i)
        current_pc_minor_norm = np.roll(t_pc_minor_norm,i)
        distance[i] = np.sqrt(sum((picth_chroma_norm[i]-current_pc_major_norm[i])**2 for i in range(12)))
        distance[i+12] = np.sqrt(sum((picth_chroma_norm[i]-current_pc_minor_norm[i])**2 for i in range(12)))
        
    keyEstimate = np.argmin(distance)
       
    return keyEstimate
      
       
def read_label(path):
    
    f = open(path, "r")
    for x in f:
        oup = x
    return oup


def hz2cents(freq_hz, base_frequency=10.0):
        freq_cent = np.zeros(freq_hz.shape[0])
        freq_nonz_ind = np.flatnonzero(freq_hz)
        normalized_frequency = np.abs(freq_hz[freq_nonz_ind]) / base_frequency
        freq_cent[freq_nonz_ind] = 1200 * np.log2(normalized_frequency)
        return freq_cent


def eval_tfe(pathToAudio, pathToGT):
    
    files = ['cycling_road', 'pallet_town', 'pirates', 'poke_center', 'vader']
    
    blockSize=4096
    hopSize=2048

    diff = 0

    for file in files:
        [fs, x] = ToolReadAudio(pathToAudio + '/' + file + '.wav')

        gtHz1 = read_label(pathToGT + '/' + file + '.txt')
        gtHz = np.float(gtHz1)
        print('gtHz:')
        print (gtHz)
        tuningEstimate = estimate_tuning_freq(x, blockSize, hopSize, fs)
        print('tuning Estimate:')
        print(tuningEstimate)
        diff = diff + abs(gtHz-tuningEstimate)
        
        
    freq_cent = 1200 * np.log2(diff)
    avgDeviation = freq_cent/5;

    return avgDeviation


def eval_key_detection(pathToAudio, pathToGT):

    files = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    
    accuracy = []
    blockSize = 4096
    hopSize = 2048

    i = 0
    j = 0

    for file in files:

        [fs, x] = ToolReadAudio(pathToAudio + '/' + file + '.wav')

        key_GT = int(read_label(pathToGT + '/' + file + '.txt'))
        print('Correct_key:')
        print (key_GT)

        keyEstimate_True = detect_key(x, blockSize, hopSize, fs, True)
        print('Estimatekey_T:')
        print(keyEstimate_True)
        keyEstimate_False = detect_key(x, blockSize, hopSize, fs, False)
        print('Estimatekey_F:')
        print(keyEstimate_False)

        if keyEstimate_True == key_GT:
            i = i + 1

        if keyEstimate_False == key_GT:
            j = j + 1

    accuracy_T = i/10
    accuracy_F = j/10
    accuracy.append(accuracy_T)
    accuracy.append(accuracy_F)
    

    return np.array(accuracy)

def evaluate(pathToAudioKey, pathToGTKey,pathToAudioTf, pathToGTTf):

    Tuning_freq = eval_tfe(pathToAudioTf, pathToGTTf)
    Accuracy = eval_key_detection(pathToAudioKey, pathToGTKey)

    return Tuning_freq, Accuracy
        

if __name__ == "__main__":    
    
    [T_Freq, Acc] = evaluate('/Users/yuyifei/Desktop/6201/key_tf-2/key_eval/audio', '/Users/yuyifei/Desktop/6201/key_tf-2/key_eval/GT','/Users/yuyifei/Desktop/6201/key_tf-2/tuning_eval/audio', '/Users/yuyifei/Desktop/6201/key_tf-2/tuning_eval/GT')
    
    print(T_Freq)
    print(Acc)        












'''if __name__ == "__main__":    
    wav_file='/Users/yuyifei/Downloads/key_tf/tuning_eval/audio/cycling_road.wav'
    fs, x = read(wav_file)
    wav_file2 = '/Users/yuyifei/Desktop/6201/key_tf-2/key_eval/audio/1.wav'
    fs, x2 = read(wav_file2)
    blockSize=4096
    hopSize=2048
    xb,t = block_audio(x2,blockSize,hopSize,fs)
    X,fInHz = compute_spectrogram(xb, fs)
    peaks= get_spectral_peaks(X)
    tfInHz =  estimate_tuning_freq(x2, blockSize, hopSize, fs)
    print(tfInHz)
    pitch_chroma = extract_pitch_chroma(X, fs, tfInHz)
    print(pitch_chroma[:,0])
    
    filepath2 = '/Users/yuyifei/Desktop/6201/key_tf-2/key_eval/audio'
    files = os.listdir(filepath2)
    filepath_txt = '/Users/yuyifei/Desktop/6201/key_tf-2/key_eval/GT'
    print(files)
    blockSize=4096
    hopSize=2048
    num1=0
    num2=0
    for file in files:
        index =os.path.split(file)[1].split('.')[0]
        name = os.path.join(filepath2,file)
        txt_file = os.path.join(filepath_txt,index+'.txt')
        num = np.loadtxt(txt_file,dtype=int)
        fs, x = read(name)
        keyEstimate1 = detect_key(x, blockSize, hopSize, fs, 1)
        keyEstimate2 = detect_key(x, blockSize, hopSize, fs, 0)
        if keyEstimate1 == num:
            num1 +=1
        if keyEstimate2 ==num:
            num2 +=1
    print(num1)
    print(num2)'''


    
    
    
    
    
    
    
    
    
    


