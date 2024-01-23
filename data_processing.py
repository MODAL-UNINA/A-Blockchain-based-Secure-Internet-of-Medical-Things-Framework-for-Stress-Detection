import numpy as np
import pickle
import neurokit2 as nk
from more_itertools import chunked
import io
import os


def get_chest_signal(data):
    chest_ACC = data['signal']['chest']['ACC']        # ACC includes 3 columns     #700HZ
    chest_ECG = data['signal']['chest']['ECG']                                     #700HZ
    chest_EMG = data['signal']['chest']['EMG']                                     #700HZ
    chest_EDA = data['signal']['chest']['EDA']                                     #700HZ
    #chest_Temp = data['signal']['chest']['Temp']                                   #700HZ
    chest_Resp = data['signal']['chest']['Resp']                                   #700HZ
    label = data['label']                                                           #700HZ
    label = label.reshape(len(label),1)
    
    chest_Data = np.hstack((chest_ACC, chest_ECG, chest_EMG, chest_EDA, chest_Resp, label))

    raw_chest_baseline = chest_Data[chest_Data[:,7]==1., :]      #The label of baseline is 1
    raw_chest_stress = chest_Data[chest_Data[:,7]==2., :]        #The label of stress is 2
    raw_chest_amusement = chest_Data[chest_Data[:,7]==3., :]     #The label of amusement is 3
    
    return raw_chest_baseline[:,0:7], raw_chest_stress[:,0:7], raw_chest_amusement[:,0:7]


def chest_noise_filtering(chest_baseline,chest_stress,chest_amusement):

    # Noise filtering for baseline: ECG, EMG, EDA, Resp   
    clean_ecg_baseline = nk.ecg_clean(chest_baseline[:,3], sampling_rate=700)       #ecg_clean():default 5th order 0.5 Hz highpass Butterworth filter
    # peaks for ecg
    _, info_baseline = nk.ecg_peaks(clean_ecg_baseline, sampling_rate=700, method='neurokit')
    # rate for ecg
    chest_baseline[:,3] = nk.ecg_rate(peaks=info_baseline["ECG_R_Peaks"], desired_length=len(clean_ecg_baseline))
    
    clean_emg_baseline = nk.emg_clean(chest_baseline[:,4], sampling_rate=700)       #emg_clean: default 4 order 100 Hz highpass Butterworth filter
    #amplitude for emg
    chest_baseline[:,4] = nk.emg_amplitude (clean_emg_baseline)
    
    chest_baseline[:,5] = nk.eda_clean(chest_baseline[:,5], sampling_rate=700)       #eda_clean:default 4th order 3 Hz lowpass Butterworth filter
    chest_baseline[:,6] = nk.rsp_clean(chest_baseline[:,6], sampling_rate=700, method='khodadad2018')       #rep_clean:5th order 2Hz low-pass IIR Butterworth filter
    
    
    # Noise filtering for stress: ECG, EMG, EDA, Resp    
    clean_ecg_stress = nk.ecg_clean(chest_stress[:,3], sampling_rate=700)       #ecg_clean():default 5th order 0.5 Hz highpass Butterworth filter
    # peaks for ecg
    _, info_stress = nk.ecg_peaks(clean_ecg_stress, sampling_rate=700, method='neurokit')
    # rate for ecg
    chest_stress[:,3] = nk.ecg_rate(peaks=info_stress["ECG_R_Peaks"], desired_length=len(clean_ecg_stress))
    
    clean_emg_stress = nk.emg_clean(chest_stress[:,4], sampling_rate=700)       #emg_clean: default 4 order 100 Hz highpass Butterworth filter
    # amplitude for emg
    chest_stress[:,4] = nk.emg_amplitude (clean_emg_stress)

    chest_stress[:,5] = nk.eda_clean(chest_stress[:,5], sampling_rate=700)       #eda_clean:default 4th order 3 Hz lowpass Butterworth filter
    chest_stress[:,6] = nk.rsp_clean(chest_stress[:,6], sampling_rate=700, method='khodadad2018')       #rep_clean:5th order 2Hz low-pass IIR Butterworth filter


    # Noise filtering for amusement: ECG, EMG, EDA, Resp    
    clean_ecg_amusement = nk.ecg_clean(chest_amusement[:,3], sampling_rate=700)       #ecg_clean():default 5th order 0.5 Hz highpass Butterworth filter
    # peaks for ecg
    _, info_amusement = nk.ecg_peaks(clean_ecg_amusement, sampling_rate=700, method='neurokit')
    # rate for ecg
    chest_amusement[:,3] = nk.ecg_rate(peaks=info_amusement["ECG_R_Peaks"], desired_length=len(clean_ecg_amusement))
    
    #amplitude for emg
    clean_emg_amusement = nk.emg_clean(chest_amusement[:,4], sampling_rate=700)       #emg_clean: default 4 order 100 Hz highpass Butterworth filter   
    chest_amusement[:,4] = nk.emg_amplitude (clean_emg_amusement)    
    
    chest_amusement[:,5] = nk.eda_clean(chest_amusement[:,5], sampling_rate=700)       #eda_clean:default 4th order 3 Hz lowpass Butterworth filter
    chest_amusement[:,6] = nk.rsp_clean(chest_amusement[:,6], sampling_rate=700, method='khodadad2018')       #rep_clean:5th order 2Hz low-pass IIR Butterworth filter
    
    return chest_baseline, chest_stress, chest_amusement


# collect all data
D={}
file_path = os.listdir('..data/dataset/WESAD')

for file in file_path:
    f = io.open('..data/dataset/WESAD/' + file, 'rb') 
    data=pickle.load(f, encoding='latin1')
    
    # For chest data
    raw_chest_baseline, raw_chest_stress, raw_chest_amusement = get_chest_signal(data)    
    clean_chest_baseline, clean_chest_stress, clean_chest_amusement = chest_noise_filtering(raw_chest_baseline, raw_chest_stress, raw_chest_amusement)
    
    # Downsampling for christ data
    downsampled_baseline = np.array([sum(x) / len(x) for x in chunked(clean_chest_baseline, 175)])
    downsampled_stress = np.array([sum(x) / len(x) for x in chunked(clean_chest_stress, 175)])
    downsampled_amusement = np.array([sum(x) / len(x) for x in chunked(clean_chest_amusement, 175)])

    chest_clean_baseline = downsampled_baseline
    chest_clean_stress = downsampled_stress
    chest_clean_amusement = downsampled_amusement
    
    # For wrist data
    wrist_ACC = data['signal']['wrist']['ACC']        # ACC includes 3 columns       # 32 Hz
    wrist_BVP = data['signal']['wrist']['BVP']                                        # 64 Hz
    wrist_EDA = data['signal']['wrist']['EDA']                                        # 4 Hz
    wrist_TEMP = data['signal']['wrist']['TEMP']                                      # 4 Hz
    label = data['label']                                                             #700HZ
    label = label.reshape(len(label),1)
    
    # noise filtering for BVP
    clean_BVP = nk.ppg_clean(wrist_BVP, sampling_rate=64)     #ppg_clean:default 3th order 0.5 - 8 Hz bandpass Butterworth filter
    # Find peaks
    bvp_info = nk.ppg_findpeaks(clean_BVP, sampling_rate=64)
    # get rate
    bvp_rate = nk.signal_rate(bvp_info["PPG_Peaks"], sampling_rate=64, desired_length=len(clean_BVP))
    bvp_rate = bvp_rate.reshape(len(bvp_rate),1)

    # Downsampling for wrist data    
    wrist_downsampled_ACC = np.array([sum(x) / len(x) for x in chunked(wrist_ACC, 8)])
    wrist_downsampled_BVP = np.array([sum(x) / len(x) for x in chunked(bvp_rate, 16)])
    wrist_downsampled_EDA = np.array([sum(x) / len(x) for x in chunked(wrist_EDA, 1)])
    wrist_downsampled_label = np.array([sum(x) / len(x) for x in chunked(label, 175)])
    
    wrist_DATA =  np.hstack((wrist_downsampled_ACC, wrist_downsampled_BVP, wrist_downsampled_EDA, wrist_downsampled_label))
    
    #get 3 emotion
    wrist_baseline = wrist_DATA[wrist_DATA[:,5]==1., :]      #The label of baseline is 1
    wrist_stress = wrist_DATA[wrist_DATA[:,5]==2., :]        #The label of stress is 2
    wrist_amusement = wrist_DATA[wrist_DATA[:,5]==3., :]     #The label of amusement is 3

    wrist_clean_baseline = wrist_baseline[:,0:5]
    wrist_clean_stress = wrist_stress[:,0:5]
    wrist_clean_amusement = wrist_amusement[:,0:5]

    # combine chest data and wrist data
    name_baseline = file.split(".")[0] + '_chest+wrist_baseline'
    name_stress = file.split(".")[0] + '_chest+wrist_stress'
    name_amusement = file.split(".")[0] + '_chest+wrist_amusement'
    
    baseline_shape0 = min(chest_clean_baseline.shape[0], wrist_clean_baseline.shape[0])
    stress_shape0 = min(chest_clean_stress.shape[0], wrist_clean_stress.shape[0])
    amusement_shape0 = min(chest_clean_amusement.shape[0], wrist_clean_amusement.shape[0])

    D[name_baseline] = np.hstack((chest_clean_baseline[0:baseline_shape0,:], wrist_clean_baseline[0:baseline_shape0,:]))
    D[name_stress] = np.hstack((chest_clean_stress[0:stress_shape0,:], wrist_clean_stress[0:stress_shape0,:]))
    D[name_amusement] = np.hstack((chest_clean_amusement[0:amusement_shape0,:], wrist_clean_amusement[0:amusement_shape0,:]))
    


s2_baseline, s2_amusement, s2_stress = D['S2_chest+wrist_baseline'], D['S2_chest+wrist_amusement'], D['S2_chest+wrist_stress']
s3_baseline, s3_amusement, s3_stress = D['S3_chest+wrist_baseline'], D['S3_chest+wrist_amusement'], D['S3_chest+wrist_stress']
s4_baseline, s4_amusement, s4_stress = D['S4_chest+wrist_baseline'], D['S4_chest+wrist_amusement'], D['S4_chest+wrist_stress']
s5_baseline, s5_amusement, s5_stress = D['S5_chest+wrist_baseline'], D['S5_chest+wrist_amusement'], D['S5_chest+wrist_stress']
s6_baseline, s6_amusement, s6_stress = D['S6_chest+wrist_baseline'], D['S6_chest+wrist_amusement'], D['S6_chest+wrist_stress']
s7_baseline, s7_amusement, s7_stress = D['S7_chest+wrist_baseline'], D['S7_chest+wrist_amusement'], D['S7_chest+wrist_stress']
s8_baseline, s8_amusement, s8_stress = D['S8_chest+wrist_baseline'], D['S8_chest+wrist_amusement'], D['S8_chest+wrist_stress']
s9_baseline, s9_amusement, s9_stress = D['S9_chest+wrist_baseline'], D['S9_chest+wrist_amusement'], D['S9_chest+wrist_stress']
s10_baseline, s10_amusement, s10_stress = D['S10_chest+wrist_baseline'], D['S10_chest+wrist_amusement'], D['S10_chest+wrist_stress']
s11_baseline, s11_amusement, s11_stress = D['S11_chest+wrist_baseline'], D['S11_chest+wrist_amusement'], D['S11_chest+wrist_stress']
s13_baseline, s13_amusement, s13_stress = D['S13_chest+wrist_baseline'], D['S13_chest+wrist_amusement'], D['S13_chest+wrist_stress']
s14_baseline, s14_amusement, s14_stress = D['S14_chest+wrist_baseline'], D['S14_chest+wrist_amusement'], D['S14_chest+wrist_stress']
s15_baseline, s15_amusement, s15_stress = D['S15_chest+wrist_baseline'], D['S15_chest+wrist_amusement'], D['S15_chest+wrist_stress']
s16_baseline, s16_amusement, s16_stress = D['S16_chest+wrist_baseline'], D['S15_chest+wrist_amusement'], D['S16_chest+wrist_stress']
s17_baseline, s17_amusement, s17_stress = D['S17_chest+wrist_baseline'], D['S17_chest+wrist_amusement'], D['S17_chest+wrist_stress']


# Data segmentation
window_size = 240
stride = 2

###
#for train set
baseline = []
amusement = []
stress = []

def create_sequences_train(baseline_value, amusement_value, stress_value):
    baseline_split_point = int(len(baseline_value)*0.8)
    amusement_split_point = int(len(amusement_value)*0.8)
    stress_split_point = int(len(stress_value)*0.8)

    for i in range(baseline_split_point - window_size + 1):
        if i % stride == 0:
            baseline.append(baseline_value[i : (i + window_size)])

    for i in range(amusement_split_point - window_size + 1):
        if i % stride == 0:
            amusement.append(amusement_value[i : (i + window_size)])

    for i in range(stress_split_point - window_size + 1):
        if i % stride == 0:
            stress.append(stress_value[i : (i + window_size)])

    return baseline, amusement, stress

baseline, amusement, stress = create_sequences_train(s2_baseline, s2_amusement, s2_stress)      
baseline, amusement, stress = create_sequences_train(s3_baseline, s3_amusement, s3_stress)      
baseline, amusement, stress = create_sequences_train(s4_baseline, s4_amusement, s4_stress)        
baseline, amusement, stress = create_sequences_train(s5_baseline, s5_amusement, s5_stress)       
baseline, amusement, stress = create_sequences_train(s6_baseline, s6_amusement, s6_stress)
baseline, amusement, stress = create_sequences_train(s7_baseline, s7_amusement, s7_stress)
baseline, amusement, stress = create_sequences_train(s8_baseline, s8_amusement, s8_stress)
baseline, amusement, stress = create_sequences_train(s9_baseline, s9_amusement, s9_stress)
baseline, amusement, stress = create_sequences_train(s10_baseline, s10_amusement, s10_stress)
baseline, amusement, stress = create_sequences_train(s11_baseline, s11_amusement, s11_stress)
baseline, amusement, stress = create_sequences_train(s13_baseline, s13_amusement, s13_stress)
baseline, amusement, stress = create_sequences_train(s14_baseline, s14_amusement, s14_stress)
baseline, amusement, stress = create_sequences_train(s15_baseline, s15_amusement, s15_stress)
baseline, amusement, stress = create_sequences_train(s16_baseline, s16_amusement, s16_stress)
baseline, amusement, stress = create_sequences_train(s17_baseline, s17_amusement, s17_stress)

# Combined into stress group, non-stress group and all_data.
x_baseline = np.array(baseline) 
x_amusement = np.array(amusement)
x_stress = np.array(stress)

D={}
D['chest+wrist_segementation_baseline'] = x_baseline
D['chest+wrist_segementation_amusement'] = x_amusement
D['chest+wrist_segementation_stress'] = x_stress


###
#for test set
baseline_test = []
amusement_test = []
stress_test = []

def create_sequences_test(baseline_value, amusement_value, stress_value):
    baseline_split_point = int(len(baseline_value)*0.8)
    amusement_split_point = int(len(amusement_value)*0.8)
    stress_split_point = int(len(stress_value)*0.8)

    for i in range(baseline_split_point, len(baseline_value) - window_size + 1):
        if i % stride == 0:
            baseline_test.append(baseline_value[i : (i + window_size)])

    for i in range(amusement_split_point, len(amusement_value) - window_size + 1):
        if i % stride == 0:
            amusement_test.append(amusement_value[i : (i + window_size)])

    for i in range(stress_split_point, len(stress_value) - window_size + 1):
        if i % stride == 0:
            stress_test.append(stress_value[i : (i + window_size)])

    return baseline_test, amusement_test, stress_test


baseline_test, amusement_test, stress_test = create_sequences_test(s2_baseline, s2_amusement, s2_stress)      
baseline_test, amusement_test, stress_test = create_sequences_test(s3_baseline, s3_amusement, s3_stress)      
baseline_test, amusement_test, stress_test = create_sequences_test(s4_baseline, s4_amusement, s4_stress)        
baseline_test, amusement_test, stress_test = create_sequences_test(s5_baseline, s5_amusement, s5_stress)       
baseline_test, amusement_test, stress_test = create_sequences_test(s6_baseline, s6_amusement, s6_stress)
baseline_test, amusement_test, stress_test = create_sequences_test(s7_baseline, s7_amusement, s7_stress)
baseline_test, amusement_test, stress_test = create_sequences_test(s8_baseline, s8_amusement, s8_stress)
baseline_test, amusement_test, stress_test = create_sequences_test(s9_baseline, s9_amusement, s9_stress)
baseline_test, amusement_test, stress_test = create_sequences_test(s10_baseline, s10_amusement, s10_stress)
baseline_test, amusement_test, stress_test = create_sequences_test(s11_baseline, s11_amusement, s11_stress)
baseline_test, amusement_test, stress_test = create_sequences_test(s13_baseline, s13_amusement, s13_stress)
baseline_test, amusement_test, stress_test = create_sequences_test(s14_baseline, s14_amusement, s14_stress)
baseline_test, amusement_test, stress_test = create_sequences_test(s15_baseline, s15_amusement, s15_stress)
baseline_test, amusement_test, stress_test = create_sequences_test(s16_baseline, s16_amusement, s16_stress)
baseline_test, amusement_test, stress_test = create_sequences_test(s17_baseline, s17_amusement, s17_stress)
        

# Combined into stress group, non-stress group and all_data.
y_baseline = np.array(baseline_test)  
y_amusement = np.array(amusement_test) 
y_stress = np.array(stress_test)

T={}
T['chest+wrist_segementation_baseline_test'] = y_baseline
T['chest+wrist_segementation_amusement_test'] = y_amusement
T['chest+wrist_segementation_stress_test'] = y_stress


with open("../data/train.pkl", "wb") as fp:
    pickle.dump(D, fp, protocol = pickle.HIGHEST_PROTOCOL)

with open("../data/test.pkl", "wb") as fp:
    pickle.dump(T, fp, protocol = pickle.HIGHEST_PROTOCOL)
