import numpy as np
import numpy as np
import scipy.signal as sg
import pywt
import cv2
import operator
from tompkins import Pan_Tompkins_QRS, heart_rate


sampling_rate = 360
# for correct R-peak location
tol = 0.05

def loads_cwt(record):
    # read ML II signal & r-peaks position and labels
    signal = np.array(record)
    
    QRS_detector = Pan_Tompkins_QRS()
    output_signal = QRS_detector.solve(signal)
    
    # # Convert ecg signal to numpy array
    # signal = ecg.iloc[:,1].to_numpy()

    # Find the R peak locations
    hr = heart_rate(output_signal,360)
    result = hr.find_r_peaks()
#     result = np.array(result)

#     # Clip the x locations less than 0 (Learning Phase)
#     result = result[result > 0]

    r_peaks = np.array(result)

    # denoising
    # filtering uses a 200-ms width median filter and 600-ms width median filter
    baseline = sg.medfilt(sg.medfilt(signal, int(0.2 * sampling_rate) - 1), int(0.6 * sampling_rate) - 1)
    filtered_signal = signal - baseline

    # align r-peaks
    newR = []
    for r_peak in r_peaks:
        r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
        r_right = np.minimum(r_peak + int(tol * sampling_rate), len(filtered_signal))
        newR.append(r_left + np.argmax(filtered_signal[r_left:r_right]))
    r_peaks = np.array(newR, dtype="int")

    # remove inter-patient variation
    normalized_signal = filtered_signal / np.mean(filtered_signal[r_peaks])

    return {
        "record": record,
        "signal": normalized_signal, "r_peaks": r_peaks
    }



def worker_cwt(data, wavelet, scales, sampling_period):

    # heartbeat segmentation interval
    before, after = 90, 110

    # decompose a signal in the time-frequency domain
    coeffs, frequencies = pywt.cwt(data["signal"], scales, wavelet, sampling_period)
    r_peaks = data["r_peaks"]

    # for remove inter-patient variation
    avg_rri = np.mean(np.diff(r_peaks))

    x1, x2 = [], []
    for i in range(len(r_peaks)):
        if i == 0 or i == len(r_peaks) - 1:
            continue

        # cv2.resize is used to sampling the scalogram to (100 x100)
        x1.append(cv2.resize(coeffs[:, r_peaks[i] - before: r_peaks[i] + after], (100, 100)))
        x2.append([
            r_peaks[i] - r_peaks[i - 1] - avg_rri,  # previous RR Interval
            r_peaks[i + 1] - r_peaks[i] - avg_rri,  # post RR Interval
            (r_peaks[i] - r_peaks[i - 1]) / (r_peaks[i + 1] - r_peaks[i]),  # ratio RR Interval
            np.mean(np.diff(r_peaks[np.maximum(i - 10, 0):i + 1])) - avg_rri  # local RR Interval
        ])    
    
    return x1, x2

def load_swt(record):
    beats = []
    size_RR_max = 20
    winL = 90
    winR = 90
    sampling_rate = 360
    signal = np.array(record)
    
    QRS_detector = Pan_Tompkins_QRS()
    output_signal = QRS_detector.solve(signal)
    hr = heart_rate(output_signal,360)
    result = hr.find_r_peaks()
    r_peaks = np.array(result)
    
    baseline = sg.medfilt(sg.medfilt(signal, int(0.2 * sampling_rate) - 1), int(0.6 * sampling_rate) - 1)
    filtered_signal = signal - baseline

    for a in range(len(r_peaks)):
        pos = r_peaks[a]
        if (pos > size_RR_max) and (pos < (len(record) - size_RR_max)):
            index, value = max(enumerate(record[pos - size_RR_max : pos + size_RR_max]), key=operator.itemgetter(1))
            pos = (pos - size_RR_max) + index

        if(pos > winL and pos < (len(record) - winR)):
            beats.append( (record[pos - winL : pos + winR]))

    return pos, beats

def worker_swt(data):
    _, beats = load_swt(data)
        
    coef = []
    for i in beats:
        coeffs = pywt.swt(i, 'db6', level=2)
        coef.append(coeffs[0])
    
    return coef

def worker_dwt(data):
    _, beats = load_swt(data)
        
    coef = []
    for i in beats:
        coeffs = pywt.wavedec(i, 'db6', level=3)
        coef.append(coeffs[0])
    
    return coef
