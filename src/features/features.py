import pandas as pd
import numpy as np

def FFT(array,samplFreq=1,preproc=None):
    # this function perform the FFT trasform of a signal with windowind preprocessing
    # it return the FFT array (freq domain), the frequency points (freq domain)
    # and the windowed array (time domaint) preprocessing the data
    array=np.asarray(array)
    match preproc:
        case 'Hann':
            window = np.hanning(len(array))
            _prepArray=np.multiply(array,window) #preprocessed array
        case 'Hamming':
            window = np.hamming(len(array))
            _prepArray=np.multiply(array,window) #preprocessed array
        case 'Flip':
            np.flip(array[0:len(array)-1])
            _prepArray=np.concatenate((array, np.flip(array[0:len(array)-1])), axis=0)
        case _:
            _prepArray=array
    _aux = np.fft.fft(_prepArray)/len(_prepArray)          # Normalize amplitude
    if preproc=='Flip':
        _aux=_aux[::2] # if flip the number of point doubled, now i drop the odd numbers
    _timePeriod  = len(_prepArray)/samplFreq
    _frequencies = np.arange(int(len(_prepArray)/2))/_timePeriod #frequencies array of fft
    
    return abs(_aux[range(int(len(_prepArray)/2))]), _frequencies, _prepArray
