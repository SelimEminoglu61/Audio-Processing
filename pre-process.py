import librosa,librosa.display
import matplotlib.pyplot as plt
import numpy as np

file=(r"C:\Users\SELİM\Desktop\Arvis Proje\deneme.wav")

#waveform
signal,sr=librosa.load(file,sr=22050)# sr * T--> 22050 * 30
#librosa.display.waveshow(signal,sr=sr)
#plt.xlabel("Time")
#plt.ylabel("Amplitude")
#plt.show()

#fft-->spectrum
fft=np.fft.fft(signal)

magnitude=np.abs(fft)
frequency=np.linspace(0,sr,len(magnitude))

#We dont need whole graph, just take one side(left)--> Nyquist therom
left_frequency=frequency[:int(len(frequency)/2)]
left_magnitude=magnitude[:int(len(frequency)/2)]

#plt.plot(left_frequency,left_magnitude)
#plt.xlabel("Frequency")
#plt.ylabel("Magnitude")
#plt.show()

#stft-->spectogram

n_fft=2048 #--> number of samples fft
hop_length=512 #--> how much sliding on each interval to right

stft=librosa.core.stft(signal,hop_length=hop_length,n_fft=n_fft)
spectogram=np.abs(stft)

log_spectogram=librosa.amplitude_to_db(spectogram) #--> convert to desibel

#librosa.display.specshow(log_spectogram,sr=sr,hop_length=hop_length)
#plt.xlabel("Time")
#plt.ylabel("Frequency")
#plt.colorbar()
#plt.show()

#MFCCs
MFFCs=librosa.feature.mfcc(signal,n_fft=n_fft,hop_length=hop_length,n_mfcc=20)
librosa.display.specshow(MFFCs,sr=sr,hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCCs")
plt.colorbar()
plt.show()