# ASR-Assignments
2021 Autumn Tongji University SSE ASR assignments

## Assignment 1: MFCC Feature Extracting

**Requirements:**

`numpy`: math calculations

`matplotlib`: draw some diagrams

`librosa`: read and write .wav files

**Function invoking chain:**

```
read_audio(): read the .wav audio by librosa(sample rate 8000KHz)
     |
     V
pre_emphasis(): just do pre-emphasis 
     |
     V
frame_divide(): divide the frame with length 25ms and movement 10ms
     |
     V
windowing(): just windowing using hamming window
     |
     V
audio_fft(): implementing FFT(by calling the API of numpy)
     |
     V
mel_filter(): apply the mel-filter to the FFTed data and calculate the energy(by summing up the data)
     |
     V
dct(): DCT implemented by myself using the formula in the slides
     |
     V
lifter(): just lifter
     |
     V
delta(): calculate the delta of the data (twice to calculate the 2-rank delta)
     |
     V
norm(): normalize the data to get the final model
```

**References:**

1. https://blog.csdn.net/weixin_45272908/article/details/115641702
2. https://blog.csdn.net/Magical_Bubble/article/details/90295814
3. http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
4. https://github.com/jameslyons/python_speech_features
5. https://blog.csdn.net/jojozhangju/article/details/18678861
