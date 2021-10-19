import matplotlib.pyplot as plt
import numpy as np
import librosa


# the framework is referenced:
# reference: https://blog.csdn.net/weixin_45272908/article/details/115641702

# the functions below are mainly referenced:
# reference: https://blog.csdn.net/Magical_Bubble/article/details/90295814

# Got answers of some questions in:
# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
# and
# https://github.com/jameslyons/python_speech_features

# time-domain diagrams
def plot_time(data, sample_rate):
    """
    :param data: the audio data
    :param sample_rate: sample rate(44100Hz here)
    :return: no return
    """
    time = np.arange(0, len(data)) * (1.0 / sample_rate)
    plt.figure(figsize=(20, 5))
    plt.plot(time, data)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()


# freq-domain diagram
def plot_freq(signal, sample_rate, fft_size=512):
    """
    :param signal: as above
    :param sample_rate: as above
    :param fft_size: fast Fourier transform size
    :return: no return
    """
    xf = np.fft.rfft(signal, fft_size) / fft_size
    frequencies = np.linspace(0, sample_rate / 2, int(fft_size / 2 + 1))
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    plt.figure(figsize=(20, 5))
    plt.plot(frequencies, xfp)
    plt.xlabel('Frequency(Hz)')
    plt.ylabel('dB')
    plt.grid()
    plt.show()


# freq-spectrogram
def plot_spectrogram(spec, note):
    """
    :param spec: the spectrogram
    :param note: label for axis y
    :return: no return
    """
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.show()


# read the file in
def read_audio(path):
    """
    :param path: the path of the audio file
    :return: the data and the sample rate
    """
    data, sample_rate = librosa.load(path, sr=8000)  # self-defined sample rate
    # draw diagrams
    plot_time(data, sample_rate)
    plot_freq(data, sample_rate)
    # print(sample_rate)
    return data, sample_rate


def pre_emphasis(data, alpha=0.97):
    """
    :param data: the audio
    :param alpha: a constant ranged (0.95,0.99)
    :return: the emphasised data
    """
    # print(data)

    # pre-emphasis the data
    for i in range(1, int(len(data)), 1):
        data[i - 1] = data[i] - alpha * data[i - 1]
    # print(data)
    # visible_audio(data, sr)
    # draw diagrams
    plot_time(data, sr)
    plot_freq(data, sr)
    return data


def frame_divide(data, sample_rate):
    """
    :param data: the pre-emphasised audio
    :param sample_rate: sample rate of the audio
    :return: the divided frames
    """
    sig_len = len(data)  # the length of signal
    frame_len = int(sample_rate * 0.025)  # the length of frame(25 ms)
    frame_mov = int(sample_rate * 0.010)  # the frame movement(10 ms)
    frame_num = int(np.ceil((sig_len - frame_len) / frame_mov))  # frame number

    # pad zeros (that is because the ceil function makes the actual length of frame longer,
    # so we have to pad zeros)
    zero_num = (frame_num * frame_mov + frame_len) - sig_len
    zeros = np.zeros(zero_num)

    # concat data with zeros
    filled_signal = np.concatenate((data, zeros))
    # print(filled_signal)
    # extract the frame time (What is this exactly doing?)
    indices = np.tile(np.arange(0, frame_len), (frame_num, 1)) + \
              np.tile(np.arange(0, frame_num * frame_mov, frame_mov), (frame_len, 1)).T
    # print(indices[:2])

    # get the data
    indices = np.array(indices, dtype=np.int32)
    divided = filled_signal[indices]

    return divided, frame_len


def windowing(data_div, frame_len):
    """
    :param data_div: the divided data
    :param frame_len: length of frame
    :return: the windowed data
    """
    hamming_win = np.hamming(frame_len)  # hamming window
    windowed = data_div * hamming_win  # windowing
    # draw diagram
    plot_time(windowed[188], sr)
    plot_freq(windowed[188], sr)
    return windowed


def audio_fft(win_data):
    """
    :param win_data: the windowed audio data
    :return: the FFTed data
    """
    n_fft = 512  # points of FFT
    magnitude = np.absolute(np.fft.rfft(win_data, n_fft))  # the magnitude
    power = (1.0 / n_fft) * (magnitude ** 2)  # the powered data(normalized?)

    # test diagram
    plt.figure(figsize=(20, 5))
    plt.plot(power[188])
    plt.grid()
    plt.show()
    return power


# mel-filter
def mel_filter(fft_audio):
    """
    :param fft_audio: the FFTed audio
    :return: the filtered data
    """
    n_fft = 512  # as above
    low_mel = 300  # lowest mel-filter value
    high_mel = 1125 * np.log(1 + (sr / 2) / 700)  # (f = sr / 2(Nyquist theorem)) with the function in the slides
    n_filters = 26  # number of filters(26 - 40)
    points = np.linspace(low_mel, high_mel, n_filters + 2)  # generate sequential points(+2 for index convenience)
    inverses = 700 * (np.e ** (points / 1125) - 1)  # calculate the frequency using the inverse function like hi_mel

    filter_bank = np.zeros((n_filters, int(n_fft / 2 + 1)))  # the filter bank
    f = (n_fft / 2) * inverses / (sr / 2)  # f shares the same meaning as shown in the slide pp.39

    # generate filters
    for i in range(1, n_filters + 1):
        left = int(f[i - 1])
        center = int(f[i])
        right = int(f[i + 1])
        # f(m-1)<k<f(m)
        for j in range(left, center):
            filter_bank[i - 1, j + 1] = (j + 1 - f[i - 1]) / (f[i] - f[i - 1])
        # f(m)<k<f(m+1)
        for j in range(center, right):
            filter_bank[i - 1, j + 1] = (f[i + 1] - (j + 1)) / (f[i + 1] - f[i])
    # apply the filters to the audio
    energy = np.sum(fft_audio, 1)  # get the energy
    filtered_data = np.dot(fft_audio, filter_bank.T)  # dot product
    # print(fft_audio.shape)
    # print(filter_bank.shape)
    filtered_data = np.where(filtered_data == 0, np.finfo(float).eps,
                             filtered_data)  # if zero then replace with eps, else no change
    filtered_data = np.log10(filtered_data)
    plot_data = 20 * filtered_data  # log mel-filter outputs(converted to dB)
    # plot as a freq-spectrogram
    plot_spectrogram(plot_data.T, 'Filter Banks')
    # print(filtered_data.shape)
    return filtered_data, energy


# DCT function
def dct(log_mel, n_mfcc=26, n_ceps=12):
    """
    :param log_mel: log10(mel-filtered data)
    :param n_mfcc: number of MFCC(default 26, equal to the number of mel filters)
    :param n_ceps: number of cepstral coefficients
    :return: the DCTed audio data(keep only 12 frames of 26)
    """
    transpose = log_mel.T
    len_data = len(transpose)
    # print(len_data)
    dct_audio = []
    for j in range(n_mfcc):
        temp = 0
        for m in range(len_data):
            temp += (transpose[m]) * np.cos(j * (m + 0.5) * np.pi / len_data)
        dct_audio.append(temp)
    ret = np.array(dct_audio[1:n_ceps + 1])
    plot_spectrogram(ret, "MFCC coefficients")
    return ret


# lifter function
def lifter(dct_audio, n_lifter=22):
    """
    :param dct_audio: the DCTed audio
    :param n_lifter: the number of lifters
    :return: the liftered data
    """
    (n_coeff, n_frame) = dct_audio.shape
    n = np.arange(n_coeff)
    lift_audio = 1 + (n_lifter / 2) * np.sin(np.pi * n / n_lifter)
    liftered = dct_audio.T * lift_audio
    plot_spectrogram(liftered.T, "Liftered Audio Data")
    return liftered


# get the delta by the formula from https://blog.csdn.net/jojozhangju/article/details/18678861
def delta(lift_audio, k=1):
    """
    :param lift_audio: liftered audio
    :param k: the time gap for first-rank derivation
    :return: the delta array
    """
    delta_feat = []
    transpose = lift_audio.T
    q = len(transpose)  # the dimension of the mfcc
    for t in range(q):
        if t < k:
            delta_feat.append(transpose[t + 1] - transpose[t])
        elif t >= q - k:
            delta_feat.append(transpose[t] - transpose[t - 1])
        else:
            denominator = 2 * sum([i ** 2 for i in range(1, k + 1)])
            numerator = sum([i * (transpose[t + i] - transpose[t - i]) for i in range(1, k + 1)])
            delta_feat.append(numerator / denominator)
    return np.array(delta_feat)


# normalize the appended(deltas) data
def norm(mfcc):
    """
    :param mfcc: the un-normalized mfcc data
    :return: the normalized data
    """
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    transpose = mfcc.T
    for i in range(len(transpose)):
        transpose[i] = (transpose[i] - mean) / std
    return transpose.T


if __name__ == '__main__':
    x, sr = read_audio('./audio/hello/zh/zh_hello.wav')
    emp_data = pre_emphasis(x)  # pre-emphasis
    div_data, l_frame = frame_divide(emp_data, sr)  # frame dividing
    window_data = windowing(div_data, l_frame)  # windowing
    fft_data = audio_fft(window_data)  # fft
    mel_data, energy = mel_filter(fft_data)  # mel filters(outputs log mel-filter)
    dct_data = dct(mel_data)  # DCT transform
    liftered_data = lifter(dct_data)  # lifter DCT
    liftered_data = np.insert(liftered_data, 0, values=np.log10(energy), axis=1)  # add the energy dimension
    delta_data = delta(liftered_data)  # first delta
    delta_square = delta(delta_data.T)  # second delta
    mfcc_with_delta1 = np.insert(liftered_data, len(liftered_data[0]), values=delta_data, axis=1)  # append first delta
    mfcc_with_delta2 = np.insert(mfcc_with_delta1, len(mfcc_with_delta1[0]), values=delta_square, axis=1)  # append second delta
    norm_model = norm(mfcc_with_delta2)  # normalize the data

    '''
    2021/10/17
    Feature extracting completed! Congratulations to myself!
    2021/10/19
    shall insert log10(energy) to the liftered data
    '''
