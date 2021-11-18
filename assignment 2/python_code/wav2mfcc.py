import librosa
import pickle as pkl
import os


# mfcc using librosa API
def mfcc_extract():
    os.mkdir("./mfcc")
    wav_files = {}
    # create new dirs
    for dirpath, dirname, files in os.walk('./wav'):
        if len(dirname) != 0:
            pass
        wav_files[dirpath] = files
    for key, value in wav_files.items():
        if len(value) == 0:
            continue
        os.mkdir('./mfcc/' + key[-2:])
        for file in value:
            # extracting the mfcc
            wav_signal, fs = librosa.load('./wav/' + key[-2:] + '/' + file)
            label = file[0]
            if label == 'O':
                label = 10
            elif label == 'Z':
                label = 11
            else:
                label = int(label)
            feature = librosa.feature.mfcc(wav_signal, sr=fs)
            temp = {label: feature}

            # write into the pkl file
            mfcc_file = './mfcc/' + key[-2:] + '/' + file.split('.')[0] + '.pkl'
            with open(mfcc_file, 'wb') as pkl_file:
                pkl.dump(temp, pkl_file)


if __name__ == '__main__':
    mfcc_extract()
