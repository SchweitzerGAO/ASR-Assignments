from audio_to_model import my_mfcc
import pickle as pkl
import os

import numpy


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
            file_path = './wav/' + key[-2:] + '/' + file
            label = file[0]
            if label == 'O':
                label = 10
            elif label == 'Z':
                label = 11
            else:
                label = int(label)
            feature = my_mfcc(file_path)
            temp = {label: feature}

            # write into the pkl file
            mfcc_file = './mfcc/' + key[-2:] + '/' + file.split('.')[0] + '.pkl'
            with open(mfcc_file, 'wb') as pkl_file:
                pkl.dump(temp, pkl_file)


if __name__ == '__main__':
    mfcc_extract()
    print("Done")
