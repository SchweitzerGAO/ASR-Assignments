import csv
import os


def generate_trainingfile_list():
    os.mkdir('./csvfiles')
    training_dirs = ['AE', 'AJ', 'AL', 'AW', 'BD', 'CB', 'CF', 'CR', 'DL', 'DN', 'EH', 'EL', 'FC', 'FD', 'FF', 'FI',
                     'FJ', 'FK', 'FL', 'GG']
    for directory in training_dirs:
        full_dir = './mfcc/' + directory
        for dirpath, dirname, files in os.walk(full_dir):
            for file in files:
                label = file[0]
                if label == 'O':
                    label = 10
                elif label == 'Z':
                    label = 11
                else:
                    label = int(label)
                row = [label, dirpath + '/' + file]
                with open('./csvfiles/training.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)


if __name__ == '__main__':
    generate_trainingfile_list()
    print("Done")
