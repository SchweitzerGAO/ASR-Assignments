import csv
import os


def generate_testingfile_list():
    testing_dirs = ['AH', 'AR', 'AT', 'BC', 'BE', 'BM', 'BN', 'CC', 'CE', 'CP', 'DF', 'DJ', 'ED', 'EF', 'ET', 'FA',
                    'FG', 'FH', 'FM', 'FP', 'FR', 'FS', 'FT', 'GA', 'GP', 'GS', 'GW', 'HC', 'HJ', 'HM', 'HR', 'IA',
                    'IB', 'IM', 'IP', 'JA', 'JH', 'KA', 'KE', 'KG', 'LE', 'LG', 'MI', 'NL', 'NP', 'NT', 'PC', 'PG',
                    'PH', 'PR', 'RK', 'SA', 'SL', 'SR', 'SW', 'TC']

    for directory in testing_dirs:
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
                with open("./csvfiles/testing.csv", 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow(row)


if __name__ == '__main__':
    generate_testingfile_list()
    print("Done")
