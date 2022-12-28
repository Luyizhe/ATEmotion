import warnings
import os
import csv
import numpy as np
import logging

def opensmile_extract(opensmile_path ,opensmile_config_path,wav_path,csv_tmp_file):
    if csv_tmp_file==None:
        tmp_path='tmp'
    else:
        tmp_path=csv_tmp_file
    os.system(opensmile_path+' -C '+opensmile_config_path+' -I '+wav_path+' -O '+tmp_path)
    logging.info('Opensmile extract '+wav_path+' features done')

def extract_from_csvfile(filename):
    invalid_num = 391
    rows = read_csvfile(filename)
    if len(rows) <= invalid_num:
        raise Exception('error: csv_fils=%s only has %d rows < %d' % (filename, len(rows), invalid_num))
    rows = rows[invalid_num:]
    rows = np.delete(rows, [0, len(rows[0]) - 1], axis=1)
    return rows

def read_csvfile(filename):
    with open(filename, 'r') as f:
        all_lines = csv.reader(f)
        rows = [row for row in all_lines]
        return rows

class ExtractFeature():
    def __init__(self,modal='audio',opensmile_path=None,opensmile_config_path=None,csv_tmp_file=None):
        if modal=='audio':
            if opensmile_path==None:
                raise ValueError('Please provide correct opensmile exe path')
            else:
                if opensmile_config_path==None:
                    raise ValueError('Please provide correct opensmile config file path')
                else:
                    self.opensmile_path = opensmile_path
                    self.opensmile_config_path=opensmile_config_path
                    self.csv_tmp_file=None
                    if csv_tmp_file==None:
                        warnings.warn("Because the value of the csv_tmp_file is missing," 
                                      " a tmp csv file will be created in current path."
                                      " Please ensure that you have read and write permissions for the current path.")
                    else:
                        self.csv_tmp_file = csv_tmp_file

    def audio_feature_extract(self,wav_dir):
        print(wav_dir)
        exit()
        total_features=[]
        try:
            for wav_file in os.listdir(wav_dir):
                if wav_file.endswith('wav'):
                    opensmile_extract(self.opensmile_path,self.opensmile_config_path,wav_file,self.csv_tmp_file)
                    total_features.append(extract_from_csvfile(self.csv_tmp_file))
        except:
            raise ValueError('The value of wav_dir is not a legal path')
        return np.array(total_features)

if __name__=='__main__':
    opensmile_path=r'D:\software\openSMILE\SMILExtract.exe'
    opensmile_config_path=r'D:\software\openSMILE\config\IS09_emotion.conf'
    extractor=ExtractFeature('audio',opensmile_path,opensmile_config_path)
    extractor.audio_feature_extract('dir')