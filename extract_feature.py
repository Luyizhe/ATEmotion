import warnings
import os
import csv
import numpy as np
import logging
import pickle

def opensmile_extract(opensmile_path ,opensmile_config_path,wav_path,csv_tmp_file):
    if csv_tmp_file==None:
        tmp_path='tmp.csv'
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
                        self.csv_tmp_file = 'tmp.csv'
                    else:
                        self.csv_tmp_file = csv_tmp_file
        self.IDs=[]
        self.videoAudio=[]
        self.videoLabels=[]
        self.videoText=[]
        self.videoVisual = []
        self.Speaker = []
        self.Sentence = []

    def audio_feature_extract(self,wav_scp):
        wav_scp=open(wav_scp,'r')
        for wav in wav_scp.readlines():
            vid,wav_path=wav.split('\t')
            opensmile_extract(self.opensmile_path,self.opensmile_config_path,wav_path,self.csv_tmp_file)
            self.videoAudio[vid]=extract_from_csvfile(self.csv_tmp_file)


    def text_feature_extract(self,trans):
        trans = open(trans, 'r')
        for every_line in trans.readlines():
            vid, sentence = every_line.split('\t')
            self.Sentence[vid]=sentence
            #pretrain_extract


    def convert_label_file(self,wav_label):
        wav_label = open(wav_label, 'r')
        for label in wav_label.readlines():
            vid, emotion_label = label.split('\t')
            self.videoLabels[vid]=emotion_label

    def convert_speaker_file(self,wav_speaker):
        wav_speaker = open(wav_speaker, 'r')
        for label in wav_speaker.readlines():
            vid, speaker_ID = label.split('\t')
            self.Speaker[vid] = speaker_ID

    def done(self):
        pickle.dump(
            (self.IDs, self.Speaker, self.videoLabels, self.videoText, self.videoAudio, self.videoVisual, self.Sentence,
             trainVid, testVid),
            open("./IEMOCAP_features_HubertBertText4_Class.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    #opensmile_path=r'D:\software\openSMILE\SMILExtract.exe'
    opensmile_path=r'E:\NLP\SpeechEmotion\demo\openSMILE\SMILExtract.exe'
    #opensmile_config_path=r'D:\software\openSMILE\config\IS09_emotion.conf'
    opensmile_config_path=r'E:\NLP\SpeechEmotion\demo\openSMILE\config\IS09_emotion.conf'
    wav_scp=r'E:\NLP\SpeechEmotion\dataset\extract_script_test\test.scp'
    wav_label=r'E:\NLP\SpeechEmotion\dataset\extract_script_test\label.scp'
    extractor=ExtractFeature('audio',opensmile_path,opensmile_config_path)
    audio_feature=extractor.audio_feature_extract(r'E:\NLP\SpeechEmotion\dataset\extract_script_test')
    print(audio_feature.shape)