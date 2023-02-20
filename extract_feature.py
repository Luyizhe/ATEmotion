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
    print(opensmile_path+' -C '+opensmile_config_path+' -I '+wav_path+' -O '+tmp_path)
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
    def __init__(self,modal='audio',opensmile_path=None,opensmile_config_path=None,csv_tmp_file=None,trainVid=None,testVid=None,wav_scp=None,dialogue_file=None,trainVid_dialogue=None,testVid_dialogue=None):
        if wav_scp==None:
            raise KeyError('Please provide wav_scp file path')
        if 'audio' in modal:
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
        self.wav_scp=wav_scp
        self.IDs=[]
        self.IDs_dialogue = []
        self.videoAudio={}
        self.videoAudio_dialogue = {}
        self.videoLabels={}
        self.videoText={}
        self.videoText_dialogue = {}
        self.videoVisual = {}
        self.videoLabels_dialogue={}
        self.Speaker = {}
        self.Sentence = {}
        self.TrainVid = []
        self.TestVid = []
        self.trainVid = trainVid
        self.testVid = testVid
        self.dialogue_file=dialogue_file
        self.trainVid_dialogue=trainVid_dialogue
        self.testVid_dialogue=testVid_dialogue

        self.create_IDs()
        self.split_train_test()

    def split_train_test(self):
        if self.trainVid==None or self.testVid==None:
            self.TrainVid=self.IDs[:int(0.7 * len(self.IDs))]
            self.TestVid = self.IDs[int(0.7 * len(self.IDs)):]
        else:
            trainVid=open(self.trainVid,'r')
            self.TrainVid=trainVid.readlines()[0].strip().split(' ')
            trainVid.close()
            testVid = open(self.testVid, 'r')
            self.TestVid = testVid.readlines()[0].strip().split(' ')
            testVid.close()
        if self.dialogue_file != None:
            if self.trainVid_dialogue==None or self.testVid_dialogue==None:
                self.TrainVid_dialogue = self.IDs_dialogue[:int(0.7 * len(self.IDs_dialogue))]
                self.TestVid_dialogue = self.IDs_dialogue[int(0.7 * len(self.IDs_dialogue)):]

    def create_IDs(self):
        wav_scp = open(self.wav_scp, 'r')
        for wav in wav_scp.readlines():
            vid, _ = wav.strip().split('\t')
            self.IDs.append(vid)
        if self.dialogue_file != None:
            dialogue = open(self.dialogue_file, 'r')
            for dialogue_wav_pair in dialogue.readlines():
                dialogue_id, wavs = dialogue_wav_pair.strip().split('\t')
                self.IDs_dialogue.append(dialogue_id)
            dialogue.close()

    def audio_feature_extract(self):
        wav_scp=open(self.wav_scp,'r')
        for wav in wav_scp.readlines():
            vid,wav_path=wav.strip().split('\t')
            opensmile_extract(self.opensmile_path,self.opensmile_config_path,wav_path,self.csv_tmp_file)
            self.videoAudio[vid]=extract_from_csvfile(self.csv_tmp_file).astype(float)
            os.system('rm %s'%self.csv_tmp_file)
            self.IDs.append(vid)
        if self.dialogue_file != None:
            dialogue=open(self.dialogue_file,'r')
            for dialogue_wav_pair in dialogue.readlines():
                dialogue_id,wavs=dialogue_wav_pair.strip().split('\t')
                wavs=wavs.strip().split(' ')
                tmp_feature=[]
                for wav_id in wavs:
                    tmp_feature.append(np.mean(self.videoAudio[wav_id],axis=0))
                self.videoAudio_dialogue[dialogue_id]=np.array(tmp_feature)
            dialogue.close()

    def text_feature_extract(self,trans,pretrain_path=r"../bert-base-uncased"):
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
        model = AutoModel.from_pretrained(pretrain_path)
        trans = open(trans, 'r')
        for every_line in trans.readlines():
            vid, sentence = every_line.strip().split('\t')
            self.Sentence[vid]=sentence
            DicID = tokenizer(self.Sentence[vid], return_tensors="pt")
            self.videoText[vid] = model(DicID['input_ids'])[0].detach().numpy()
            print(sentence+' Extracted Success')
        if self.dialogue_file != None:
            dialogue = open(self.dialogue_file, 'r')
            for dialogue_wav_pair in dialogue.readlines():
                dialogue_id, wavs = dialogue_wav_pair.strip().split('\t')
                wavs = wavs.strip().split(' ')
                tmp_feature = []
                for wav_id in wavs:
                    tmp_feature.append(np.mean(self.videoText[wav_id].squeeze(0), axis=0))
                self.videoText_dialogue[dialogue_id] = np.array(tmp_feature)
            dialogue.close()

            #pretrain_extract

    def convert_label_file(self,wav_label):
        wav_label = open(wav_label, 'r')
        for label in wav_label.readlines():
            vid, emotion_label = label.strip().split('\t')
            self.videoLabels[vid]=int(emotion_label)
        wav_label.close()
        if self.dialogue_file != None:
            dialogue = open(self.dialogue_file, 'r')
            for dialogue_wav_pair in dialogue.readlines():
                dialogue_id, wavs = dialogue_wav_pair.strip().split('\t')
                wavs = wavs.strip().split(' ')
                tmp_feature = []
                for wav_id in wavs:
                    tmp_feature.append(self.videoLabels[wav_id])
                self.videoLabels_dialogue[dialogue_id]=np.array(tmp_feature)
            dialogue.close()


    def convert_speaker_file(self,wav_speaker):
        wav_speaker = open(wav_speaker, 'r')
        for label in wav_speaker.readlines():
            vid, speaker_ID = label.strip().split('\t')
            self.Speaker[vid] = speaker_ID

    def done(self,output_file=r'./feature.pkl'):
        pickle.dump(
            (self.IDs, self.IDs_dialogue, self.Speaker, self.videoLabels, self.videoLabels_dialogue, self.videoText, self.videoText_dialogue,self.videoAudio, self.videoAudio_dialogue, self.Sentence,
             self.TrainVid, self.TestVid,self.TrainVid_dialogue, self.TestVid_dialogue),
            open(output_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    opensmile_path=r'../opensmile-2.3.0//SMILExtract'
    opensmile_config_path=r'../opensmile-2.3.0//config/IS09_emotion.conf'
    wav_scp=r'./wav.scp'
    dialogue_scp=r'./dialogue.scp'
    wav_label=r'./label.scp'
    trans=r'./trans.scp'
    feature_file=r'./feature.pkl'
    extractor=ExtractFeature(['audio'],opensmile_path,opensmile_config_path,wav_scp=wav_scp,dialogue_file=dialogue_scp)
    extractor.audio_feature_extract()
    extractor.text_feature_extract(trans)
    extractor.convert_label_file(wav_label)
    extractor.done(feature_file)