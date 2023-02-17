import torch

import os
import logging
import numpy as np
import re
from torch.utils.data import Dataset
import copy
import pickle
from torch.nn import functional
from torch.utils.data import Dataset, DataLoader
import io

class LoadSingleWav(Dataset):
    def __init__(self,label_classes, modal='audio', train_or_test='train',dataset_name='feature.pkl'):
        file = open(dataset_name, 'rb')
        self.modal=modal
        self.label_classes=label_classes
        self.filename=file.name
        self.IDs, _, self.Speaker, self.videoLabels,_, self.videoText, _, self.videoAudio, _, self.Sentence, \
            self.TrainVid, self.TestVid,_, _= pickle.load(file, encoding='latin1')
        self.indexes = np.arange(len(self.IDs))
        self.trainVid = list(self.TrainVid)
        self.testVid = list(self.TestVid)
        self.text_max = 0
        self.audio_max=0
        self.train_or_test = train_or_test
        if self.modal in ['text', 'multi']:
            for vid in self.trainVid + self.testVid:
                self.videoText[vid]=self.videoText[vid].squeeze(0)
                if len(self.videoText[vid]) > self.text_max:
                    self.text_max = len(self.videoText[vid])
        if self.modal in ['audio', 'multi']:
            for vid in self.trainVid + self.testVid:
                if len(self.videoAudio[vid]) > self.audio_max:
                    self.audio_max = len(self.videoAudio[vid])

    def __getitem__(self, batch_index):

        indexes = self.indexes[batch_index]
        audio_feat=[]
        audio_len=0
        audio_mask=[]
        text_feat = []
        text_len = 0
        text_mask = []
        # 处理返回各种特征值
        if self.train_or_test == 'train':
            vid = self.trainVid[indexes]
        if self.train_or_test == 'test':
            vid = self.testVid[indexes]

        if self.modal in ['audio', 'multi']:
            tmp = np.array(self.videoAudio[vid]).reshape(
                [np.shape(self.videoAudio[vid])[0], np.shape(self.videoAudio[vid])[1], 1])

            # 将音频特征处理为统一长度方便放入batch。
            audio_len = len(self.videoAudio[vid])
            gap = self.audio_max - audio_len
            #print(gap)
            audio_feat = np.pad(tmp, [(0, gap), (0, 0), (0, 0)], mode='constant')
            audio_feat,audio_len = torch.tensor(audio_feat[:, :, 0]).type(torch.FloatTensor), torch.tensor(audio_len)
            audio_mask = np.zeros(np.shape(audio_feat)[0])
            audio_mask[:audio_len] = 1
        if self.modal in ['text', 'multi']:
            # 将文本特征处理为统一长度方便放入batch。
            self.videoText[vid]=np.array(self.videoText[vid])
            if len(np.shape(self.videoText[vid]))!=2:
                self.videoText[vid]=self.videoText[vid].squeeze(0)
            tmp = np.array(self.videoText[vid]).reshape(
                [np.shape(self.videoText[vid])[0], np.shape(self.videoText[vid])[1], 1])
            text_len = len(self.videoText[vid])
            gap = self.text_max - text_len

            text_feat = np.pad(tmp, [(0, gap), (0, 0), (0, 0)], mode='constant')
            text_feat, text_len = torch.tensor(text_feat[:, :, 0]), torch.tensor(text_len)
            text_mask = np.zeros(np.shape(text_feat)[0])
            text_mask[:text_len] = 1

        # 将label处理为统一长度方便放入batch。
        label=self.videoLabels[vid]

        #labels=functional.one_hot(torch.from_numpy(np.array(label).astype(np.int64)), num_classes=self.label_classes)

        return audio_feat, text_feat, audio_mask,text_mask, label, audio_len, text_len

    def __len__(self):
        if self.train_or_test == 'train':
            return len(self.trainVid)
        if self.train_or_test == 'test':
            return len(self.testVid)

class LoadDialogueWav(Dataset):
    def __init__(self,label_classes, modal='audio', train_or_test='train',dataset_name='feature.pkl'):
        file = open(dataset_name, 'rb')
        self.modal=modal
        self.label_classes=label_classes
        self.filename=file.name
        _, self.IDs_dialogue, self.Speaker, _, self.videoLabels_dialogue, _, self.videoText_dialogue, _, self.videoAudio_dialogue, self.Sentence, \
            _, _,self.TrainVid_dialogue, self.TestVid_dialogue= pickle.load(file, encoding='latin1')
        self.indexes = np.arange(len(self.IDs_dialogue))
        self.trainVid = list(self.TrainVid_dialogue)
        self.testVid = list(self.TestVid_dialogue)
        self.text_max = 0
        self.audio_max=0
        self.train_or_test = train_or_test
        if self.modal in ['text', 'multi']:
            for vid in self.trainVid + self.testVid:
                #self.videoText_dialogue[vid]=self.videoText_dialogue[vid].squeeze()
                if len(self.videoText_dialogue[vid]) > self.text_max:
                    self.text_max = len(self.videoText_dialogue[vid])
        if self.modal in ['audio', 'multi']:
            for vid in self.trainVid + self.testVid:
                if len(self.videoAudio_dialogue[vid]) > self.audio_max:
                    self.audio_max = len(self.videoAudio_dialogue[vid])

    def __getitem__(self, batch_index):

        indexes = self.indexes[batch_index]
        audio_feat=[]
        audio_len=0
        audio_mask=[]
        text_feat = []
        text_len = 0
        text_mask = []
        # 处理返回各种特征值
        if self.train_or_test == 'train':
            vid = self.trainVid[indexes]
        if self.train_or_test == 'test':
            vid = self.testVid[indexes]

        if self.modal in ['audio', 'multi']:
            tmp = np.array(self.videoAudio_dialogue[vid]).reshape(
                [np.shape(self.videoAudio_dialogue[vid])[0], np.shape(self.videoAudio_dialogue[vid])[1], 1])

            # 将音频特征处理为统一长度方便放入batch。
            audio_len = len(self.videoAudio_dialogue[vid])
            gap = self.audio_max - audio_len
            #print(gap)
            audio_feat = np.pad(tmp, [(0, gap), (0, 0), (0, 0)], mode='constant')
            audio_feat,audio_len = torch.tensor(audio_feat[:, :, 0]).type(torch.FloatTensor), torch.tensor(audio_len)
            audio_mask = np.zeros(np.shape(audio_feat)[0])
            audio_mask[:audio_len] = 1
        if self.modal in ['text', 'multi']:
            # 将文本特征处理为统一长度方便放入batch。
            self.videoText_dialogue[vid]=np.array(self.videoText_dialogue[vid])
            # if len(np.shape(self.videoText_dialogue[vid]))!=2:
            #     self.videoText_dialogue[vid]=self.videoText_dialogue[vid].squeeze()
            tmp = np.array(self.videoText_dialogue[vid]).reshape(
                [np.shape(self.videoText_dialogue[vid])[0], np.shape(self.videoText_dialogue[vid])[1], 1])
            text_len = len(self.videoText_dialogue[vid])
            gap = self.text_max - text_len

            text_feat = np.pad(tmp, [(0, gap), (0, 0), (0, 0)], mode='constant')
            text_feat, text_len = torch.tensor(text_feat[:, :, 0]), torch.tensor(text_len)
            text_mask = np.zeros(np.shape(text_feat)[0])
            text_mask[:text_len] = 1

        tmp = np.array(self.videoLabels_dialogue[vid]).reshape(
            [np.shape(self.videoLabels_dialogue[vid])[0], 1])
        # 将label处理为统一长度方便放入batch。
        labels = np.pad(tmp, [(0, gap), (0, 0)], mode='constant', constant_values=(3, 3))
        labels = torch.LongTensor(labels)

        #labels=functional.one_hot(torch.from_numpy(np.array(label).astype(np.int64)), num_classes=self.label_classes)

        return audio_feat, text_feat, audio_mask,text_mask, labels, audio_len, text_len

    def __len__(self):
        if self.train_or_test == 'train':
            return len(self.trainVid)
        if self.train_or_test == 'test':
            return len(self.testVid)

if __name__ == '__main__':
    batch_data_train = LoadDialogueWav(3,'multi','train', "feature.pkl")
    train_loader = DataLoader(dataset=batch_data_train, batch_size=2, drop_last=False, shuffle=True)
    i_1=0
    i_2=0
    i_3=3
    for i, features in enumerate(train_loader):
        audio_test, text_test, test_audio_mask, test_text_mask,test_label,\
                     seqlen_audio_test,seqlen_text_test = features
        if test_label.item()==0:
            i_1+=1
        if test_label.item()==1:
            i_2+=1
        if test_label.item()==2:
            i_3+=1
    print(i_1,i_2,i_3)