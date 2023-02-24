from ATEmotion.model import *
import argparse
import ATEmotion.LoadData
from torch.utils.data import Dataset, DataLoader
from ATEmotion.train_test import *
import ATEmotion


class Model():
    def __init__(self,modal='audio',fusion='ADD',label_classes=4,wav_or_dialogue='wav',feature='feature.pkl'):
        self.wav_or_dialogue=wav_or_dialogue
        self.modal=modal
        self.label_classes = label_classes
        if self.wav_or_dialogue=='dialogue':
            batch_data_train = ATEmotion.LoadData.LoadDialogueWav(label_classes,modal,'train',dataset_name='feature.pkl')
            batch_data_test = ATEmotion.LoadData.LoadDialogueWav(label_classes, modal, 'test',dataset_name='feature.pkl')
        elif self.wav_or_dialogue == 'wav':
            batch_data_train = ATEmotion.LoadData.LoadSingleWav(label_classes, modal, 'train',dataset_name='feature.pkl')
            batch_data_test = ATEmotion.LoadData.LoadSingleWav(label_classes, modal, 'test',dataset_name='feature.pkl')
        self.train_loader = DataLoader(dataset=batch_data_train, batch_size=batch_size, drop_last=False, shuffle=True)
        self.test_loader = DataLoader(dataset=batch_data_test, batch_size=batch_size, drop_last=False, shuffle=False)

        if self.wav_or_dialogue=='dialogue':
            self.model = Dialogue_Model(label_classes,fusion).to(device)
        elif self.wav_or_dialogue=='wav':
            self.model = Sentence_Model(label_classes,fusion).to(device)

        self.lr,self.num_epochs = 1e-3,epochs
        #######################设置不同学习率################
        all_params = self.model.parameters()
        Text_params = []
        Audio_params = []
        Speaker_params=[]
        # 根据自己的筛选规则  将所有网络参数进行分组
        for pname, p in self.model.named_parameters():
           # print(pname.split('.')[0])zhang
            if pname.split('.')[0].endswith('audio'):
                Audio_params += [p]
            elif pname.split('.')[0].endswith('text'):
                Text_params += [p]
            # 取回分组参数的id
        params_id = list(map(id, Text_params)) + list(map(id, Audio_params))+list(map(id, Speaker_params))
        # 取回剩余分特殊处置参数的id
        other_params = list(filter(lambda p: id(p) not in params_id, all_params))
        # 构建不同学习参数的优化器
        self.optimizer = torch.optim.Adam([
            {'params': other_params,'lr':self.lr},
            {'params': Audio_params, 'lr':self.lr},
            {'params': Text_params, 'lr': self.lr}],
            weight_decay=1e-5
        )

    def train(self):
        train_and_test(self.train_loader, self.test_loader, self.model, self.optimizer, self.num_epochs, self.wav_or_dialogue, self.modal, 'matrix')

    def inference(self,model_path=None):
        if model_path!=None:
            self.model=torch.load(model_path)
        if self.wav_or_dialogue=='dialogue':
            batch_data_test = ATEmotion.LoadData.LoadDialogueWav(self.label_classes, self.modal, 'test')
        elif self.wav_or_dialogue == 'wav':
            batch_data_test = ATEmotion.LoadData.LoadSingleWav(self.label_classes, self.modal, 'test')
        self.test_loader = DataLoader(dataset=batch_data_test, batch_size=batch_size, drop_last=False, shuffle=False)
        inference(self.test_loader, self.model, self.wav_or_dialogue, self.modal, 'matrix')

if __name__ == "__main__":
    model=Model('audio','ADD',4,'wav')
    model.train()