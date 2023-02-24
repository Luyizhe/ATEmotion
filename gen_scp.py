import os
import soundfile as sf
import numpy as np
import pickle

videoIDs, _, videoLabels, _, _, _, videoSentence, trainVid, testVid = pickle.load(
    open(r"E:\NLP\SpeechEmotion\SpeechEmotionByself\IEMOCAP_features_BertText_4Class.pkl", "rb"), encoding='latin1')

label_scp=open('label.scp','w')
wav_scp=open('wav.scp','w')
trans_scp=open('trans.scp','w')
dialogue_scp=open('dialogue.scp','w')
for vid in videoIDs:
    dialogue=vid+'\t'
    for clean in range(len(videoLabels[vid])):
        id=videoIDs[vid][clean]
        wav_path=r'E:\NLP\SpeechEmotion\dataset\IEMOCAP_full_release\Session%s\sentences\wav\\'%vid.strip().split('_')[0][4]+vid+'\\'+id+'.wav'
        sentence=videoSentence[vid][clean]
        label=videoLabels[vid][clean]
        dialogue+=id+' '
        wav_scp.writelines(id + '\t' + wav_path+'\n')
        label_scp.writelines(id+'\t'+str(label)+'\n')
        trans_scp.writelines(id+'\t'+sentence+'\n')
    dialogue_scp.writelines(dialogue[:-1]+'\n')
