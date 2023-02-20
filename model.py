import torch


import numpy as np

from torch import nn
import os
# 绘制多分类混淆矩阵

torch.set_printoptions(threshold=np.inf)
dropout = 0.2
batch_size = 20
max_audio_length = 700
max_text_length = 100
epochs = 150
audio_feature_Dimension = 100
audio_Linear = 100
audio_lstm_hidden = 100
text_embedding_Dimension = 100
Bert_text_embedding_Dimension = 768
Word2vec_text_embedding_Dimension=300
Elmo_text_embedding_Dimension = 1024
MFCC_Dimension = 40
Fbank_Dimension = 40
Wav2Vec_Dimension=512
IS13_Dimension=6373
IS09_Dimension=384
Hubert_Dimension=1024
text_Linear = 100
text_lstm_hidden = 100
mix_lstm_hidden = 100
gru_hidden = 50
attention_weight_num = 100
attention_head_num = 1
bidirectional = 2  # 2表示双向LSTM,1表示单向


audio_feature_total=0
mask_total=0
text_feature_total=0

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt

class Dialogue_Model(nn.Module):
    def __init__(self,label_classes, fusion="ADD"):
        super(Dialogue_Model, self).__init__()
        self.fusion = fusion
        # 串联concat

        self.Concat2_Linear = torch.nn.Linear(2 * label_classes, label_classes,
                                              bias=False)
        # 并联concat
        self.Concat1_Linear = torch.nn.Linear(label_classes, label_classes,
                                              bias=False)
        self.Omega_f = torch.normal(mean=torch.full((label_classes, 1), 0.0),
                                    std=torch.full((label_classes, 1), 0.01))
        # 特征直接相加
        # self.ADD= torch.nn.Linear(text_embedding_Dimension, audio_Linear,bias=False)
        self.Elmo_Linear_text = torch.nn.Linear(Elmo_text_embedding_Dimension, text_Linear, bias=True)
        self.Word2vec_Linear_text = torch.nn.Linear(Word2vec_text_embedding_Dimension, text_Linear, bias=True)
        self.MFCC_Linear = torch.nn.Linear(MFCC_Dimension, text_Linear, bias=True)
        self.Wav2Vec_Audio_Linear = torch.nn.Linear(Wav2Vec_Dimension, audio_Linear, bias=True)
        self.Bert_Text_Linear = torch.nn.Linear(Bert_text_embedding_Dimension, text_Linear, bias=True)
        self.Fbank_Linear = torch.nn.Linear(Fbank_Dimension, audio_Linear, bias=True)
        self.IS09_Linear = torch.nn.Linear(IS09_Dimension, audio_Linear, bias=True)
        self.IS13_Linear = torch.nn.Linear(IS13_Dimension, audio_Linear, bias=True)
        self.Audio_Linear = torch.nn.Linear(audio_feature_Dimension, audio_Linear, bias=True)
        self.Text_Linear = torch.nn.Linear(text_embedding_Dimension, text_Linear, bias=True)
        self.Linear_audio = torch.nn.Linear(attention_weight_num, 100, bias=True)
        self.Linear_text = torch.nn.Linear(attention_weight_num, 100, bias=True)
        self.Linear_fusion = torch.nn.Linear(attention_weight_num, 100, bias=True)
        self.Classify_Linear_audio = torch.nn.Linear(100, label_classes, bias=True)
        self.Classify_Linear_text = torch.nn.Linear(100, label_classes, bias=True)
        self.Classify_Linear = torch.nn.Linear(100, label_classes, bias=True)
        self.Softmax = torch.nn.Softmax(dim=2)
        # self.LN=torch.nn.functional.layer_norm([batch_size,])
        self.GRU_audio = torch.nn.GRU(input_size=audio_Linear, hidden_size=gru_hidden, num_layers=1,
                                      bidirectional=True)
        self.GRU_text = torch.nn.GRU(input_size=audio_Linear, hidden_size=gru_hidden, num_layers=1,
                                     bidirectional=True)
        self.GRU_fusion = torch.nn.GRU(input_size=label_classes, hidden_size=label_classes // 2, num_layers=1,
                                       bidirectional=True)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.Attention_audio = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num, dropout=0.2,
                                                           bias=True)
        self.Attention_audio_logits_weight = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num,
                                                                         dropout=0.2,
                                                                         bias=True)
        self.Linear_audio_logits_weight = torch.nn.Linear(attention_weight_num, label_classes, bias=True)

        self.Attention_text = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num, dropout=0.2,
                                                          bias=True)
        self.Attention_text_logits_weight = torch.nn.MultiheadAttention(attention_weight_num, attention_head_num,
                                                                        dropout=0.2,
                                                                        bias=True)
        self.Linear_text_logits_weight = torch.nn.Linear(attention_weight_num, label_classes, bias=True)
        self.Linear_Concat_to_100 = torch.nn.Linear(gru_hidden * 4, 100, bias=True)
        self.Linear_Concat_to_100_1 = torch.nn.Linear(gru_hidden * 4, 100, bias=True)
        self.Linear_100_to_weight = torch.nn.Linear(100, label_classes, bias=True)
        self.Speaker_Linear = torch.nn.Linear(100, 2, bias=True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, Audio_Features, Texts_Embedding, Seqlen, Mask, Modal):
        # input_text=Texts_Embedding
        Text_Emotion_Predict=None
        Audio_Emotion_Predict=None

        if Modal in ['audio','multi']:
            # input_audio = Audio_Features
            # mean = torch.mean(Audio_Features, dim=(1))
            # var = torch.var(Audio_Features, dim=(1))
            # div = torch.sqrt(var + 1e-05)
            # Audio_Features = (Audio_Features - mean[:, None, :]) / div[:, None, :]
            input_audio = self.IS09_Linear(Audio_Features)

            Audio_Padding = torch.nn.utils.rnn.pack_padded_sequence(input_audio, Seqlen,
                                                                    batch_first=True, enforce_sorted=False)
            # Audio_LSTM_Out、Text_LSTM_Out的维度分别是audio序列长度，batchsize，特征维度。
            Audio_GRU_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU_audio(Audio_Padding)[0])

            Audio_MinMask = Mask[:, :Audio_GRU_Out.shape[0]]
            Audio_Contribute = self.dropout(Audio_GRU_Out)

            Audio_Attention_Out, Audio_Attention_Weight = self.Attention_audio(Audio_Contribute, Audio_Contribute,
                                                                               Audio_Contribute,
                                                                               key_padding_mask=(~Audio_MinMask),
                                                                               need_weights=True)

            Audio_Dense1 = torch.tanh(self.Linear_audio(Audio_Attention_Out.permute([1, 0, 2])))
            Audio_Masked_Dense1 = Audio_Dense1 * Audio_MinMask[:, :, None]
            Audio_Dropouted_Dense1 = self.dropout(Audio_Masked_Dense1)
            Audio_Emotion_Output = self.Classify_Linear_audio(Audio_Dropouted_Dense1.permute([1, 0, 2]))
            Audio_Emotion_Predict = self.Softmax(Audio_Emotion_Output)
            Emotion_Predict=Audio_Emotion_Predict

        if Modal in ['text','multi']:
            ################使用text进行分类训练############################
            input_text = self.Bert_Text_Linear(Texts_Embedding)
            # 为了batch统一长度后标记原长度。
            Text_Padding = torch.nn.utils.rnn.pack_padded_sequence(input_text, Seqlen,
                                                                   batch_first=True, enforce_sorted=False)
            # Audio_LSTM_Out、Text_LSTM_Out的维度分别是audio序列长度，batchsize，特征维度。
            Text_GRU_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU_text(Text_Padding)[0])

            Text_MinMask = Mask[:, :Text_GRU_Out.shape[0]]
            Text_Contribute = self.dropout(Text_GRU_Out)
            # Text_Attention_Out = self.transformer_encoder_text(Text_Contribute, src_key_padding_mask=(1 - Text_MinMask))
            Text_Attention_Out, Text_Attention_Weight = self.Attention_text(Text_Contribute, Text_Contribute,
                                                                            Text_Contribute,
                                                                            key_padding_mask=(~Text_MinMask),
                                                                            need_weights=True)

            Text_Dense1 = torch.tanh(self.Linear_text(Text_Attention_Out.permute([1, 0, 2])))
            Text_Masked_Dense1 = Text_Dense1 * Text_MinMask[:, :, None]

            Text_Dropouted_Dense1 = self.dropout(Text_Masked_Dense1)

            Text_Emotion_Output = self.Classify_Linear_text(Text_Dropouted_Dense1.permute([1, 0, 2]))
            Text_Emotion_Predict = self.Softmax(Text_Emotion_Output)
            Emotion_Predict = Text_Emotion_Predict

        if Modal=='multi':
            if self.fusion == "ADD":
                Emotion_Output=Text_Emotion_Output+Audio_Emotion_Output
            # elif self.fusion == "Dot":
            #     Emotion_Output = Audio_Emotion_Output * Text_Emotion_Output
            # elif self.fusion == "Concat":
            #     Concat = torch.cat([Audio_Emotion_Output[:, :, :], Text_Emotion_Output[:, :, :]], 2)
            #     Emotion_Output = self.Concat2_Linear(Concat)
            # elif self.fusion == "AT_fusion":
            #     Concat = torch.cat([Text_Emotion_Output[:, :, None, :], Audio_Emotion_Output[:, :, None, :]], 2)
            #     u_cat = self.Concat1_Linear(Concat)
            #     NonLinear = self.dropout(torch.tanh(u_cat))
            #     alpha_fuse = torch.matmul(NonLinear, self.Omega_f)
            #     alpha_fuse = alpha_fuse.squeeze(3)
            #     normalized_alpha = self.Softmax(alpha_fuse)
            #     Emotion_Output = torch.matmul(u_cat.permute([0, 1, 3, 2]), normalized_alpha[:, :, :, None]).squeeze(dim=3)
            Emotion_Predict = self.Softmax(Emotion_Output)

        # return Emotion_Predict, Text_Emotion_Output, Audio_Emotion_Output,soft_logits_weight[:,:,:,1],soft_logits_weight[:,:,:,0]

        return Emotion_Predict, Text_Emotion_Predict, Audio_Emotion_Predict


class Sentence_Model(Dialogue_Model):
    def __init__(self,label_classes, fusion="ADD"):
        super(Sentence_Model, self).__init__(label_classes)
        self.fusion = fusion
        # 串联concat

        self.Attention1 = torch.nn.MultiheadAttention(gru_hidden * 2, attention_head_num, dropout=0.2,
                                                      bias=True,
                                                      add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.Attention2 = torch.nn.MultiheadAttention(text_Linear, attention_head_num, dropout=0.2,
                                                      bias=True,
                                                      add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None)
        self.Linear = torch.nn.Linear(100, 200, bias=True)
        self.Classify_text = torch.nn.Linear(100, label_classes, bias=True)
        self.Classify_audio = torch.nn.Linear(100, label_classes, bias=True)
        self.Softmax = torch.nn.Softmax(dim=1)
        self.GRU_text = torch.nn.GRU(input_size=text_Linear, hidden_size=gru_hidden, num_layers=1,
                                     bidirectional=True)
        self.GRU_audio = torch.nn.GRU(input_size=audio_Linear, hidden_size=gru_hidden, num_layers=1,
                                      bidirectional=True)
        self.GRU_fusion = torch.nn.GRU(input_size=gru_hidden * 2, hidden_size=gru_hidden, num_layers=1,
                                       bidirectional=True)
        self.GRU_fusion2 = torch.nn.GRU(input_size=text_Linear, hidden_size=gru_hidden, num_layers=1,
                                        bidirectional=True)
        self.CNN = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 12), stride=(1, 5),
                                   bias=True)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, Audio_Features, Texts_Embedding, Audio_Seqlen, Text_Seqlen, Audio_Mask, Text_Mask, Modal):
        Audio_Predict=None
        Text_Predict=None
        if Modal in ['text', 'multi']:
            input_text = self.Bert_Text_Linear(Texts_Embedding)
            text_Padding = torch.nn.utils.rnn.pack_padded_sequence(input_text, Text_Seqlen.to('cpu'), batch_first=True,
                                                                   enforce_sorted=False)
            # LSTM_Out、Text_LSTM_Out的维度分别是audio序列长度，batchsize，特征维度。
            GRU_text_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU_text(text_Padding)[0])
            MinMask_text = Text_Mask[:, :GRU_text_Out.shape[0]]
            new_input = input_text[:, :GRU_text_Out.shape[0], :].permute([1, 0, 2])
            # #print(new_input.shape)
            attention_out_text, _ = self.Attention2(new_input, new_input, new_input,
                                                    key_padding_mask=(~MinMask_text).type(torch.bool))
            attention_gru_text = attention_out_text
            GRUfusion_out_text = attention_gru_text.permute([1, 0, 2])
            Pooling_text = GRUfusion_out_text[0].mean(0)[None, :]
            for i in range(1, GRUfusion_out_text.shape[0]):
                Pooling_text = torch.cat([Pooling_text, GRUfusion_out_text[i][:Text_Seqlen[i]].mean(0)[None, :]], 0)
            Text_Output = self.Classify_text(Pooling_text)
            Text_Predict = self.Softmax(Text_Output)
            Emotion_Predict = Text_Predict

        if Modal in ['audio', 'multi']:
            mean = torch.mean(Audio_Features, dim=(1))
            var = torch.var(Audio_Features, dim=(1))
            div = torch.sqrt(var + 1e-05)
            Audio_Features = (Audio_Features - mean[:, None, :]) / div[:, None, :]
            input_audio = self.IS09_Linear(Audio_Features)
            # input_audio =  self.Wav2vec2_Linear(Audio_Features)

            # 获得audio过lstm的输出
            audio_Padding = torch.nn.utils.rnn.pack_padded_sequence(input_audio, Audio_Seqlen, batch_first=True,
                                                                    enforce_sorted=False)
            # LSTM_Out、Text_LSTM_Out的维度分别是audio序列长度，batchsize，特征维度。
            GRU_audio_Out, _ = torch.nn.utils.rnn.pad_packed_sequence(self.GRU_audio(audio_Padding)[0])

            MinMask_audio = Audio_Mask[:, :GRU_audio_Out.shape[0]]
            Contribute_audio = self.dropout(GRU_audio_Out)

            # ################两种设置QKV的方式###########################
            attention_out_audio, _ = self.Attention1(Contribute_audio, Contribute_audio, Contribute_audio,
                                                     key_padding_mask=(~MinMask_audio).type(torch.bool))
            attention_gru_audio = attention_out_audio
            GRUfusion_out_audio = attention_gru_audio.permute([1, 0, 2])

            Pooling_audio = GRUfusion_out_audio[0].mean(0)[None, :]
            for i in range(1, GRUfusion_out_audio.shape[0]):
                Pooling_audio = torch.cat([Pooling_audio, GRUfusion_out_audio[i][:Audio_Seqlen[i]].mean(0)[None, :]], 0)
            Audio_Output = self.Classify_audio(Pooling_audio)
            Audio_Predict = self.Softmax(Audio_Output)
            Emotion_Predict=Audio_Predict
        # print(Pooling.shape)
        ###############FC层后得到结果#######################

        if Modal =='multi':
            Fusion_out = Text_Output + Audio_Output
            Emotion_Predict = self.Softmax(Fusion_out)


        return Emotion_Predict, Text_Predict, Audio_Predict




