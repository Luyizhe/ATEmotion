# ATEmotion
- 这是一个用作情感识别的python库。

- 支持音频/文本单模态，并且也支持两个模态的融合。

- 支持以单个句子为单位训练模型，并且预测单个句子的情感。也支持以对话为单位，并且预测对话中每个句子的情感。

- 可以直接通过```pip install ATEmotion```
安装本库，或者通过
```git clone https://github.com/Luyizhe/CN_Emotion.git```将代码下载到本地使用。

## 需要的依赖/其他工具
```
  python:
    torch==1.10.0
    numpy==1.21.6
    transformers==4.1.0
  如果使用IS09特征:
    opensmile==2.3.0
```

## 单个句子的情感识别
### 模型结构
![模型结构](https://github.com/Luyizhe/CN_Emotion/blob/main/wavfusion.png "Model")
- 音频按照帧提取每帧的特征，所有帧的特征组成该句子的音频总特征。然后通过BiGRU获取音频时序的信息后，再通过self-attention获得对情感有用的信息作为高层级特征，之后将特征进行mean-pooling，再通过一个FC层和一个softmax层得到分类结果。
- 文本按照字/子词提取句子特征，通过self-attention获得对情感有用的信息作为高层级特征（在IEMOCAP上尝试过attention之前加上GRU，模型性能没有什么提升，所以去掉了），之后将特征进行mean-pooling，再通过一个FC层和一个softmax层得到分类结果。
- 为了让单模态本身的特征差异化足够大，所以进行了logits层的融合，并且增加了perspective loss以加强各模态的特点。
### 用法
```
示例（音频使用IS09特征，文本使用bert特征）：
  import ATEmotion
  opensmile_path = r'.\openSMILE\SMILExtract'             #opensmile路径
  opensmile_config_path = r'.\openSMILE\config\IS09_emotion.conf'   #opensmile IS09特征的配置路径
  wav_scp = r'.\wav.scp'
  wav_label = r'.\label.scp'
  trans = r'.\trans.scp'
  feature_file = r'.\feature.pkl'                         #特征保存的路径
  pretrained_model = r'.\bert-base-uncased'               #transformers预训练模型的路径
  extractor = ATEmotion.extract_feature.ExtractFeature(['audio'], opensmile_path, opensmile_config_path, 
                                                    wav_scp=wav_scp)
  extractor.audio_feature_extract()
  extractor.text_feature_extract(trans,pretrain_path=pretrained_model)
  extractor.convert_label_file(wav_label)
  extractor.done(feature_file)
```
wav.scp 每一行格式：ID\t音频文件路径
label.scp 每一行格式：ID\t情感标签
trans.scp 每一行格式：ID\tWav对应文本内容


## Dialogue
### Model
对话级别的情感识别模型的结构见我们的论文：An Empirical Study and Improvement for Speech Emotion Recognition（ICASSP 2023）.
