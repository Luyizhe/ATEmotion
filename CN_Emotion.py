from model import *
import model
import sys
import argparse
import LoadData
from torch.utils.data import Dataset, DataLoader
from train_test import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', type=str, default="tmp.txt", help='output confusion matrix to a file')
    parser.add_argument('--modal', type=str, default="audio", help='choose "text","audio","multi"')
    parser.add_argument('--fusion', type=str, default="ADD",
                        help='choose "AT_fusion" "Concat" "ADD" ,or "ADD" "Dot" in try models')
    parser.add_argument('--dataset', type=str, default="ground_truth",
                        help='choose "google_cloud" "speech_recognition" "ground_truth" "resources" or "v" "a" "d"')
    parser.add_argument('--criterion', type=str, default="CrossEntropyLoss", help='choose "MSELoss" "CrossEntropyLoss"')
    parser.add_argument('--loss_delta', type=float, default=1, help='change loss proportion')
    parser.add_argument('--label_classes', type=float, default=4, help='the numbers of classifications')
    parser.add_argument('--wav_or_dialogue', type=str, default='wav', help='modeling unit is wav or dialogue')
    args = parser.parse_args()
    modal = args.modal  # "text","audio","multi"
    fusion = args.fusion  # "AT_fusion" "Concat" "ADD"
    dataset = args.dataset  # "google cloud" "speech recognition" "ground truth" "v" "a" "d"
    criterion = args.criterion  # "MSELoss" "CrossEntropyLoss"
    label_classes = args.label_classes
    wav_or_dialogue = args.wav_or_dialogue
    # matrix_save_file=sys.argv[1]

    if wav_or_dialogue=='dialogue':
        batch_data_train = LoadData.LoadDialogueWav(label_classes,modal,'train')
        batch_data_test = LoadData.LoadDialogueWav(label_classes, modal, 'test')
        # batch_data_train = LoadData.LoadDiaData('train')
        # batch_data_test = LoadData.LoadDiaData('test')
    elif wav_or_dialogue == 'wav':
        batch_data_train = LoadData.LoadSingleWav(label_classes, modal, 'train')
        batch_data_test = LoadData.LoadSingleWav(label_classes, modal, 'test')
    # batch_data_train = LoadData.LoadDiaData_4('train')
    train_loader = DataLoader(dataset=batch_data_train, batch_size=batch_size, drop_last=False, shuffle=True)
    # batch_data_test = LoadData.LoadDiaData_4('test')
    test_loader = DataLoader(dataset=batch_data_test, batch_size=batch_size, drop_last=False, shuffle=False)

    #model = Multilevel_Multiple_Attentions(modal, fusion).to(device)
    if wav_or_dialogue=='dialogue':
        model = Dialogue_Model(label_classes,fusion).to(device)
    elif wav_or_dialogue=='wav':
        model = Sentence_Model(label_classes,fusion).to(device)

    lr,num_epochs = 1e-3,epochs
    #######################设置不同学习率################
    all_params = model.parameters()
    Text_params = []
    Audio_params = []
    Speaker_params=[]
    # 根据自己的筛选规则  将所有网络参数进行分组
    for pname, p in model.named_parameters():
       # print(pname.split('.')[0])zhang
        if pname.split('.')[0].endswith('audio'):
            Audio_params += [p]
        elif pname.split('.')[0].endswith('text'):
            Text_params += [p]
        if pname.split('.')[0].startswith('Speaker'):
            Speaker_params+=[p]
        # 取回分组参数的id
    params_id = list(map(id, Text_params)) + list(map(id, Audio_params))+list(map(id, Speaker_params))
    # 取回剩余分特殊处置参数的id
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    # 构建不同学习参数的优化器
    optimizer = torch.optim.Adam([
        {'params': other_params,'lr':lr},
        {'params': Audio_params, 'lr':lr},
        {'params': Text_params, 'lr': lr},
        {'params': Speaker_params, 'lr': 0}],
        weight_decay=1e-5
    )

    # criterion = nn.CrossEntropyLoss(reduction='none')


    train_and_test(train_loader, test_loader, model, optimizer, num_epochs, wav_or_dialogue, modal, args.outfile)

