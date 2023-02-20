import numpy as np
from torch import nn
import torch
import os
from sklearn.metrics import confusion_matrix


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt

def plot_matrix(matrix):
    labels_order = ['hap', 'sad', 'neu', 'ang']
    # labels_order = ['1', '2', '3', '4', '5']
    # 利用matplot绘图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels_order)
    ax.set_yticklabels([''] + labels_order)
    plt.xlabel('True')
    plt.ylabel('Predicted')
    for x in range(len(matrix)):
        for y in range(len(matrix)):
            plt.annotate(matrix[y, x], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    return plt


def train_and_test(train_loader, test_loader, model, optimizer, num_epochs, wav_or_dialogue,modal,
                               savefile=None):
    Best_Valid = 0
    Loss_Function = nn.CrossEntropyLoss(reduction='none')

    for epoch in range(num_epochs):
        confusion_Ypre = []
        confusion_Ylabel = []
        text_confusion_Ypre = []
        audio_confusion_Ypre = []
        model.train()

        for i, features in enumerate(train_loader):
            audio_train, text_train, train_audio_mask, train_text_mask, train_label, train_audio_seqlen, train_text_seqlen= features
            if modal in ['text', 'multi']:
                train_text_mask = train_text_mask.to(torch.bool).to(device)
                train_text_seqlen = train_text_seqlen.to(torch.int).to('cpu')
                text_train = text_train.to(device)
            if modal in ['audio', 'multi']:
                train_audio_mask = train_audio_mask.to(torch.bool).to(device)
                train_audio_seqlen = train_audio_seqlen.to(torch.int).to('cpu')
                audio_train = audio_train.to(device)
            train_label = train_label.to(device)

            if wav_or_dialogue=='wav':
                outputs, text_outputs, audio_outputs = model.forward(audio_train, text_train,
                                                                                         train_audio_seqlen,train_text_seqlen,
                                                                                        train_audio_mask,train_text_mask, modal)
                optimizer.zero_grad()
                loss = Loss_Function(outputs, train_label)
            elif wav_or_dialogue=='dialogue':
                if modal=='audio':
                    train_seqlen=train_audio_seqlen
                    train_mask=train_audio_mask
                else:
                    train_seqlen = train_text_seqlen
                    train_mask = train_text_mask
                outputs, text_outputs, audio_outputs= model.forward(audio_train, text_train,
                                                                                           train_seqlen,
                                                                                           train_mask, modal)
                train_label = train_label[:, 0:outputs.shape[0]]
                outputs = outputs.permute([1, 2, 0])
                train_label = train_label.permute([0, 2, 1]).squeeze(1)
                optimizer.zero_grad()
                loss = Loss_Function(outputs, train_label)
                if modal == 'multi':
                    audio_outputs = audio_outputs.permute([1, 2, 0])
                    text_outputs = text_outputs.permute([1, 2, 0])

            if modal == 'multi':
                loss_audio = Loss_Function(audio_outputs, train_label)
                loss_text = Loss_Function(text_outputs, train_label)
                total_loss_ = loss + loss_audio + loss_text
            else:
                total_loss_=loss

            if wav_or_dialogue=='dialogue':
                total_loss_ = total_loss_ * train_mask[:, :loss.shape[1]]

            total_loss = torch.sum(total_loss_, dtype=torch.float)
            total_loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            correct = 0
            text_correct = 0
            audio_correct = 0
            total = 0

            for i, features in enumerate(test_loader):
                audio_test, text_test, test_audio_mask, test_text_mask, test_label, test_audio_seqlen, test_text_seqlen = features
                if modal in ['text', 'multi']:
                    test_text_mask = test_text_mask.to(torch.bool).to(device)
                    test_text_seqlen = test_text_seqlen.to(torch.int).to('cpu')
                    text_test = text_test.to(device)
                if modal in ['audio', 'multi']:
                    test_audio_mask = test_audio_mask.to(torch.bool).to(device)
                    test_audio_seqlen = test_audio_seqlen.to(torch.int).to('cpu')
                    audio_test = audio_test.to(device)
                test_label = test_label.to(device)

                if wav_or_dialogue == 'wav':
                    outputs, text_outputs, audio_outputs = model.forward(audio_test, text_test,
                                                                         test_audio_seqlen, test_text_seqlen,
                                                                         test_audio_mask, test_text_mask, modal)
                    _, predict = torch.max(outputs, 1)
                    if modal=='multi':
                        _, text_predict = torch.max(text_outputs, 1)
                        _, audio_predict = torch.max(audio_outputs, 1)
                elif wav_or_dialogue == 'dialogue':
                    if modal == 'audio':
                        test_seqlen = test_audio_seqlen
                        test_mask = test_audio_mask
                    else:
                        test_seqlen = test_text_seqlen
                        test_mask = test_text_mask
                    outputs, text_outputs, audio_outputs = model.forward(audio_test, text_test,
                                                                         test_seqlen,
                                                                         test_mask, modal)
                    outputs = outputs.permute([1, 0, 2])
                    _, predict = torch.max(outputs, 2)
                    if modal=='multi':
                        text_output = text_outputs.permute([1, 0, 2])
                        audio_output = audio_outputs.permute([1, 0, 2])
                        _, text_predict = torch.max(text_output, 2)
                        _, audio_predict = torch.max(audio_output, 2)


                if wav_or_dialogue == 'wav':
                    test_label = test_label
                    total += np.sum(np.shape(test_label))
                elif wav_or_dialogue == 'dialogue':
                    test_label_original = test_label[:, :predict.shape[1]]
                    test_mask = test_mask[:, :predict.shape[1]]
                    test_label = torch.argmax(test_label_original, dim=2)
                    test_label = test_label * test_mask
                    total += test_mask.sum()
                total=total.item()

                if wav_or_dialogue=='wav':

                    correct += (predict == test_label).sum()
                    if modal == 'multi':
                        text_correct += (text_predict == test_label).sum()
                        audio_correct += (audio_predict == test_label).sum()
                        text_correct=text_correct.item()
                        audio_correct=audio_correct.item()
                elif wav_or_dialogue=='dialogue':
                    predict = predict * test_mask
                    correct += ((predict == test_label) * test_mask).sum()
                    if modal == 'multi':
                        text_predict = text_predict * test_mask
                        audio_predict = audio_predict * test_mask
                        text_correct += ((text_predict == test_label) * test_mask).sum()
                        audio_correct += ((audio_predict == test_label) * test_mask).sum()
                        text_correct = text_correct.item()
                        audio_correct = audio_correct.item()
                correct=correct.item()

                if wav_or_dialogue=='dialogue':
                    for i in range(predict.shape[0]):
                        confusion_Ypre.extend(predict[i][:test_seqlen[i]].cpu().numpy())
                        confusion_Ylabel.extend(test_label[i][:test_seqlen[i]].cpu().numpy())
                        if modal=='multi':
                            text_confusion_Ypre.extend(text_predict[i][:test_seqlen[i]].cpu().numpy())
                            audio_confusion_Ypre.extend(audio_predict[i][:test_seqlen[i]].cpu().numpy())
                elif wav_or_dialogue=='wav':
                    confusion_Ypre.extend(predict.cpu().numpy())
                    confusion_Ylabel.extend(test_label.cpu().numpy())
                    if modal=='multi':
                        audio_confusion_Ypre.extend(audio_predict.cpu().numpy())
                        text_confusion_Ypre.extend(text_predict.cpu().numpy())
            if correct / total >= Best_Valid:
                #torch.save(model, 'best.pt')
                text_Best_Valid=None
                text_acc_matrix=None
                audio_Best_Valid=None
                audio_acc_matrix=None
                ####################总的混淆矩阵##################
                Best_Valid = correct / total
                matrix = confusion_matrix(confusion_Ylabel, confusion_Ypre)
                total_num = np.sum(matrix, axis=1)
                acc_matrix = np.round(matrix / total_num[:, None], decimals=4)

                if modal=='multi':
                    # ###################文本的混淆矩阵#################################
                    text_Best_Valid = text_correct / total
                    matrix = confusion_matrix(confusion_Ylabel, text_confusion_Ypre)
                    total_num = np.sum(matrix, axis=1)
                    text_acc_matrix = np.round(matrix / total_num[:, None], decimals=4)
                    #######################音频的混淆矩阵########################
                    audio_Best_Valid = audio_correct / total
                    matrix = confusion_matrix(confusion_Ylabel, audio_confusion_Ypre)
                    total_num = np.sum(matrix, axis=1)
                    audio_acc_matrix = np.round(matrix / total_num[:, None], decimals=4)
                    torch.save(model, "best.pt")

        print(
            'Epoch: %d/%d; total utterance: %d ; correct utterance: %d ; Acc: %.2f%%; AudioAcc: %.2f%%; TextAcc: %.2f%%' % (
            epoch + 1, num_epochs, total, correct, 100 * (correct / total),
            100 * (audio_correct / total), 100 * (text_correct / total)))

    print("Best Valid Accuracy: %0.2f%%" % (100 * Best_Valid))
    print(acc_matrix)
    if modal=='multi':
        print("Best Text Valid Accuracy: %0.2f%%" % (100 * text_Best_Valid))
        print("Best Audio Valid Accuracy: %0.2f%%" % (100 * audio_Best_Valid))
    if savefile != None:
        np.savez(savefile, matrix=acc_matrix, ACC=Best_Valid, text_matrix=text_acc_matrix, text_ACC=text_Best_Valid,
                 audio_matrix=audio_acc_matrix, audio_ACC=audio_Best_Valid)
