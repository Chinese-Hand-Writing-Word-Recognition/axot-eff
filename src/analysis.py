from argparse import ArgumentParser
import base64
import datetime
import hashlib
import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np
from torch import nn
import time
import torch
import torchvision.transforms as transforms
from model import Classifier ,EfNetModel , EnsembleModel
from dataloader import YuShanDataset, get_image_transforms
from torch.utils.data import DataLoader , ConcatDataset
from utils import get_class_weight, F1Score, compute_acc, get_predict_index
import config
from tqdm import tqdm
from collections import defaultdict

def get_label_map():
    label2word = {}
    map = pd.read_csv('./label_map.csv')
    for idx,word in enumerate(map['Word']):
        label2word[idx] = word
    return label2word



def load_model():
    # 柏翰 原始資料 training
    # model1 = Classifier().to(device)
    # model1_path = './models/CNN_SGD_3e2__drop05085_300epoch.ckpt'
    # model1.load_state_dict(torch.load(model1_path))
    # model1.eval()

    # model2 = Classifier().to(device)
    # model2_path = './models/CNN_SGD_3e2__drop05085_500epoch.ckpt'
    # model2.load_state_dict(torch.load(model2_path))
    # model2.eval()

    # pretrained on .945 axot
    # model2 = EfNetModel(num_classes=801,pretrained_path='./models/eff_ori128_t_d9.pth').to(device)
    # model2.eval()
    # # axot 0.9455
    # model3 = EfNetModel(num_classes=801,dropout=0.9,pretrained_path='./models/eff_ori128_t_d9_f94550.pth').to(device)
    # model3.eval()

    # pretrained on .945 axot using 615 data
    model4 = EfNetModel(num_classes=801,dropout=0.9,pretrained_path='./models/eff_ori128_t_d9_0615.pth').to(device)
    model4.eval()


    # # 柏翰用615 data 
    # model5 = Classifier().to(device)
    # model5_path = './models/CNN_SGD_3e3_drop05085_300epoch_2_day1.ckpt'
    # model5.load_state_dict(torch.load(model5_path))
    # model5.eval()
    # # 柏翰用615 data
    # model6 = Classifier().to(device)
    # model6_path = './models/CNN_SGD_3e3_drop05085_300epoch_day1.ckpt'
    # model6.load_state_dict(torch.load(model6_path))
    # model6.eval()

    # # #柏翰的 old + 615 as training data
    # model7 = Classifier().to(device)
    # model7_path = './models/CNN_SGD_3e2_drop05085_300epoch_mix_data.ckpt'
    # model7.load_state_dict(torch.load(model7_path))
    # model7.eval()

    # #
    # model8 = Classifier().to(device)
    # model8_path = './models/CNN_SGD_3e3_drop05085_300epoch_mix_data_day2.ckpt'
    # model8.load_state_dict(torch.load(model8_path))
    # model8.eval()

    # 0615 + 0616 + olddata
    model9 = EfNetModel(num_classes=801,dropout=0.9,pretrained_path='./models/eff_ori128_t_d9_0616_with_old_data.pth').to(device)
    model9.eval()


    # 
    model10 = Classifier().to(device)
    model10_path = './models/CNN_SGD_3e2_drop05085_300epoch_2_weight_day1.ckpt'
    model10.load_state_dict(torch.load(model10_path))
    model10.eval()

    model11 = Classifier().to(device)
    model11_path = './models/CNN_SGD_3e3_drop05085_300epoch_2_weight_day1.ckpt'
    model11.load_state_dict(torch.load(model11_path))
    model11.eval()

    model12 = Classifier().to(device)
    model12_path = './models/CNN_SGD_3e2_drop05085_200epoch_all_data_weight.ckpt'
    model12.load_state_dict(torch.load(model12_path))
    model12.eval()

    # 615 616 only
    model13 = EfNetModel(num_classes=801,dropout=0.9,pretrained_path='./models/eff_ori128_t_d9_615_616.pth').to(device)
    model13.eval()

    models = [
        ('eff', model4),
        ('eff', model9), 
        ('eff', model13),
        ('mouth', model10),
        ('mouth', model11),
        ('mouth', model12)
    ]


    return models

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print(device)
    models = load_model()
    num_of_models = len(models)
    num_of_eff_models = 3
    num_of_mouth_models = 3

    level = config.base_level
    tf_train, tf_valid = get_image_transforms(level)
    test_set = YuShanDataset(
        "0615_allval.csv", 
        img_path="0615_allval", 
        root=config.data_root, 
        transform=tf_valid
    )
    test_set2 = YuShanDataset(
        "0616_allval.csv", 
        img_path="0616_allval", 
        root=config.data_root, 
        transform=tf_valid
    )
    test_sets = ConcatDataset([test_set,test_set2])
    print(len(test_sets))
    test_data = DataLoader(dataset=test_sets, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers)
    f1_score = F1Score()
    #評估每顆model
    # with torch.no_grad():
    #     for i, ( _ , model) in enumerate(models):
    #         print(f'model={i+1}')
    #         for th in np.arange(0, 1, 0.1):
            
    #             # progress_bar = tqdm(enumerate(test_data))
    #             preds = []
    #             targets = []
    #             # for j, (data, target) in progress_bar:
    #             for (data, target) in test_data:
    #                 data = data.to(device=device)
    #                 target = target.to(device=device)
    #                 targets.append(target.detach())
    #                 first = True
    #                 soft = nn.Softmax()
                    
    #                 output = model(data)
    #                 logits = soft(output)
                        
    #                 pred = torch.argmax(logits,dim=-1)
    #                 after , _ = torch.max(logits.cpu(),dim=-1)

    #                 after_th = after < th
    #                 pred[after_th] = 800
    #                 preds.append(pred)
            
    #             valid_f1 = f1_score(torch.cat(preds), torch.cat(targets))
    #             print(f'th={th} f1={valid_f1.item()}')
    #         print('----------')




    # # 評估 ensemble
    # with torch.no_grad():
    #     for a in np.arange(0, 1.1, 0.1):
            
    #         # progress_bar = tqdm(enumerate(test_data))
    #         preds = []
    #         targets = []
    #         # for j, (data, target) in progress_bar:
    #         for (data, target) in test_data:
    #             data = data.to(device=device)
    #             target = target.to(device=device)
    #             targets.append(target.detach())
    #             first = True
    #             soft = nn.Softmax()

    #             votes = {}
                
    #             for model_type , model in models:
    #                 output = model(data)
    #                 logits = soft(output)

    #                 if model_type == 'eff':
    #                     try:
    #                         votes['eff'] += logits
    #                     except:
    #                         votes['eff'] = logits

    #                 elif model_type == 'mouth':
    #                     try:
    #                         votes['mouth'] += logits
    #                     except:
    #                         votes['mouth'] = logits


    #             # eff vote
    #             logits_eff = votes['eff'] / num_of_eff_models
    #             max_prob_eff , max_idx_eff = torch.max(logits_eff.cpu(),dim=-1)
    #             should_be_isnull_eff = max_prob_eff < 0.9

    #             # mouth vote
    #             logits_mouth = votes['mouth'] / num_of_mouth_models
    #             max_prob_mouth , _ = torch.max(logits_mouth.cpu(),dim=-1)
    #             should_be_isnull_mouth = max_prob_mouth < 0.8
                
    #             should_be_isnull = torch.logical_and(should_be_isnull_mouth,should_be_isnull_eff)
                
    #             # a = 0.
    #             logits =  (a) * logits_mouth + (1-a) * logits_eff
    #             pred = torch.argmax(logits,dim=-1)
    #             pred[should_be_isnull] = 800

    #             # pred[max_prob_eff>b] = max_idx_eff[max_prob_eff>b].to(device)

    #             preds.append(pred)
        
    #         valid_f1 = f1_score(torch.cat(preds), torch.cat(targets))
    #         print(f'a={a} f1={valid_f1.item()}')
    #         # print(f'f1={valid_f1.item()}')
    #         # exit(1)
    #         # print(f'')

    with torch.no_grad():
        for th in np.arange(0, 1.1, 0.1):
            
            # progress_bar = tqdm(enumerate(test_data))
            preds = []
            targets = []
            # for j, (data, target) in progress_bar:
            for (data, target) in test_data:
                data = data.to(device=device)
                target = target.to(device=device)
                targets.append(target.detach())
                soft = nn.Softmax()

                first = True
                # votes = None
                
                for model_type , model in models:
                    output = model(data)
                    logits = soft(output)

                    if not first:
                        votes += logits
                    else:
                        votes = logits
                        first = False

                logits = votes / (num_of_eff_models + num_of_mouth_models)
                max_prob , max_idx= torch.max(logits.cpu(),dim=-1)
                should_be_isnull = max_prob < th

                pred = torch.argmax(logits,dim=-1)
                pred[should_be_isnull] = 800

                # pred[max_prob_eff>b] = max_idx_eff[max_prob_eff>b].to(device)

                preds.append(pred)
        
            valid_f1 = f1_score(torch.cat(preds), torch.cat(targets))
            print(f'th={th} f1={valid_f1.item()}')
            
            
