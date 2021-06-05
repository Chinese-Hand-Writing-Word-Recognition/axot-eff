# std lib
import warnings
import os

# 3rd lib
import torch
from torch import optim, nn
from ranger import Ranger
import glob
from PIL import Image
from tqdm import tqdm
import pandas as pd

# my lib
import config
from model import EfNetModel
from dataloader import YuShanDataset, get_image_transforms
from torch.utils.data import DataLoader
from utils import get_class_weight, F1Score, compute_acc, get_predict_index

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def output_test(path):
    model = EfNetModel(num_classes=801, pretrained_path=f"{config.save_dir}/{path}")
    model.to(device)

    # Preprocess image
    _, tf_valid = get_image_transforms()
    test_set = YuShanDataset(
        "test.csv", 
        img_path="test", 
        root=config.data_root, 
        transform=tf_valid
    )
    test_data = DataLoader(dataset=test_set, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers)

    # predict
    model.eval()
    preds = []
    with torch.no_grad():
        for img, _ in tqdm(test_data):
            img = img.to(device)
            pred = model(img)
            pred = get_predict_index(pred)
            preds += list(pred.cpu().numpy())
    
    # load_index
    df = pd.read_csv(f"{config.data_root}/test.csv")
    files_name = list(df["filename"])
    files_index = [file_name.split("_")[0] for file_name in files_name]

    df = pd.DataFrame()
    df['filename'] = files_index
    df['label'] = preds

    df.to_csv(f'result{config.threshold}.csv', index=0)
    print('Result file saved !')


def train(model, save_path = './efnet.pth'):
    level = config.base_level
    tf_train, tf_valid = get_image_transforms(level)
    
    # Data Loader
    train_set = YuShanDataset(
        "train.csv", 
        img_path="train", 
        root=config.data_root, 
        transform=tf_train
    )
    val_set = YuShanDataset(
        "val.csv", 
        img_path="val", 
        root=config.data_root, 
        transform=tf_valid
    )
    train_data = DataLoader(dataset=train_set, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)
    val_data = DataLoader(dataset=val_set, shuffle=False, batch_size=config.batch_size, num_workers=config.num_workers)
    
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=get_class_weight().to(device))
    # optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer = Ranger(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    f1_score = F1Score()

    best_f1 = 0.0
    # Training Epoch
    for epoch in range(1, config.EPOCHS+1):
        print('Epoch {}/{}'.format(epoch, config.EPOCHS))
        print('-' * 10)
        
        # train
        model.train()
        sum_train_loss = 0.0
        sum_train_acc = 0.0
        
        progress_bar = tqdm(enumerate(train_data))
        
        for i, (data, target) in progress_bar:

            data = data.to(device=device)
            target = target.to(device=device)

            optimizer.zero_grad()
            
            # forward
            pred = model(data)

            # backward
            loss = criterion(pred, target)
            loss.backward()
            
            # Clip the gradient norms for stable training.A
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # gradient
            optimizer.step()

            # metric
            acc = compute_acc(pred, target)

            # Record Loss & Acc
            sum_train_loss += loss
            sum_train_acc += acc
            
            progress_bar.set_description(f"[Train]: loss: {sum_train_loss/(i+1): .5f}, acc: {sum_train_acc/(i+1): .5f}")
            progress_bar.update()
        

        # Validation
        model.eval()
        sum_valid_loss = 0.0
        sum_valid_acc = 0.0

        preds = []
        targets = []
        progress_bar = tqdm(enumerate(val_data))
        for j, (data, target) in progress_bar:
            data = data.to(device=device)
            target = target.to(device=device)

            with torch.no_grad():
                pred = model(data)
                loss = criterion(pred, target)

            acc = compute_acc(pred, target)

            preds.append(get_predict_index(pred))
            targets.append(target.detach())

            sum_valid_loss += loss
            sum_valid_acc += acc

            progress_bar.set_description(f"[Valid]: loss: {sum_valid_loss/(j+1): .5f}, acc: {sum_valid_acc/(j+1): .5f}")
            progress_bar.update()
        
        valid_f1 = f1_score(torch.cat(preds), torch.cat(targets))
        print(f"[F1_score]: {valid_f1:.5f}, best: {best_f1:.5f}")

        # save the best model
        if valid_f1 > best_f1:
            torch.save(model.state_dict(), f"{config.save_dir}/{save_path}")
            print('save model with f1:',valid_f1)
            best_f1 = valid_f1
            
        # if (sum_train_acc / len(train_data) - valid_acc) > 0.05:
        #     level += 1
        #     train_set.transform, val_set.transform = get_image_transforms(level)
        #     print(f"=== Transform Level UP to {level} ===")


def main():
    model = EfNetModel(num_classes=801, dropout=config.dropout, pretrained_path=config.pretrained_path)
    # model = EfNetModel(num_classes=801, dropout=config.dropout)
    # output_test(config.pretrained_path)
    train(model, save_path=config.save_name)
    output_test(config.save_name)


if __name__ == "__main__":
    main()