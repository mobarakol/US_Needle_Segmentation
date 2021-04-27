import math
import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from model import ExcitationNet
from dataset import USNeedleDataset
from utils import seed_everything, get_dice

def train(train_loader, model, criterion, optimizer, epoch, epoch_iters):
    model.train()
    for batch_idx, (inputs, labels_seg) in enumerate(train_loader):
        inputs, labels_seg = inputs.to(device), labels_seg.to(device)
        optimizer.zero_grad()
        pred_seg = model(inputs)
        loss = criterion(pred_seg, labels_seg)
        pred_seg = pred_seg.data.max(1)[1].squeeze_(1).cpu().numpy()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % args.log_interval == 0:
            print('[epoch %d], [iter %d / %d], [train main loss %.5f], [lr %.4f]' % (
                epoch, batch_idx + 1, epoch_iters, loss.item(),
                optimizer.param_groups[0]['lr']))

def validate(valid_loader, model, args, criterion):
    model.eval()
    dice_all = []
    with torch.no_grad():
        for batch_idx, (inputs, labels_seg) in enumerate(valid_loader):
            inputs, labels_seg = inputs.to(device), np.array(labels_seg)
            pred_seg = model(inputs)
            labels_seg = np.array(labels_seg.cpu())
            pred_seg = pred_seg.data.max(1)[1].squeeze_(1).cpu().numpy()
            dice_all.append(get_dice(pred_seg, labels_seg))
        
    return np.mean(dice_all)

def main():
    dataset_train = USNeedleDataset(img_dir=args.data_root, is_train=True)
    train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)

    dataset_test = USNeedleDataset(img_dir=args.data_root, is_train=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2,
                              drop_last=True)
    model = ExcitationNet(n_classes=2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print('Length of dataset- train:', dataset_train.__len__(), ' valid:', test_loader.__len__())
    epoch_iters = dataset_train.__len__() / args.batch_size
    best_dice = 0
    best_epoch = 0
    for epoch in range( args.num_epoch):
        train(train_loader, model, criterion, optimizer, epoch, epoch_iters)
        avg_dice = validate(test_loader, model, args, criterion)
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'best_epoch_us_needle.pth.tar'))

        print('Epoch:%d ' % epoch,' Dice:%.4f'%avg_dice, 'Best Epoch:%d'%best_epoch, ' Best Dice:%.4f'%best_dice)
    
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything()
    parser = argparse.ArgumentParser(description='US Needle Segmentation')
    parser.add_argument('--num_classes', default=2, type=int, help="num of classes")
    parser.add_argument('--num_epoch', default=200, type=int, help="num of epochs")
    parser.add_argument('--log_interval', default=500, type=int, help="log interval")
    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate")
    parser.add_argument('--data_root', default='/media/mmlab/data/jiayi/data32_washeem_v6/', help="data root dir")
    parser.add_argument('--ckpt_dir', default='ckpt', help="data root dir")
    parser.add_argument('--batch_size', default=6, type=int, help="batch size")
    args = parser.parse_args()
    main()