import os
import argparse
import time
from datetime import timedelta
import csv
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
import videotransforms

import numpy as np

from configs import Config
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import NSLT as Dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('--num_class', type=int)

args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def run(configs,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        save_model='',
        weights=None):
    print(configs)

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=configs.batch_size,
                                             shuffle=True,
                                             num_workers=2,
                                             pin_memory=True)

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=configs.batch_size,
                                                 shuffle=True,
                                                 num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'train': dataloader, 'test': val_dataloader}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    num_classes = dataset.num_classes
    i3d.replace_logits(num_classes)

    if weights:
        print('loading weights {}'.format(weights))
        i3d.load_state_dict(torch.load(weights))

    i3d.cuda()
    # i3d = nn.DataParallel(i3d)  # nếu nhiều GPU thì bật

    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)

    num_steps_per_update = configs.update_per_step
    steps = 0
    epoch = 0

    best_val_score = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)

    # ====== CSV Logging setup ======
    log_file = "train_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "step",
                "train_loc_loss", "train_cls_loss", "train_loss", "train_acc",
                "val_loc_loss", "val_cls_loss", "val_loss", "val_acc",
                "lr"
            ])
    # ===============================

    while steps < configs.max_steps and epoch < 350:
        epoch_start = time.time()
        print('Step {}/{}'.format(steps, configs.max_steps))
        print('-' * 10)

        epoch += 1
        metrics = {"train": {}, "val": {}}

        for phase in ['train', 'test']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

            phase_start = time.time()

            for batch_idx, data in enumerate(dataloaders[phase]):
                num_iter += 1
                if data == -1:
                    continue

                inputs, labels, vid = data
                inputs = inputs.cuda()
                t = inputs.size(2)
                labels = labels.cuda()

                per_frame_logits = i3d(inputs, pretrained=False)
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data.item()

                predictions = torch.max(per_frame_logits, dim=2)[0]
                gt = torch.max(labels, dim=2)[0]

                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                              torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data.item()

                for i in range(per_frame_logits.shape[0]):
                    confusion_matrix[torch.argmax(gt[i]).item(), torch.argmax(predictions[i]).item()] += 1

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data.item()
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()

                # ==== Print ETA mỗi 10 batch ====
                if batch_idx % 10 == 0:
                    elapsed = time.time() - phase_start
                    avg_time = elapsed / (batch_idx + 1)
                    remaining = (len(dataloaders[phase]) - (batch_idx + 1)) * avg_time
                    print(f"[{phase}] Epoch {epoch} Step {steps} | "
                          f"Batch {batch_idx+1}/{len(dataloaders[phase])} | "
                          f"ETA: {str(timedelta(seconds=int(remaining)))}")
                # ================================

            acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
            avg_loc_loss = tot_loc_loss / max(1, num_iter)
            avg_cls_loss = tot_cls_loss / max(1, num_iter)
            avg_loss = (tot_loss * num_steps_per_update) / max(1, num_iter)

            if phase == 'train':
                metrics["train"] = {"loc": avg_loc_loss, "cls": avg_cls_loss, "tot": avg_loss, "acc": acc}
            else:
                metrics["val"] = {"loc": avg_loc_loss, "cls": avg_cls_loss, "tot": avg_loss, "acc": acc}
                scheduler.step(avg_loss)

                if acc > best_val_score:
                    best_val_score = acc
                    model_name = save_model + f"nslt_{num_classes}_{str(steps).zfill(6)}_{acc:.3f}.pt"
                    torch.save(i3d.state_dict(), model_name)
                    print("Saved model:", model_name)

        # === Save to CSV ===
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, steps,
                metrics["train"]["loc"], metrics["train"]["cls"], metrics["train"]["tot"], metrics["train"]["acc"],
                metrics["val"]["loc"], metrics["val"]["cls"], metrics["val"]["tot"], metrics["val"]["acc"],
                optimizer.param_groups[0]['lr']
            ])
        # ===================

        # === Print epoch total time ===
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} finished in {str(timedelta(seconds=int(epoch_time)))}")
        # ==============================


if __name__ == '__main__':
    mode = 'rgb'
    root = {'word': "D:/Workplace/Ori_WLASL/WLASL/data"}
    save_model = 'checkpoints/'
    train_split = 'preprocess/nslt_100.json'
    weights = None
    config_file = 'configfiles/asl100.ini'

    configs = Config(config_file)
    print(root, train_split)
    run(configs=configs, mode=mode, root=root, save_model=save_model,
        train_split=train_split, weights=weights)
