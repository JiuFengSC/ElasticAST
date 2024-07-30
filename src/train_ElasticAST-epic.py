import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import timm
import os
import ast
from tqdm import tqdm
import argparse
import wandb
from torch.cuda.amp import autocast,GradScaler
from models import ElasticAST
from scipy.stats import truncnorm
import numpy as np
from PIL import Image, ImageOps
import dataloader
from torch.utils.data import WeightedRandomSampler
from utilities import *


# training pipeline with the offical schduler and settings


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--data_train", type=str, default='', help="training data json")
parser.add_argument("--data_val", type=str, default='', help="validation data json")
parser.add_argument("--data_eval", type=str, default='', help="evaluation data json")
parser.add_argument("--label_csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used")
parser.add_argument("--exp_dir", type=str, default="", help="directory to dump experiments")
parser.add_argument("--n_epochs", type=int, default=1, help="number of maximum training epochs")

parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")

parser.add_argument("--fstride", type=int, default=16, help="soft split freq stride, overlap=patch_size-stride")
parser.add_argument("--tstride", type=int, default=16, help="soft split time stride, overlap=patch_size-stride")

parser.add_argument("--dataset_mean", type=float, default=-4.2677393, help="the dataset spectrogram mean")
parser.add_argument("--dataset_std", type=float, default=4.5689974, help="the dataset spectrogram std")
parser.add_argument("--audio_length", type=int, default=1024, help="the dataset spectrogram std")
parser.add_argument("--max_token_len", type=int, default=2048, help="the dataset spectrogram std")
parser.add_argument('--noise', help='if augment noise', type=ast.literal_eval, default='False')

parser.add_argument("--metrics", type=str, default=None, help="evaluation metrics", choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default=None, help="loss function", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if warmup the learning rate', type=ast.literal_eval, default='False')
parser.add_argument('-w', '--num_workers', default=8, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')

parser.add_argument("--lrscheduler_start", type=int, default=2, help="which epoch to start reducing the learning rate")
parser.add_argument("--lrscheduler_step", type=int, default=1, help="how many epochs as step to reduce the learning rate")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay rate at each step")

parser.add_argument('--Fshift', help='Hop size when making fbank',  type=int, default=10)
parser.add_argument('--window_size', help='Window size when making fbank',  type=int, default=25)
parser.add_argument('--window_change', help='if window_change', type=ast.literal_eval, default='False')
parser.add_argument("--mel_bins", type=int, default=128, help="mel_bins")
parser.add_argument("--quality_ratio", type=str, default="[1,2,3,4]", help="mel_bins")
parser.add_argument("--imagenet_pretrain", action="store_true")

parser.add_argument("--Elastic_len", action="store_true")
parser.add_argument("--Elastic_quality", action="store_true")
parser.add_argument("--random_token_dropout",  type=float, default=0,)



parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='')
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--exp_name', default="", type=str, metavar='N', help='experiment name')
parser.add_argument('--seed', default=1337, type=int, metavar='N', help='random seed')
parser.add_argument("--wandb", action="store_true")
parser.add_argument('--skip_norm', help='', action="store_true")
parser.add_argument('--random_len_cut', help='', action="store_true")




args = parser.parse_args()

assert not (args.Elastic_len and args.Elastic_quality), "Elastic_len and Elastic_quality should have and at most one True"
assert args.Elastic_len or args.Elastic_quality,  "Elastic_len and Elastic_quality should have and at most one True"

import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
print('random seed: ', args.seed)

wandb_active = args.wandb
exp_name = args.exp_name
wandb_dir = args.exp_dir

# str to list
args.quality_ratio = ast.literal_eval(args.quality_ratio)
args.expand_ratio = args.quality_ratio[0]
# import ipdb; ipdb.set_trace()

os.makedirs(wandb_dir,exist_ok=True)
os.makedirs(f"{wandb_dir}/models/",exist_ok=True)

# if wandb_active:
#     wandb.init(project="ElasticAST", entity="kmmai",name=exp_name, dir=wandb_dir)
#     wandb.save("/home/jfeng/FJ/ElasticAST/src/*.py",base_path="/home/jfeng/FJ/ElasticAST/src",policy="now")
#     wandb.save("/home/jfeng/FJ/ElasticAST/src/models/*.py",base_path="/home/jfeng/FJ/ElasticAST/src",policy="now")
#     wandb.save("/home/jfeng/FJ/ElasticAST/src/utilities/*.py",base_path="/home/jfeng/FJ/ElasticAST/src",policy="now")
#     wandb.config.update(args)




audio_conf = {'num_mel_bins': args.mel_bins, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 
                'dataset': args.dataset, 'mode': 'train', 'mean': args.dataset_mean, 'std': args.dataset_std,'noise': args.noise,
                'Fshift': args.Fshift,"window_size":args.window_size,"window_change":args.window_change}
val_audio_conf = {'num_mel_bins': args.mel_bins, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                    'dataset': args.dataset, 'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False,
                    'Fshift': args.Fshift,"window_size":args.window_size,"window_change":args.window_change}

if args.Elastic_quality:
    if args.bal == 'bal':
        print('balanced sampler is being used')
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        train_loader = torch.utils.data.DataLoader(
            dataloader.ElasticQuality_Dataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf,quality_ratio=args.quality_ratio),
            batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True,collate_fn=dataloader.list_collate,)
    else:
        print('balanced sampler is not used')
        train_loader = torch.utils.data.DataLoader(
            dataloader.ElasticQuality_Dataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf,quality_ratio=args.quality_ratio),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True,collate_fn=dataloader.list_collate)

    val_loader = torch.utils.data.DataLoader(
        dataloader.ElasticQuality_Dataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf,quality_ratio=[1]),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True,collate_fn=dataloader.list_collate)
elif args.Elastic_len:
    from fvcore.common.config import CfgNode
    import yaml
    from epic_data import loader

    # Load the YAML file
    yaml_file = "/home/jfeng/FJ/epic-sounds-annotations/src/config.yaml"
    with open(yaml_file, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    # Convert dictionary to CfgNode
    cfg = CfgNode(cfg_dict)
    cfg.TRAIN.DATASET = "Epicsounds_elastic"
    cfg.AUDIO_DATA.NUM_FRAMES = args.audio_length
    cfg.T_MASK = args.timem
    cfg.F_MASK = args.freqm
    cfg.T_WARP = 5
    cfg.MIN_AUDIO_LENGTH = 192
    cfg.DATA_LOADER.NUM_WORKERS = 4
    if args.random_len_cut:
        cfg.RANDOM = True
    else:
        cfg.RANDOM = False

    # cfg.AUDIO_DATA.CLIP_SECS=int(args.audio_length/100)


    print("MIN_AUDIO_LENGTH: ", cfg.MIN_AUDIO_LENGTH)
    if args.exp_dir.split('/')[-1]=="debug":
        cfg.DATA_LOADER.NUM_WORKERS=0

    train_loader = loader.construct_loader_elastic(cfg, "train")
    val_loader = loader.construct_loader_elastic(cfg, "val")

# Configuration
batch_size = args.batch_size  # Adjust based on your GPU memory
num_epochs = args.n_epochs  # Updated to use args.n_epochs
exp_dir = args.exp_dir
global_step = 0


# Model, Loss, and Optimizer
model = ElasticAST(
    sample_size = (args.mel_bins,int(args.audio_length/args.expand_ratio)),
    patch_size = 16,
    num_classes = args.n_class,
    dim = 768,
    depth = 12,
    heads = 12,
    dropout = 0,
    emb_dropout = 0,
    token_dropout_prob = 0.1,  # token dropout of 10% (keep 90% of tokens)
    channels=1,
    imagenet_pretrain=args.imagenet_pretrain,
    random_token_dropout=args.random_token_dropout,
)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
model.to(device)

if args.loss == 'BCE':
    criterion = nn.BCEWithLogitsLoss()
elif args.loss == 'CE':
    criterion = nn.CrossEntropyLoss()
args.loss_fn = criterion

trainables = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainables, args.lr, weight_decay=0.0001, betas=(0.9, 0.999))

def lr_schedule(epoch):
    if epoch < 10:
        return 1.0  # No decay for epochs < 10
    elif epoch < 20:
        return 0.05  # Decays to 5% of the initial LR for epochs 10-19
    else:
        return 0.01  # Decays to 1% of the initial LR for epochs >= 20

# Apply the custom schedule to the optimizer using LambdaLR
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

main_metrics = args.metrics
warmup = args.warmup

# Move model to GPU if available

scaler = GradScaler()
loss_meter = AverageMeter()

torch.set_grad_enabled(True)

warmup_step = 2 * len(train_loader)


# Training and Validation Loop
for epoch in range(1,num_epochs+1):
    model.train()
    total_loss = 0
    with tqdm(train_loader) as tepoch:
        for data_items, labels in tepoch:
            data_items, labels = data_items, labels.to(device, non_blocking=True)
            B = len(data_items)
            for i in range(len(data_items)):
                data_items[i] = data_items[i].to(device, non_blocking=True)

             # first several steps for warm-up
            if global_step < warmup_step and warmup == True:
                warm_lr = args.lr * 0.01 + global_step * (args.lr - args.lr * 0.01) / warmup_step
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                if global_step % 100 == 0:
                    print(f'warm-up learning rate is {warm_lr}')
            elif global_step >= warmup_step and warmup == True:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                if global_step == warmup_step:    
                    print('end of warm-up, learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))


            # Forward pass
            with autocast():                
                outputs = model(data_items,group_samples=True,group_max_seq_len=args.max_token_len)
                if isinstance(criterion, torch.nn.CrossEntropyLoss):
                    loss = criterion(outputs, torch.argmax(labels.long(), axis=1))
                else:
                    loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # total_loss += loss.item()
            loss_meter.update(loss.item(), B)
            bar_dict = {"Epoch":epoch,"T_Loss":loss_meter.avg}
            tepoch.set_postfix(bar_dict)
            global_step += 1
            # torch.cuda.empty_cache()

    scheduler.step()
    print(f"Epoch [{epoch}/{num_epochs}], Average Training Loss: {loss_meter.avg:.4f}")

    # Validation
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        if not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
        A_predictions = []
        A_targets = []
        A_loss = []
        with tqdm(val_loader) as vepoch:
            for data_items, labels in vepoch:
                data_items, labels = data_items, labels
                for i in range(len(data_items)):
                    data_items[i] = data_items[i].to(device)
                outputs = model(data_items,group_samples=True,group_max_seq_len=args.max_token_len)
                outputs = torch.sigmoid(outputs)
                predictions = outputs.to('cpu').detach()

                A_predictions.append(predictions)
                A_targets.append(labels)

                labels = labels.to(device)
                if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss):
                    v_loss = args.loss_fn(outputs, torch.argmax(labels.long(), axis=1))
                else:
                    v_loss = args.loss_fn(outputs, labels)
                
                A_loss.append(v_loss.to('cpu').detach())


        outputs = torch.cat(A_predictions)
        labels = torch.cat(A_targets)
        v_loss = np.mean(A_loss)
        stats = calculate_stats(outputs, labels)

    mAP = np.mean([stat['AP'] for stat in stats])
    acc = stats[0]['acc']

    if main_metrics == 'mAP':
        print("mAP: {:.6f}".format(mAP))
    else:
        print("acc: {:.6f}".format(acc))
    print("train_loss: {:.6f}".format(loss_meter.avg))
    print("valid_loss: {:.6f}".format(v_loss))

    if wandb_active:
        wandb.log({"T_loss":loss_meter.avg,"V_loss":v_loss, "mAP":mAP,"ACC":acc})

    torch.save(model.state_dict(), "%s/models/audio_model.%d.pth" % (args.exp_dir, epoch))
    loss_meter.reset()


