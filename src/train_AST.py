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
from models import ElasticAST, ASTModel
from scipy.stats import truncnorm
import numpy as np
from PIL import Image, ImageOps
import dataloader
from torch.utils.data import WeightedRandomSampler
from utilities import *





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
parser.add_argument('--noise', help='if augment noise', type=ast.literal_eval, default='False')

parser.add_argument("--metrics", type=str, default=None, help="evaluation metrics", choices=["acc", "mAP"])
parser.add_argument("--loss", type=str, default=None, help="loss function", choices=["BCE", "CE"])
parser.add_argument('--warmup', help='if warmup the learning rate', type=ast.literal_eval, default='False')
parser.add_argument('-w', '--num_workers', default=6, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')

parser.add_argument("--lrscheduler_start", type=int, default=2, help="which epoch to start reducing the learning rate")
parser.add_argument("--lrscheduler_step", type=int, default=1, help="how many epochs as step to reduce the learning rate")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay rate at each step")

parser.add_argument('--Fshift', help='Hop size when making fbank',  type=int, default=10)
parser.add_argument('--window_size', help='Window size when making fbank',  type=int, default=25)
parser.add_argument('--window_change', help='if window_change', type=ast.literal_eval, default='False')
parser.add_argument("--mel_bins", type=int, default=128, help="mel_bins")
parser.add_argument("--drop_compression", type=int, default=1, help="drop_compression")



parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='')
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--exp_name', default="", type=str, metavar='N', help='experiment name')
parser.add_argument('--model_size', default="vit224", type=str, metavar='N', help='experiment name')

parser.add_argument('--seed', default=1337, type=int, metavar='N', help='random seed')
parser.add_argument("--wandb", action="store_true")

parser.add_argument('--no_cls', help='we do not use cls_token, use avg pooling instead.', default=None)
parser.add_argument('--factorized_pos', help='we do not use cls_token, use avg pooling instead.', action="store_true")
parser.add_argument('--load_SSAST', help='we do not use cls_token, use avg pooling instead.', action="store_true")
parser.add_argument('--load_AST', help='we do not use cls_token, use avg pooling instead.', action="store_true")
parser.add_argument('--skip_norm', help='', action="store_true")

parser.add_argument('--pooling_size', default=1, type=int, metavar='N', help='random seed')


args = parser.parse_args()

import random
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
print('random seed: ', args.seed)

wandb_active = args.wandb
exp_name = args.exp_name
wandb_dir = args.exp_dir


os.makedirs(wandb_dir,exist_ok=True)
os.makedirs(f"{wandb_dir}/models/",exist_ok=True)

if wandb_active:
    wandb.init(project="ElasticAST", entity="kmmai",name=exp_name, dir=wandb_dir)
    wandb.save("/home/jfeng/FJ/ElasticAST/src/*.py",base_path="/home/jfeng/FJ/ElasticAST/src",policy="now")
    wandb.save("/home/jfeng/FJ/ElasticAST/src/models/*.py",base_path="/home/jfeng/FJ/ElasticAST/src",policy="now")
    wandb.save("/home/jfeng/FJ/ElasticAST/src/utilities/*.py",base_path="/home/jfeng/FJ/ElasticAST/src",policy="now")
    wandb.config.update(args)

if args.Fshift != 10:
    ratio = args.Fshift/10
    args.audio_length = int(args.audio_length/ratio)
    args.timem = int(args.timem/ratio)

audio_conf = {'num_mel_bins': args.mel_bins, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 
                'dataset': args.dataset, 'mode': 'train', 'mean': args.dataset_mean, 'std': args.dataset_std,'noise': args.noise,
                'Fshift': args.Fshift,'skip_norm': args.skip_norm,}
val_audio_conf = {'num_mel_bins': args.mel_bins, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                    'dataset': args.dataset, 'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std, 'noise': False,
                    'Fshift': args.Fshift,"window_size":args.window_size,'skip_norm': args.skip_norm,}
if args.bal == 'bal':
    print('balanced sampler is being used')
    samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDatasetR(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDatasetR(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf,pooling_size=args.pooling_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDatasetR(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf,pooling_size=args.pooling_size),
    batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# Configuration
batch_size = args.batch_size  # Adjust based on your GPU memory
num_epochs = args.n_epochs  # Updated to use args.n_epochs
exp_dir = args.exp_dir
global_step = 0
best_epoch, best_mAP, best_acc = 0, -np.inf, -np.inf



model = ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=args.mel_bins,
                                input_tdim=args.audio_length, imagenet_pretrain=True,
                                audioset_pretrain=False, model_size=args.model_size,
                                no_cls=args.no_cls, factorized_pos=args.factorized_pos,
                                drop_compression=args.drop_compression)

if args.load_AST:
    AST = torch.load('/mnt/bear1/users/fj/ElasticAST/flexiast2/audioset/audioset_AST_pretrained/models/audio_model.5.pth')
    out_dict = {}

    pos_emb = AST["module.v.pos_embed"] # [1, 514, 768]
    new_size = get_shape(16,16,16,input_tdim=args.audio_length)
    ori_size = get_shape(16,16,16,input_tdim=1024)
    pos_emb = resample_abs_pos_embed(pos_emb,
                        new_size=new_size,
                        old_size=ori_size,
                        num_prefix_tokens=1,
                        verbose=True)
    AST["module.v.pos_embed"] = torch.nn.Parameter(pos_emb)
    for k, v in AST.items(): # Adjust the name of dict
        # Why there is no mlp_head before?
        if "mlp_head" in k:
            continue
        out_dict[k[7:]] = v
    missed,unexpect =model.load_state_dict(out_dict, strict=False)



##### Load from SSAST model #####
if args.load_SSAST:
    SSAST = torch.load('/home/jfeng/FJ/ElasticAST/egs/SSAST-Base-Patch-400.pth')['model_state']
    out_dict = {}

    pos_emb = SSAST["module.v.pos_embed"] # [1, 514, 768]
    pos_emb = torch.cat((pos_emb[:,0,:].unsqueeze(1),pos_emb[:,2:,:]),dim=1)
    new_size = get_shape(16,16,16,input_tdim=args.audio_length)
    ori_size = get_shape(16,16,16,input_tdim=1024)
    pos_emb = resample_abs_pos_embed(pos_emb,
                        new_size=new_size,
                        old_size=ori_size,
                        num_prefix_tokens=1,
                        verbose=True)
    SSAST["module.v.pos_embed"] = torch.nn.Parameter(pos_emb)

    for k, v in SSAST.items(): # Adjust the name of dict
        # Why there is no mlp_head before?
        out_dict[k[7:]] = v
    missed,unexpect =model.load_state_dict(out_dict, strict=False)


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
optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)


main_metrics = args.metrics
warmup = args.warmup

# Move model to GPU if available

scaler = GradScaler()
loss_meter = AverageMeter()

torch.set_grad_enabled(True)


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
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            # Forward pass
            with autocast():                
                outputs = model(data_items)
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
                outputs = model(data_items)
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

    if mAP > best_mAP:
        best_mAP = mAP
        if main_metrics == 'mAP':
            best_epoch = epoch

    if acc > best_acc:
        best_acc = acc
        if main_metrics == 'acc':
            best_epoch = epoch


    torch.save(model.state_dict(), "%s/models/audio_model.%d.pth" % (args.exp_dir, epoch))

    if best_epoch == epoch:
        torch.save(model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
        torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))
    loss_meter.reset()


