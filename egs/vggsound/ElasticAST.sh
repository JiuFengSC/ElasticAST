dataset=vggsound
# full or balanced for audioset
set=full

bal=full
lr=1e-5
epoch=40
tr_data=/home/jfeng/FJ/MixedHopAST/egs/vggsound/data/datafiles/vgg_final_train.json
lrscheduler_start=10
lrscheduler_step=1
lrscheduler_decay=0.75
wa_start=1
wa_end=5

te_data=/home/jfeng/FJ/MixedHopAST/egs/vggsound/data/datafiles/vgg_final_test.json
freqm=48
timem=192
mixup=0
# corresponding to overlap of 6 for 16*16 patches
fstride=16
tstride=16
batch_size=12

# if we don't change the windon size
dataset_mean=-5.08
dataset_std=4.453
window_change=False


# if we change the windon size accordingly
# dataset_mean=-4.40
# dataset_std=4.41856705
# window_change=True

audio_length=1024
max_token_len=2048
Fshift=10
noise=False

metrics=acc
loss=BCE
warmup=True
wa=True
quality_ratio=[1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6,3.8,4]

seed=1337
exp=VGGSound_${max_token_len}_${batch_size}_[1:4:0.2]_ep${epoch}_seed${seed}
exp_dir=/mnt/bear2/users/fj/ElasticAST/elastic/${dataset}/${exp}

mkdir -p $exp_dir

CUDA_VISIBLE_DEVICES=1 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/train_ElasticAST.py \
--dataset ${dataset} \
--data_train ${tr_data} --data_val ${te_data} --exp_dir $exp_dir \
--label_csv /home/jfeng/FJ/MixedHopAST/egs/vggsound/data/class_labels_indices.csv --n_class 309 \
--lr $lr --n_epochs ${epoch} --batch_size $batch_size \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std}  --noise ${noise} \
--audio_length ${audio_length} --max_token_len ${max_token_len} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--Fshift ${Fshift} --window_change ${window_change} --quality_ratio ${quality_ratio} \
--seed ${seed} \
--exp_name ${exp} --exp_dir ${exp_dir} \
--imagenet_pretrain \
--Elastic_quality \
--wandb \
