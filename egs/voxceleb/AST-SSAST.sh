dataset=voxceleb
# full or balanced for audioset
set=full

bal=full
lr=1e-5
epoch=30
tr_data=./data/datafile/train_data.json
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.75
wa_start=1
wa_end=5

te_data=./data/datafile/test_data.json
freqm=48
timem=192
mixup=0
# corresponding to overlap of 6 for 16*16 patches
fstride=16
tstride=16
batch_size=12

# if we don't change the windon size
dataset_mean=-4.553
dataset_std=4.1933
window_change=False


# if we change the windon size accordingly
# dataset_mean=-4.40
# dataset_std=4.41856705
# window_change=True

audio_length=1024
Fshift=10
noise=False

metrics=acc
loss=CE
warmup=True
wa=True

seed=2024
exp=VoxCeleb_AST_SSAST_5-1-0.75-cls_${seed}
# exp=debug
exp_dir=/mnt/bear1/users/fj/ElasticAST/ast/${dataset}/${exp}

mkdir -p $exp_dir

CUDA_VISIBLE_DEVICES=0 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/train_AST.py \
--dataset ${dataset} \
--data_train ${tr_data} --data_val ${te_data} --exp_dir $exp_dir \
--label_csv ./data/class_labels_indices.csv --n_class 1251 \
--lr $lr --n_epochs ${epoch} --batch_size $batch_size \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std}  --noise ${noise} \
--audio_length ${audio_length} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--Fshift ${Fshift} --window_change ${window_change} \
--seed ${seed} \
--exp_name ${exp} --exp_dir ${exp_dir} \
--load_SSAST \
--wandb \
# --no_cls avg \


# --factorized_pos \

# --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} \
