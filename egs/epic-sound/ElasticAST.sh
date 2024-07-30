dataset=epic-sound
# full or balanced for audioset
set=full

bal=full
lr=1e-5
epoch=30
tr_data=./data/datafiles/EPIC_Sounds_train.json
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.75
wa_start=1
wa_end=5

te_data=./data/datafiles/EPIC_Sounds_val.json
freqm=48
timem=192
mixup=0
# corresponding to overlap of 6 for 16*16 patches
fstride=16
tstride=16
batch_size=12

# if we don't change the windon size
dataset_mean=-8.660809
dataset_std=3.878928
window_change=False


audio_length=3072
max_token_len=2048
Fshift=5
window_size=10
noise=False

metrics=acc
loss=CE
warmup=True
wa=True

seed=1337
exp=${dataset}_elastic_len_pretrained-max${audio_length}-f${freqm}-t${timem}-offcial-scheduler-${lr}-random_len_cut
# exp=debug
exp_dir=/mnt/bear1/users/fj/ElasticAST/elastic/${dataset}/${exp}

mkdir -p $exp_dir

CUDA_VISIBLE_DEVICES=5 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/train_ElasticAST-epic.py \
--dataset ${dataset} \
--data_train ${tr_data} --data_val ${te_data} --exp_dir $exp_dir \
--label_csv ./data/class_labels_indices.csv --n_class 44 \
--lr $lr --n_epochs ${epoch} --batch_size $batch_size \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std}  --noise ${noise} \
--audio_length ${audio_length} --max_token_len ${max_token_len} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--Fshift ${Fshift} --window_size ${window_size} --window_change ${window_change} \
--seed ${seed} \
--exp_name ${exp} --exp_dir ${exp_dir} \
--imagenet_pretrain \
--Elastic_len \
--wandb \

# --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} \
