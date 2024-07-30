dataset=voxceleb
# full or balanced for audioset
set=full

bal=full
lr=1e-5
epoch=40
tr_data=/home/jfeng/FJ/FlexiAST/egs/voxceleb/data/datafile/train_data.json
lrscheduler_start=5
lrscheduler_step=1
lrscheduler_decay=0.75
wa_start=1
wa_end=5

te_data=/home/jfeng/FJ/FlexiAST/egs/voxceleb/data/datafile/test_data.json
freqm=48
timem=192
mixup=0
# corresponding to overlap of 6 for 16*16 patches
fstride=16
tstride=16
batch_size=12

# if we don't change the windon size
# dataset_mean=-6.6935 # weighted mean
# dataset_std=3.11822 # weighted std
dataset_mean=-5.842545 # mid mean
dataset_std=3.88305 # mid std
window_change=False


audio_length=3072
max_token_len=2048
Fshift=10
noise=False

metrics=acc
loss=CE
warmup=True
wa=True


seed=1337
exp=VoxCeleb_elastic_len_pretrained-5-1-0.75-max${audio_length}-SSAST-mid_norm-${lr}_seed${seed}_new_env
exp_dir=/mnt/bear1/users/fj/ElasticAST/elastic/${dataset}/${exp}

mkdir -p $exp_dir

CUDA_VISIBLE_DEVICES=6 CUDA_CACHE_DISABLE=1 python -W ignore ../../src/train_ElasticAST.py \
--dataset ${dataset} \
--data_train ${tr_data} --data_val ${te_data} --exp_dir $exp_dir \
--label_csv /home/jfeng/FJ/FlexiAST/egs/voxceleb/data/class_labels_indices.csv --n_class 1251 \
--lr $lr --n_epochs ${epoch} --batch_size $batch_size \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std}  --noise ${noise} \
--audio_length ${audio_length} --max_token_len ${max_token_len} \
--metrics ${metrics} --loss ${loss} --warmup ${warmup} --lrscheduler_start ${lrscheduler_start} --lrscheduler_step ${lrscheduler_step} --lrscheduler_decay ${lrscheduler_decay} \
--Fshift ${Fshift} --window_change ${window_change} \
--seed ${seed} \
--exp_name ${exp} --exp_dir ${exp_dir} \
--Elastic_len \
--load_SSAST \
--wandb \

