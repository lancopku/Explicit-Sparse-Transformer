# Main code is in fairseq_deen_ende/fairseq/modules/sparse_activated_multihead_attention.py

cd fairseq_deen_ende
pip install --editable . --user
# 1 generate data following  fairseq_deen_ende/examples/translation/prepare-iwslt14.sh
# 2 training and evaluate
# div=-k means that you want to remain top k activations in  each row of attention matrix, if k is larger than sequence length, all activations will remain.
# take k=8 as an example
export CUDA_VISIBLE_DEVICES=0
results_name=topk
mkdir results/${results_name}
for div in -8
do
for seed in 1 2 3
do
    save=deen_fp
    cur_save=${save}_s${seed}_div${div}
    python3 train.py data-bin/iwslt14.tokenized.de-en -a transformer_iwslt_de_en --optimizer adam --lr 0.001 -s de -t en --label-smoothing 0.1 --dropout 0.4 --max-tokens 4000 \
     --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
     --criterion label_smoothed_cross_entropy --max-epoch 90 \
     --warmup-updates 4000 --warmup-init-lr '1e-07'  --update-freq 4 --fp16 --keep-last-epochs 50 \
     --adam-betas '(0.9, 0.98)' --save-dir checkpoint/${cur_save}  --div ${div} --seed ${seed} \
     --log-format json --tensorboard-logdir checkpoint/${cur_save}  2>&1 |  tee  -a checkpoint/${cur_save}.txt
    for i in  50 55 60 65 70 75 80 85 90
    do
    python3 average_checkpoints.py --inputs checkpoint/$cur_save  --num-epoch-checkpoints 10 --checkpoint-upper-bound ${i} --output checkpoint/$cur_save/avg_${i}.pt
    python3 generate.py data-bin/iwslt14.tokenized.de-en --path checkpoint/$cur_save/avg_${i}.pt --batch-size 128 --beam 5 --remove-bpe --quiet  > results/${results_name}/${cur_save}_avg_${i}_test.txt
    python3 generate.py data-bin/iwslt14.tokenized.de-en --path checkpoint/$cur_save/avg_${i}.pt --batch-size 128  \
    --beam 5 --remove-bpe --quiet  --gen-subset valid > results/${results_name}/${cur_save}_avg_${i}_valid.txt
    done
done
done
