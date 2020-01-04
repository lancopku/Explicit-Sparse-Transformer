# make dir
mkdir -p t2t_data t2t_datagen t2t_train t2t_output
# make data
python3 t2t-datagen --data_dir=t2t_data --tmp_dir=t2t_datagen \
--problem=translate_envi_iwslt32k

# set gpu
export CUDA_VISIBLE_DEVICES=0

# train
# div=-k means that you want to remain top k activations in  each row of attention matrix, if k is larger than sequence length, all activations will remain.
# take k=-6 as an example
for div in -6
do
for random_seed in 1
do
name=envi_apd${div}_s${random_seed}
python3 t2t-trainer --data_dir=t2t_data --problem=translate_envi_iwslt32k \
--model=transformer --hparams_set=transformer_base  --output_dir=t2t_output/${name} \
--train_steps=35000  --random_seed  ${random_seed} \
--hparams self_attention_type=sparse_dot_product,before_softmax=True,before_padding=False,d=${div}
python3 t2t-avg-all --model_dir t2t_output/${name} --output_dir t2t_avg/${name}
python3 t2t-decoder --data_dir=t2t_data --problem=translate_envi_iwslt32k \
--model=transformer --decode_hparams="beam_size=4,alpha=0.6"  \
--decode_from_file=t2t_datagen/tst2013.en --decode_to_file=${name}_test  \
--hparams_set=transformer_base --output_dir=t2t_avg/${name} --random_seed  ${random_seed} \
--hparams self_attention_type=sparse_dot_product,before_softmax=True,before_padding=False,d=${div}
python3 t2t-decoder --data_dir=t2t_data --problem=translate_envi_iwslt32k \
--model=transformer --decode_hparams="beam_size=4,alpha=0.6"  \
--decode_from_file=t2t_datagen/tst2012.en --decode_to_file=${name}_valid  \
--hparams_set=transformer_base --output_dir=t2t_avg/${name} --random_seed  ${random_seed} \
--hparams self_attention_type=sparse_dot_product,before_softmax=True,before_padding=False,d=${div}
done
done
# evaluate
python3 t2t-bleu --translation=${name}_valid --reference=t2t_datagen/tst2012.vi
python3 t2t-bleu --translation=${name}_test --reference=t2t_datagen/tst2013.vi


# div=k means that you want to reamain top sequence_length / k  activations.
# take k=4 as an example
for div in 4
do
for random_seed in 1
do
name=envi_apd${div}_s${random_seed}
python3 t2t-trainer --data_dir=t2t_data --problem=translate_envi_iwslt32k \
--model=transformer --hparams_set=transformer_base  --output_dir=t2t_output/${name} \
--train_steps=35000  --random_seed  ${random_seed} \
--hparams self_attention_type=sparse_dot_product,before_softmax=True,before_padding=False,d=${div}
python3 t2t-avg-all --model_dir t2t_output/${name} --output_dir t2t_avg/${name}
python3 t2t-decoder --data_dir=t2t_data --problem=translate_envi_iwslt32k \
--model=transformer --decode_hparams="beam_size=4,alpha=0.6"  \
--decode_from_file=t2t_datagen/tst2013.en --decode_to_file=${name}_test  \
--hparams_set=transformer_base --output_dir=t2t_avg/${name} --random_seed  ${random_seed} \
--hparams self_attention_type=sparse_dot_product,before_softmax=True,before_padding=False,d=${div}
python3 t2t-decoder --data_dir=t2t_data --problem=translate_envi_iwslt32k \
--model=transformer --decode_hparams="beam_size=4,alpha=0.6"  \
--decode_from_file=t2t_datagen/tst2012.en --decode_to_file=${name}_valid  \
--hparams_set=transformer_base --output_dir=t2t_avg/${name} --random_seed  ${random_seed} \
--hparams self_attention_type=sparse_dot_product,before_softmax=True,before_padding=False,d=${div}
done
done
# evaluate
python3 t2t-bleu --translation=${name}_valid --reference=t2t_datagen/tst2012.vi
python3 t2t-bleu --translation=${name}_test --reference=t2t_datagen/tst2013.vi
