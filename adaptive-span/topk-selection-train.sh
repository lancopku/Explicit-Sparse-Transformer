0. baseline
# enwiki8
ngpus=8
args="
--data data/enwik8 \
--nlayers 12 \
--hid-sz 512 \
--inner-hid-sz 2048 \
--nheads 8 \
--attn-span 8192 \
--block-sz 512 \
--batch-sz 64 \
--lr 0.07 \
--momentum 0 \
--dropout 0.3 \
--optim adagrad \
--lr-warmup 32000 \
--grad-clip 0.03 \
--niter 600 \
--nbatches 1000 \
--adapt-span \
--adapt-span-loss 0.0000005 \
--adapt-span-cache \
--distributed \
--checkpoint checkpoints/enwik8.pt
"

echo "Training ..."
# using the pytorch distributed launching
python3 -m torch.distributed.launch --nproc_per_node=$ngpus main.py --batch-split 4 $args 2>&1 | tee -a enwik8_adapt.txt

1.  top32
ngpus=8
args="
--data data/enwik8 \
--nlayers 12 \
--hid-sz 512 \
--inner-hid-sz 2048 \
--nheads 8 \
--attn-span 8192 \
--block-sz 512 \
--batch-sz 64 \
--lr 0.07 \
--momentum 0 \
--dropout 0.3 \
--optim adagrad \
--lr-warmup 32000 \
--grad-clip 0.03 \
--niter 600 \
--nbatches 1000 \
--adapt-span \
--adapt-span-loss 0.0000005 \
--adapt-span-cache \
--distributed \
--checkpoint checkpoints/enwik8_top32.pt \
--div -32
"
echo "Training ..."
# using the pytorch distributed launching
python3 -m torch.distributed.launch --nproc_per_node=$ngpus main.py --batch-split 4  $args 2>&1 | tee -a enwik8_adapt_top32.txt



2. top8
ngpus=8
args="
--data data/enwik8 \
--nlayers 12 \
--hid-sz 512 \
--inner-hid-sz 2048 \
--nheads 8 \
--attn-span 8192 \
--block-sz 512 \
--batch-sz 64 \
--lr 0.07 \
--momentum 0 \
--dropout 0.3 \
--optim adagrad \
--lr-warmup 32000 \
--grad-clip 0.03 \
--niter 600 \
--nbatches 1000 \
--adapt-span \
--adapt-span-loss 0.0000005 \
--adapt-span-cache \
--distributed \
--checkpoint checkpoints/enwik8_top8.pt \
--div -8
"
echo "Training ..."
# using the pytorch distributed launching
python3 -m torch.distributed.launch --nproc_per_node=$ngpus main.py --batch-split 2 $args 2>&1 | tee -a enwik8_adapt_top8.txt
