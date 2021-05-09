# Explicit-Sparse-Transformer
In  Explicit Sparse Transformer, we propose an algorithm which sparse attention weights in transformer according to their activations.

2020 1/4  we upload code for explicit sparse transformer in tensor2tensor and fairseq, see t2t_envi_est.sh and fairseq_deen_est.sh for details.

2021 1/14  we address an import error related to SparseActivatedMultiheadAttention

2021 5/9 We find that top-k attention is additive with an adaptive local sparse attention method "Adaptive Attention Span in Transformers" https://arxiv.org/abs/1905.07799?context=cs.LG   and the top-k method can further reduce the length of the learned attention span and thus improves efficiency.

Here is an illustration drawn from training logs:
![image](https://github.com/lancopku/Explicit-Sparse-Transformer/blob/master/adaptive-span/k-adaptive-span.png)


