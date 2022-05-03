# Mandarin-Wav2Vec2

In this repository, we show the experiments of Mandarin Wav2vec2.0(base) pre-trained on AISHELL-2 dataset. The results show that pre-trained the wav2vec2.0 models on specific languages from scratch can boost the performance of downstream tasks(e.g. ASR) compared to cross-lingual scenarios. We plan to release the pre-trained weight in the future.

## Pre-training setup: 

- We filtered the wav files that contained the same transcriptions in AISHELL-1 dev/test set to avoid information leak for the research purpose. The filtered pre-training set has ~960hr of raw speeches and we randomly sampled 1% of data for validation.
- We concatenated 6 wav files without shuffle to one file to accelerate the pre-training process and make the audio length close to the length of Librispeech samples.
- We used 8x Nvidia Tesla V100 and accumulated 8 updates to simulate 64 GPU. The total training time is about 10 days.
- We used [Fairseq](https://github.com/pytorch/fairseq) for pre-training. The pre-training config was almost same as [Librispeech 960](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/config/pretraining/wav2vec2_base_librispeech.yaml).

## Fine-tuning setup & results

- We fine-tuned the Wav2Vec2.0 ASR system with ESPNET toolkit with a randomly initialized linear layer and CTC loss on AISHELL-1. 
- The configurations were keep unchanged among all experiments, mainly followed *Deng et al.* [1, 2], trained for 20 epochs and averaged the model weights from the last 10 epochs on a single Nvidia Titan RTX. 
- We fine-tuned the model from checkpoints at different updates during pre-training and from official Wav2vec2.0(en). We also list the results using Mandarin wav2vec2.0 from previous literature which we want to reproduce.
- The results shows that even with only 85k updates, the Mandarin Wav2Vec2.0 already surpassed English wav2vec2.0 and it improved significantly after full pre-training.
- It's worth mentioning that we found the fine-tuned models were sensitive to learning rate. The loss curves in some models were not always going down. It shows that it is not necessarily that the model has the lowest CER is the "best" pre-trained model. We will try to find better configuration in the future.


| AISHELL-1                           | dev | test |
|-------------------------------------|-----|------|
| @50k updates                        | 6.7 | 7.5  |
| @85k updates                        | 6.1 | 6.7  |
| @130k updates                       | 5.8 | 6.4  |
| @250k updates                       | 5.6 | 6.3  |
| @315k updates                        | 5.1 | 5.6  |
| @376k updates                       | 5.2 | 5.7  |
| @396k updates*                      | 5.2 | 5.8  |
| Wav2vec2.0(English)                 | 6.2 | 6.7  |
| Wav2vec2.0(Mandarin) Deng et al.[1] | 5.1 | 5.6  |
| Wav2vec2.0(Mandarin) Deng et al.[2] | 4.8 | 5.3  |

\*lowest pre-training valid loss


## Reference
[1]: Deng, K., Cao, S., Zhang, Y., Ma, L., Cheng, G., Xu, J., & Zhang, P. (2022). Improving CTC-based speech recognition via knowledge transferring from pre-trained language models. arXiv preprint arXiv:2203.03582.

[2]: Deng, K., Yang, Z., Watanabe, S., Higuchi, Y., Cheng, G., & Zhang, P. (2022). Improving non-autoregressive end-to-end speech recognition with pre-trained acoustic and language models. arXiv preprint arXiv:2201.10103.

## Acknowledgement

We want to thank Keqi Deng and Songjun Cao for sharing valuable experiences with us.

