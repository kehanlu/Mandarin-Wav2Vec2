# Mandarin-Wav2Vec2

In this repository, we show the Mandarin version of wav2vec2.0(base) pre-trained on AISHELL-2 dataset. This is also the pre-trained model used in “A context-aware knowledge transferring strategy for CTC-based ASR”


## Pre-trained model
The model is pre-trained using Fairseq toolkit on the AISHELL-2 dataset and evaluate on ESPNET toolkit on the AISHELL-1 dataset.

| model | fairseq ckpt | huggingface |
| ----- | -------- | ----------- |
| mandarin-wav2vec2 | [download](https://huggingface.co/kehanlu/mandarin-wav2vec2-fairseq/resolve/main/mandarin_wav2vec2_fairseq.pt) | kehanlu/mandarin-wav2vec2 |

Result: 

| AISHELL-1 | dev | test |
| - | - | - | 
| vanilla w2v2-CTC | 4.85 | 5.13 |

### Use with ESPNET

ESPNET2  fine-tuning config: [config/finetine_aishell1.yaml](https://github.com/kehanlu/Mandarin-Wav2Vec2/blob/main/config/finetine_aishell1.yaml)
- lr=0.0001
- epochs=20


### Use with Huggingface
For simple usage, we convert the fairseq/espnet checkpoint into huggingface Transformers version by `transformers.models.wav2vec2.convert_wav2vec2_original_pytorch_checkpoint_to_pytorch`

#### Pre-trained model

```python
from transformers import Wav2Vec2Model, Wav2Vec2Processor

model = Wav2Vec2Model.from_pretrained("kehanlu/mandarin-wav2vec2")
processor = Wav2Vec2Processor.from_pretrained("kehanlu/mandarin-wav2vec2")

```

#### Fine-tuned CTC model
```python
import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

class ExtendedWav2Vec2ForCTC(Wav2Vec2ForCTC):
    """
    In ESPNET there is a LayerNorm layer between encoder output and CTC classification head.
    """
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = torch.nn.Sequential(
                torch.nn.LayerNorm(config.hidden_size),
                self.lm_head
        )
        
model = ExtendedWav2Vec2ForCTC.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")
processor = Wav2Vec2Processor.from_pretrained("kehanlu/mandarin-wav2vec2-aishell1")

audio_input, sample_rate = sf.read("/path/to/data_aishell/wav/dev/S0724/BAC009S0724W0121.wav")
inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")

with torch.no_grad():
    model.eval()
    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
print(transcription[0])

# 广州市房地产中介协会分析
```

## Pre-training setup: 

- We filtered the wav files that contained the same transcriptions in the AISHELL-1 dev/test set to avoid information leaks for the research purpose. The filtered pre-training set has ~960hr of raw speeches, and we randomly sampled 1% of data for validation.
- We concatenated 6 wav files without shuffling to one file to accelerate the pre-training process and make the audio length close to the size of Librispeech samples.
- We used 8x Nvidia Tesla V100 and accumulated 8 updates to simulate 64 GPUs. The total training time is about 10 days.
- We used Fairseq(https://github.com/pytorch/fairseq) for pre-training. The pre-training config was almost the same as [Librispeech 960](https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/config/pretraining/wav2vec2_base_librispeech.yaml).

### Intermediate results

The following table shows the fine-tuned results at different pre-training updates. The models are trained with CTC loss for 20 epochs and lr=0.0002 ([config/finetine_aishell1.yaml](https://github.com/kehanlu/Mandarin-Wav2Vec2/blob/main/config/finetine_aishell1.yaml)). The released pre-trained model  is the one with lowest pre-training validation loss. We also list the results fine-tuned with wav2vec2.0(en) and from other paper.

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

## Licence
The pre-trained corpus, AISHELL-2, is supported by AISHELL fundation. The outcome models also follow the licence of AISHELL-2. It is free to use for academic purpose and **should not** be used on any commercial purpose without the permission from AISHELL fundation. (https://www.aishelltech.com/aishell_2)

```
@ARTICLE{aishell2,
   author = {{Du}, J. and {Na}, X. and {Liu}, X. and {Bu}, H.},
   title = "{AISHELL-2: Transforming Mandarin ASR Research Into Industrial Scale}",
   journal = {ArXiv},
   eprint = {1808.10583},
   primaryClass = "cs.CL",
   year = 2018,
   month = Aug,
}
```

If you find our work useful, please cite
```
@article{lu2022context,
  title={A context-aware knowledge transferring strategy for CTC-based ASR},
  author={Lu, Ke-Han and Chen, Kuan-Yu},
  journal={arXiv preprint arXiv:2210.06244},
  year={2022}
}
```

## Reference
[1]: K. Deng et al., "Improving CTC-Based Speech Recognition Via Knowledge Transferring from Pre-Trained Language Models," ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 8517-8521, doi: 10.1109/ICASSP43922.2022.9747887.

[2]: K. Deng, Z. Yang, S. Watanabe, Y. Higuchi, G. Cheng and P. Zhang, "Improving Non-Autoregressive End-to-End Speech Recognition with Pre-Trained Acoustic and Language Models," ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 8522-8526, doi: 10.1109/ICASSP43922.2022.9746316.

## Acknowledgement

We want to thank Keqi Deng and Songjun Cao for sharing valuable experiences with us.
