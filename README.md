# pytorch-text-to-speech

This repo contains a pytorch-tracable version of the (**FastSpeech2**)[https://arxiv.org/abs/2006.04558v1] model.
Note that the original code-base is taken from [ming024](https://github.com/ming024/FastSpeech2) implementation. Thus, please download the original model according to its specification.

As this repo focus on tracing, the training code has been stripped by all the knowledge distillation code: so, it can't be directly used to train or finetune a new model.
Moreover, only the model trained on the single-speaker ``LJSpeech`` dataset has been tested. 


## Tracing
1. get the pretrained model from [ming024-drive](https://drive.google.com/file/d/1r3fYhnblBJ8hDKDSUDtidJ-BN-xAM9pe/view?usp=sharing)
2. get the vocoder model from [ming024-repo](https://github.com/ming024/FastSpeech2/tree/master/hifigan)
3. adjust the vocode parameter names by

```bash
pyhton src/models/hifigan.py
```

4. trace the model running:
```bash
python tracer.py
```

5. you can test the traced model running ``tracer_test.py`` modifing the ``raw_texts`` variable according to your preferences
