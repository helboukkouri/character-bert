# CharacterBERT

[paper]: https://aclanthology.org/2020.coling-main.609/

This is the code repository for the paper "[CharacterBERT: Reconciling ELMo and BERT for Word-LevelOpen-Vocabulary Representations From Characters][paper]" that came out at COLING 2020.

> 2021-02-25: Code for pre-training BERT and CharacterBERT is now available [here](https://github.com/helboukkouri/character-bert-pretraining)!

## Table of contents

- [Paper summary](#paper-summary)
  - [TL;DR](#tldr)
  - [Motivations](#motivations)
- [How do I use CharacterBERT?](#how-do-i-use-characterbert)
  - [Installation](#installation)
  - [Pre-trained models](#pre-trained-models)
  - [Using CharacterBERT in practice](#using-characterbert-in-practice)
- [How do I pre-train CharacterBERT?](#how-do-i-pre-train-characterbert)
- [How do I reproduce the paper's results?](#how-do-i-reproduce-the-papers-results)
- [Running experiments on GPUs](#running-experiments-on-gpus)
- [References](#references)

## Paper summary

### TL;DR

`CharacterBERT` is a variant of [BERT](https://arxiv.org/abs/1810.04805) that produces **contextual representations** at the **word level**.

This is achieved by attending to the characters of each input token and dynamically building token representations from that. In fact, contrary to standard `BERT`--which relies on a matrix of pre-defined wordpieces, `this approach` uses a [CharacterCNN](./img/archi-compare.png) module, similar to [ELMo](https://arxiv.org/abs/1802.05365), that can generate representations for arbitrary input tokens.

<div style="text-align:center">
    <br>
    <img src="./img/archi-compare.png" width="45%"/>
</div>

The figure above shows how context-independent representations are built in `BERT`, vs. how they are built in `CharacterBERT`. Here, we assume that "**Apple**" is an unknown token, which results in `BERT` splitting the token into two wordpieces "**Ap**" and "**##ple**" and embedding each unit. On the other hand, `CharacterBERT` processes the token "*Apple*" as is, then attends to its characters in order to produce a `single token embedding`.

## Motivations

CharacterBERT has two main motivations:

- It is frequent to adapt the original `BERT` to new specialized domains (e.g. **medical**, **legal**, **scientific** domains..) by simply re-training it on a set of specialized corpora. This results in the original (**general domain**) wordpiece vocabulary being re-used despite the final model being actually targeted toward a different potentially highly specialized domain, which is arguably suboptimal (see _Section 2_ of the paper).

  A straightforward solution in this case would be to train a **new BERT from scratch** with a **specialized wordpiece vocabulary**. However, training a single BERT is already costly enough let alone needing to train one for each and every domain of interest.

- `BERT` uses a wordpiece system to strike a good balance between the **specificity** of tokens and **flexibility** of characters. However, working with subwords is not the most convenient in practice (should we average the representations to get the original token embedding for word similarity tasks? should we only use the first wordpiece of each token in sequence labelling tasks? ...) and most would just prefer to work with tokens.

Inspired by `ELMo`, we use a `CharacterCNN module` and manage to get a variant of BERT that produces both **word-level** and **contextual representations** which can also be re-adapted **as many times as necessary**, on **any domain**, without needing to worry about the suitability of any wordpieces. And as a cherry on top, attending to the characters of each input token also leads us to a **more robust model** against any **typos** or **misspellings** (see _Section 5.5_ of the paper).

## How do I use CharacterBERT?

### Installation

We recommend using a virtual environment that is specific to using `CharacterBERT`.

If you do not already have `conda` installed, you can install Miniconda from [this link](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html). Then, check that your conda is up to date:

```bash
conda update -n base -c defaults conda
```

Create a fresh conda environment:

```bash
conda create python=3.10 --name=character-bert
```

If not already activated, activate the new conda environment using:

```bash
conda activate character-bert
```

Then install the following packages:

```bash
conda install pytorch cudatoolkit=11.8 -c pytorch
pip install transformers==4.34.0 scikit-learn==1.3.1 gdown==4.7.1
```

> Note 1: If you will not be running experiments on a GPU, install pyTorch via this command instead:<br> `conda install pytorch cpuonly -c pytorch`

> Note 2: If you just want to be able to load pre-trained CharacterBERT weigths, you do not have to install `scikit-learn` which is only used for computing Precision, Recall, F1 metrics during evaluation.

### Pre-trained models

You can use the `download.py` script to download any of the models below:

| Keyword                | Model description                                                                                                                                                                                                                                                         |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| general_character_bert | General Domain CharacterBERT pre-trained from scratch on English Wikipedia and [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/).                                                                                                                            |
| medical_character_bert | Medical Domain CharacterBERT initialized from **general_character_bert** then further pre-trained on [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) clinical notes and [PMC OA](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/) biomedical paper abstracts. |
| general_bert           | General Domain BERT pre-trained from scratch on English Wikipedia and [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/). <sup>1</sup>                                                                                                                        |
| medical_bert           | Medical Domain BERT initialized from **general_bert** then further pre-trained on [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) clinical notes and [PMC OA](https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/) biomedical paper abstracts. <sup>2</sup>       |
| bert-base-uncased      | The original General Domain [BERT (base, uncased)](https://github.com/google-research/bert#pre-trained-models)                                                                                                                                                                                                                          |

> <sup>1, 2</sup> <small>We offer BERT models as well as CharacterBERT models since we have pre-trained both architectures in an effort to fairly compare these architectures. Our BERT models use the same architecture and starting wordpiece vocabulary as `bert-base-uncased`.</small><br>

For instance, let's download the medical version of CharacterBERT:

```bash
python download.py --model='medical_character_bert'
```

We can download also download all models in a single command:

```bash
python download.py --model='all'
```

### Using CharacterBERT in practice

CharacterBERT's architecture is almost identical to BERT's, so you can easilly adapt any code that from the [Transformers](https://github.com/huggingface/transformers) library.

#### Example 1: getting word embeddings from CharacterBERT

```python
"""Basic example: getting word embeddings from CharacterBERT"""
from transformers import BertTokenizer
from modeling.character_bert import CharacterBertModel
from utils.character_cnn import CharacterIndexer

# Example text
x = "Hello World!"

# Tokenize the text
tokenizer = BertTokenizer.from_pretrained(
    './pretrained-models/bert-base-uncased/')
x = tokenizer.basic_tokenizer.tokenize(x)

# Add [CLS] and [SEP]
x = ['[CLS]', *x, '[SEP]']

# Convert token sequence into character indices
indexer = CharacterIndexer()
batch = [x]  # This is a batch with a single token sequence x
batch_ids = indexer.as_padded_tensor(batch)

# Load some pre-trained CharacterBERT
model = CharacterBertModel.from_pretrained(
    './pretrained-models/medical_character_bert/')

# Feed batch to CharacterBERT & get the embeddings
embeddings_for_batch, _ = model(batch_ids)
embeddings_for_x = embeddings_for_batch[0]
print('These are the embeddings produces by CharacterBERT (last transformer layer)')
for token, embedding in zip(x, embeddings_for_x):
    print(token, embedding)
```

#### Example 2: using CharacterBERT for binary classification 

```python
""" Basic example: using CharacterBERT for binary classification """
from transformers import BertForSequenceClassification, BertConfig
from modeling.character_bert import CharacterBertModel

#### LOADING BERT FOR CLASSIFICATION ####

config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)  # binary classification
model = BertForSequenceClassification(config=config)

model.bert.embeddings.word_embeddings  # wordpiece embeddings
>>> Embedding(30522, 768, padding_idx=0)

#### REPLACING BERT WITH CHARACTER_BERT ####

character_bert_model = CharacterBertModel.from_pretrained(
    './pretrained-models/medical_character_bert/')
model.bert = character_bert_model

model.bert.embeddings.word_embeddings  # wordpieces are replaced with a CharacterCNN
>>> CharacterCNN(
        (char_conv_0): Conv1d(16, 32, kernel_size=(1,), stride=(1,))
        (char_conv_1): Conv1d(16, 32, kernel_size=(2,), stride=(1,))
        (char_conv_2): Conv1d(16, 64, kernel_size=(3,), stride=(1,))
        (char_conv_3): Conv1d(16, 128, kernel_size=(4,), stride=(1,))
        (char_conv_4): Conv1d(16, 256, kernel_size=(5,), stride=(1,))
        (char_conv_5): Conv1d(16, 512, kernel_size=(6,), stride=(1,))
        (char_conv_6): Conv1d(16, 1024, kernel_size=(7,), stride=(1,))
        (_highways): Highway(
        (_layers): ModuleList(
            (0): Linear(in_features=2048, out_features=4096, bias=True)
            (1): Linear(in_features=2048, out_features=4096, bias=True)
        )
        )
        (_projection): Linear(in_features=2048, out_features=768, bias=True)
    )

#### PREPARING RAW TEXT ####

from transformers import BertTokenizer
from utils.character_cnn import CharacterIndexer

text = "CharacterBERT attends to each token's characters"
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = bert_tokenizer.basic_tokenizer.tokenize(text) # this is NOT wordpiece tokenization

tokenized_text
>>> ['characterbert', 'attends', 'to', 'each', 'token', "'", 's', 'characters']

indexer = CharacterIndexer()  # This converts each token into a list of character indices
input_tensor = indexer.as_padded_tensor([tokenized_text])  # we build a batch of only one sequence
input_tensor.shape
>>> torch.Size([1, 8, 50])  # (batch_size, sequence_length, character_embedding_dim)

#### USING CHARACTER_BERT FOR INFERENCE ####

output = model(input_tensor, return_dict=False)[0]
>>> tensor([[-0.3378, -0.2772]], grad_fn=<AddmmBackward>)  # class logits
```

For more complete (but still illustrative) examples you can refer to the `run_experiments.sh` script which runs a few Classification/SequenceLabelling experiments using BERT/CharacterBERT.

```bash
bash run_experiments.sh
```

You can adapt the `run_experiments.sh` script to try out any available model. You should also be able to add real classification and sequence labelling tasks by adapting the `data.py` script.

## Running experiments on GPUs

In order to use GPUs you will need to make sure the PyTorch version that is in your conda environment matches your machine's configuration. To do that, you may want to run a few tests.

Let's assume you want to use the GPU n°0 on your machine. Then set:

```bash
export CUDA_VISIBLE_DEVICES=0
```

And run these commands to check whether pytorch can detect your GPU:

```python
import torch
print(torch.cuda.is_available())  # Should return `True`
```

If the last command returns `False`, then there is probably a mismatch between the installed PyTorch version and your machine's configuration. To fix that, run `nvidia-smi` in your terminal and check your driver version:

<center><img src="img/nvidiasmi.png" alt="drawing" width="550"/></center>

Then compare this version with the numbers given in the [NVIDIA CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html):

<center><img src="img/cudaversions.png" alt="drawing" width="800"/></center>

In this example the shown version is `390.116` which corresponds to `CUDA 9.0`. This means that the appropriate command for installing PyTorch is:

```bash
conda install pytorch cudatoolkit=9.0 -c pytorch
```

Now, everything should work fine!

## References

Please cite our paper if you use CharacterBERT in your work:

```
@inproceedings{el-boukkouri-etal-2020-characterbert,
    title = "{C}haracter{BERT}: Reconciling {ELM}o and {BERT} for Word-Level Open-Vocabulary Representations From Characters",
    author = "El Boukkouri, Hicham  and
      Ferret, Olivier  and
      Lavergne, Thomas  and
      Noji, Hiroshi  and
      Zweigenbaum, Pierre  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.609",
    doi = "10.18653/v1/2020.coling-main.609",
    pages = "6903--6915",
    abstract = "Due to the compelling improvements brought by BERT, many recent representation models adopted the Transformer architecture as their main building block, consequently inheriting the wordpiece tokenization system despite it not being intrinsically linked to the notion of Transformers. While this system is thought to achieve a good balance between the flexibility of characters and the efficiency of full words, using predefined wordpiece vocabularies from the general domain is not always suitable, especially when building models for specialized domains (e.g., the medical domain). Moreover, adopting a wordpiece tokenization shifts the focus from the word level to the subword level, making the models conceptually more complex and arguably less convenient in practice. For these reasons, we propose CharacterBERT, a new variant of BERT that drops the wordpiece system altogether and uses a Character-CNN module instead to represent entire words by consulting their characters. We show that this new model improves the performance of BERT on a variety of medical domain tasks while at the same time producing robust, word-level, and open-vocabulary representations.",
}
```
