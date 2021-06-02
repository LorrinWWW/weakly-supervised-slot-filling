# README

Code associated with the submission at AAAI2021.

*The code is still under construction, do not hesitate to ask if you have any questions or have trouble running the code.*

```bibtex
@inproceedings{wang2021effective,
  title={Effective Slot Filling via Weakly-Supervised Dual-Model Learning},
  author={Wang, Jue and Chen, Ke and Shou, Lidan and Wu, Sai and Chen, Gang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={16},
  pages={13952--13960},
  year={2021}
}
```

## Dependencies

- python3
- pytorch 1.4.0
- transformers 2.9.1
- flair 0.4.5
- gpustat

## Runing Experiments

1. Acquire data

    Since all corpora used are publicly available, we already put the preprocessed datasets in "./datasets/unified/".
    
    If you want to run your custom code, you can check data files in "./datasets/unified/".

2. Generate embeddings

   Then we prepare the pretrained word embeddings, such as GloVe. Available at https://nlp.stanford.edu/projects/glove/.

   Each line of this file represents a token or word. Here is an example with a vector of length 5:

   ```
   word 0.002 1.9999 4.323 4.1231 -1.2323
   ```

   (Optional) It is also recommended to use language model based contextualized embeddings, such as BERT. Check "./run/gen_XXX_emb.py" to generate them.

3. Start training
   
   Run the following cmd to start the training, e.g., on SNIPS with 300 utteraces retrained as labeled data.

   ```bash
   $ python train_slot_filling.py \
           --batch_size 30 \
           --evaluate_interval 500 \
           --dataset SNIPS_300 \
           --pretrained_wv ../wv/PATH_TO_WV_FILE  \
           --max_steps 50000 \
           --model_class WeakDualModel  \
           --model_write_ckpt ./PATH_TO_CKPT_TO_WRITE \
           --optimizer adam \
           --lr 1e-3 \
           --tag_form iob2  \
           --cased 0 \
           --token_emb_dim 100 \
           --char_emb_dim 30 \
           --char_encoder lstm \
           --lm_emb_dim 1024 \
           --lm_emb_path ../wv/PATH_TO_LM_EMB_PICKLE_OBJECT \
           --tag_vocab_size 79 \
           --vocab_size 20000 \
           --dropout 0.3 \
           --ratio_supervised 1 \
           --ratio_weak_supervised 0.2
   ```
   
   where "tag_vocab_size" should to set to "the number of slot types"\*2+1.
   For SNIPS, it should be 79=38\*2+1;
   for ATIS, it should be 167=83\*2+1;
   for MITRest, it should be 17=8\*2+1.
   "ratio_weak_supervised" is the weight of weakly-supervised learning loss ($L^{weak}$), set to 0 to disable it. 
   

## Arguments

### --batch_size

The batch size to use.

### --evaluate_interval

The evaluation interval, which means evaluate the model for every {evaluate_interval} training steps.

### --dataset

The name of dataset to be used. The dataset should be unified in json and placed in "./datasets/unified/", i.e., 
"train.{dataset}.json", 
"train.weak.{dataset}.json",
"valid.{dataset}.json", 
and "test.{dataset}.json" three files.

"train.{dataset}.json" contain fully labeled user utterances for supervised learning.
"valid.{dataset}.json" and "test.{dataset}.json" are dev set and test set.
All of them are in the same format, i.e., 
each json file consists of a list of items, and each item is similar to the following example:
```json
{
  "tokens": ["hello", "world", "!"],
  "tags": ["O", "B-object", "O"],
  "chunks_tags": ["O", "B-M", "O"],
  "mentions": [
      ["object", [1,2]],
  ],
  "chunks_list": [
      ["object", ["O", "B-M", "O"]],
  ],
  "label": "intent_label",
}
```
where "tags", "mentions", and "chunks_list" are equivalent and can be converted to each other;
"label" the intent label is *not* used in our method, and removing it will not make any difference.

"train.weak.{dataset}.json" are used in weakly-supervised learning part.
It consists of a list of items, and each item is similar to:
```json
{
  "tokens": ["hello", "world", "!"],
  "chunk": ["O", "B-M", "O"],
}
```
Each item represents a text chunk.

*Note: We have prepared datasets from different corpora (SNIPS, ATIS, MITRest) and with different number of labeled utterances (100, 300, 1000, "all"). So simply pass the "{corpus}\_{n\_uttr}" to this argument, such as "SNIPS\_300", "ATIS\_100", etc.*


### --pretrained_wv

The pretrained word vectors file, such as GloVe.

Each line of this file represents a token or word. Here is an example with a vector of length 5:

```
word 0.002 1.9999 4.323 4.1231 -1.2323
```

### --max_steps

max_steps.
50000 is a recommended value.

### --max_epoches

max_epoches.
One should at least specify either "max_steps" or "max_epoches".

### --model_class

model_class, should be **WeakDualModel**.

### --model_write_ckpt

Path of model_write_ckpt. None if you don't want to save checkpoints.

### --optimizer

Optimizer to use. "adam" or "sgd".

### --lr

Learning rate. E.g. 1e-3.

### --tag_form

tag_form. Currently only support IOB2 ("iob2").

### --cased

Whether cased for word embeddings (0 or 1). Note for char embs, it is always cased.
Since SLU datasets are usually lower cased, the default value is 0.

### --token_emb_dim

Word embedding dimension. This should be in line with "pretrained_wv" file.
E.g., "--token_emb_dim=100" for GloVe.100d.6B.

### --char_emb_dim

Character embedding dimension. 0 to disable it.
30 works fine.

### --char_encoder

Use "lstm" or "cnn" char encoder. Default to be "lstm".

### --lm_emb_dim

Language model based embedding dimension. 0 to disable it.
E.g., "--lm_emb_dim=1024" for "bert-large-uncased".

### --lm_emb_path

Language model embeddings. "lm_emb_path" is required if "lm_emb_dim" > 0.

which is a pickle file, representing a dictionary object, mapping a tuple of tokens to a numpy matrix:

```python
{
  (t0_0,t0_1,t0_2,...,t0_23): np.array([24, 1024]),
  (t1_0,t1_1,t1_2,...,t1_16): np.array([17, 1024]),
  ...
}
```

check "./run/gen_XXX_emb.py" to know how to generate the language model embeddings.

### --tag_vocab_size

The tag vocab size. A value equals to the possible number of IOB2 tags.

### --vocab_size

Maximum of token vocab size. A value bigger than the possible number of words.
E.g., GloVe.100d.6B contains 400,000 word vectors, so "--vocab_size=400100" is a valid value.

To reduce the vocab size for faster training, please check "./run/reduce_wv.py".

### --dropout

dropout rate

### --freeze_wv

Whether the word embeddings are freezed without updating.
0 or 1, default to be 0.

### --device

Assign a device to run the model. 
By default, the script will select the GPU with the largest available memory.


## Model

The model is defined in "./models/weak.py".

Feel free to modify and test it.