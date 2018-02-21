## Reasoning about Entailment with Neural Attention

The folders contain *TensorFlow* implementations of models that take in \{sentences, sequences, structures\} to predict entailment.

### Folders

Model_1 contains a TensorFlow implementation described in *Rocktäschel et al.* *"Reasoning about Entailment with Neural Attention"* [1]. 

Model_1 contains a TensorFlow implementation modified to abstract over "ambigous words".

### Data

#### The Stanford Natural Language Inference (SNLI) Corpus

The SNLI dataset by *Samuel R. Bowman et al.* [4]:

http://nlp.stanford.edu/projects/snli/snli_1.0.zip

#### Word Vectors

Experimenting with Glove, w2v, ConceptNet, non-static random.

### Instructions

The main script come with several options, which can be listed with the `--help` flag.
```bash
python main.py --help
```

To run the training:
```bash
python main.py --train
```
By default, the script runs on GPU 0 with these parameters values:
```python
learning_rate = 0.001
weight_decay = 0.
batch_size_train = 24
num_epochs = 45
sequence_length = 20
embedding_dim = 300
num_units = 100

```

### References

[1] Tim Rocktäschel, Edward Grefenstette, Karl Moritz Hermann, Tomáš Kočiský, Phil Blunsom, [Reasoning about Entailment with Neural Attention](https://arxiv.org/abs/1509.06664), 2015.

[2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean, [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546), 2013.

[3] Google, [Large-Scale Machine Learning on Heterogeneous Systems](http://tensorflow.org/), 2015.

[4] Samuel R. Bowman, Gabor Angeli, Christopher Potts, Christopher D. Manning, The Stanford Natural Language Processing Group, [A large annotated corpus for learning natural language inference](http://nlp.stanford.edu/projects/snli/),  2015.
