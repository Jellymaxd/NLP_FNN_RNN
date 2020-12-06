***Question 1:***
========================================================================
data.py: data preprocessing and corpus generation
========================================================================
model.py: implementation of the FNN models
========================================================================
main.py: training and prediction usinf the FNN models

To run the training, specify the path of the wikitext-2 data by running 
main.py --data=path in any IDE or use bash python main.py --data=path

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           number of hidden units per layer
  --lr LR               initial learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --dropinput DROPOUT   dropout applied to input layers (0 = no dropout)
  --drophidden DROPOUT  dropout applied to hidden layers (0 = no dropout)
  --seed SEED           random seed
  --cuda                use CUDA
  --log-interval N      report interval
  --save SAVE           path to save the final model
  --ngram		ngram size
  --SGD variant         SGD variant used for the optimizer
  --sharing             whether to turn on sharing between input and output embeddings
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40           
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --sharing   
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40        
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40  

```

best models will be saved in the same folder after training
========================================================================

generate.py: generate texts using the saved model from running main.py. 
Options are provided to use the model with sharing or without sharing 
for text generation (through user input).
generated files are saved as "generated sharing.txt"/"generated nosharing.txt"

***Question 2:***
========================================================================
original.py: unmodified script
========================================================================
main.py: training using CNN layers for word-level encoder

1) Download the codebase with CoNLL NER dataset from https://github.com/jayavardhanr/End-to-end-Sequence-Labeling-via-Bi-directional-LSTM-CNNs-CRF-Tutorial
2) Download GloVe vectors and extract glove.6B.100d.txt into "./data/" folder from http://nlp.stanford.edu/data/glove.6B.zip
3) Add main.py inside the downloaded folder
4) Parameters can be edited inside main.py script
num_cnn_layers: number of CNN layers to use for word-level encoder (between 1-5)
char_dim: character embedding dimension
word_dim: token embedding dimension
word_lstm_dim: token hidden layer size
crf: enable or disable crf
dropout: dropout in the input
epoch: number of epoch to run

5) To train, set parameters['reload'] = False located right after function adjust_learning_rate(optimizer, lr). Then run the script.
6) Plot and results will be generated after training


