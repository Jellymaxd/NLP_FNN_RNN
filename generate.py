###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model
#
###############################################################################
# coding: utf-8
import argparse

import torch
import model
import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")


"""
with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()"""

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
share=False
#load model
try:
    temp=input("1 to use model with sharing on 2 for without sharing")
    if temp=='1':
        share=True
except KeyboardInterrupt:
    print("exiting...")

gpu=0
best_model_path="best_model_"+("sharing" if share else "nosharing")+".dat"
best_model = model.FNNModel(ntokens, 200, 400, 7, share).to(device)
best_model.load_state_dict(torch.load(best_model_path))
#best_model.cuda(gpu)

input = torch.randint(ntokens, (1, 7), dtype=torch.long).to(device)

print('generating...')
with open("generated "+("sharing" if share else "nosharing")+".txt", 'w',encoding='utf-8') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            probs = best_model(input)
            word_weights = probs.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]
            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
