# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.onnx
import numpy as np
from torch.utils.data import DataLoader

import data
import model

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=400,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=2e-4,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--dropinput', type=float, default=0.2,
                    help='dropout applied to input layers (0 = no dropout)')
parser.add_argument('--drophidden', type=float, default=0.5,
                    help='dropout applied to hiddnen layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--ngram', type=int, default=8,
                    help='ngram size')
parser.add_argument('--SGDvariants', type=str, default='Adam',
                    help='SGD variants for optimizer')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--sharing', action='store_true',
                    help='sharing between embedding matrix C and output')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)
eval_batch_size = 10
###############################################################################


available_workers = 0
# data loaders
train_data = np.concatenate((corpus.train_x, corpus.train_y), axis=1)
valid_data = np.concatenate((corpus.valid_x, corpus.valid_y), axis=1)
test_data = np.concatenate((corpus.test_x, corpus.test_y), axis=1)
train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=available_workers)
val_loader = DataLoader(valid_data, batch_size=eval_batch_size, num_workers=available_workers)
test_loader = DataLoader(test_data, batch_size=eval_batch_size, num_workers=available_workers)

###############################################################################
# Build the model
###############################################################################

gpu = 0
ntokens = len(corpus.dictionary)

model = model.FNNModel(ntokens, args.emsize, args.nhid, args.ngram - 1, args.sharing, args.dropinput, args.drophidden).to(device)
criterion = nn.CrossEntropyLoss()


if args.SGDvariants != 'Adam':
    optimizer = torch.optim.rmsprop
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

###############################################################################
# Training code


# helper function to get accuracy from log probabilities
def get_accuracy(log_probs, labels):
    probs = torch.exp(log_probs)
    predicted_label = torch.argmax(probs, dim=1)
    acc = (predicted_label == labels).float().mean()
    return acc


def evaluate(model, criterion, dataloader, gpu):
    model.eval()

    total_loss = 0
    count = 0

    with torch.no_grad():
        start_time = time.time()
        for iter, input_tensor in enumerate(dataloader):
            context_tensor = input_tensor.type(torch.LongTensor)[:, 0:7]
            target_tensor = input_tensor.type(torch.LongTensor)[:, 7]
            context_tensor, target_tensor = context_tensor.cuda(gpu), target_tensor.cuda(gpu)
            probs = model(context_tensor.to(device))
            total_loss += criterion(probs, target_tensor).item()
            count += 1
            if iter % 500 == 0:
                print("Validation Iteration {} complete. Mean Loss: {}; Mean Perplexity:{}; Time taken (s): {}".format(
                    iter, total_loss / count, math.exp(total_loss / count), (time.time() - start_time)))
                start_time = time.time()

    return math.exp(total_loss / count), total_loss / count


def train():
    count = 0
    total_loss = 0

    for iter, data_tensor in enumerate(train_loader):
        context_tensor = data_tensor.type(torch.LongTensor)[:, 0:7]
        target_tensor = data_tensor.type(torch.LongTensor)[:, 7]

        context_tensor, target_tensor = context_tensor.cuda(gpu), target_tensor.cuda(gpu)

        # zero out the gradients from the old instance
        model.zero_grad()

        # get log probabilities over next words
        log_probs = model(context_tensor.to(device))

        # compute loss function
        loss = criterion(log_probs, target_tensor)

        # calculate perplexity
        perplexity = math.exp(loss.item())
        # calculate current accuracy
        accuracy = get_accuracy(log_probs, target_tensor)

        # backward pass and update gradient
        loss.backward()
        optimizer.step()

        # increment
        count += 1
        total_loss += loss.item()
        if iter % 500 == 0:
            print(
                '|Training iter {} completed | loss {:5.2f} | ppl {:8.2f} | acc {:5.2f} '.format(
                    iter, loss, perplexity, accuracy))

    return math.exp(total_loss / count), total_loss / count


best_model_path = None
best_perplexity = 10000000
# train the model
try:
    for epoch in range(args.epochs):
        start_time = time.time()
        print("\n--- Training model Epoch: {} ---".format(epoch + 1))
        train_perplexity, train_loss = train()
        print(
            "\n-- Training Epoch {} complete! Train Perplexity: {}, Train Loss: {}".format(epoch + 1, train_perplexity,
                                                                                           train_loss))
        print("\n--- Evaluating model on validation data ---")
        valid_perplexity, valid_loss = evaluate(model, criterion, val_loader, gpu)
        print("Epoch {} complete! Validation Perplexity: {}; Validation Loss: {} Time taken: {}".format(epoch + 1,
                                                                                                        valid_perplexity,
                                                                                                        valid_loss,
                                                                                                        time.time() - start_time))

        if valid_perplexity < best_perplexity:
            print("Validation perplexity reduced from {} to {}, saving model...".format(best_perplexity,
                                                                                        valid_perplexity))
            best_perplexity = valid_perplexity

            # set best model path
            best_model_path = 'best_model_'+('sharing' if args.sharing else 'nosharing')+'.dat'

            # saving best model
            torch.save(model.state_dict(), best_model_path)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
print("using best model with validation perplexity {}".format(best_perplexity))
best_model = model.FNNModel(ntokens, args.emsize, args.nhid, args.ngram - 1, args.sharing, args.dropinput, args.drophidden).to(device)
best_model.load_state_dict(torch.load(best_model_path))
best_model.cuda(gpu)

# Run on test data.
test_perplexity, test_loss = evaluate(best_model, criterion, test_loader, gpu)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test perplexity {:5.2f}'.format(
    test_loss, test_perplexity))
print('=' * 89)

