# Copyright 2015 Conchylicultor. All Rights Reserved.
# Author : Lei Sha

from model import Model
from textdata import TextData
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

class Chatbot:
    def __init__(self):
        self.args = self.predefined_args()

    def predefined_args(self):
        args = {}
        args['test'] = None
        args['createDataset'] = True
        args['playDataset'] = 10
        args['reset'] = True
        args['device'] ='gpu'
        args['rootDir'] = '/home/v-leisha/DeepQA_pytorch/'
        args['watsonMode'] = False
        args['autoEncode'] = False

        args['corpus'] = 'cornell'
        args['datasetTag'] = ''
        args['ratioDataset'] = 1.0
        args['maxLength'] = 50
        args['filterVocab'] = 1
        args['skipLines'] = True
        args['vocabularySize'] = 40000

        args['hiddenSize'] = 200
        args['numLayers'] = 2
        args['softmaxSamples'] = 0
        args['initEmbeddings'] = True
        args['embeddingSize'] = 120
        args['embeddingSource'] = "GoogleNews-vectors-negative300.bin"

        args['numEpochs'] = 30
        args['saveEvery'] = 2000
        args['batchSize'] = 256
        args['learningRate'] = 0.002
        args['dropout'] = 0.9

        args['encunit'] = 'lstm'
        args['decunit'] = 'lstm'
        args['enc_numlayer'] = 2
        args['dec_numlayer'] = 2
        

        args['maxLengthEnco'] = args['maxLength']
        args['maxLengthDeco'] = args['maxLength'] + 2

        return args

    def main(self):
        self.textData = TextData(self.args)
        self.args['vocabularySize'] = self.textData.getVocabularySize()
        print(self.textData.getVocabularySize())
        self.model = Model(self.args)

        self.train()

    def train(self,  print_every=10, plot_every=10, learning_rate=0.01):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        # criterion = nn.NLLLoss(size_average=False)
        iter = 1
        batches = self.textData.getBatches()
        n_iters = len(batches)
        for batch in batches:
            print(iter)

            # batchsize = batch.shape[0]
            x={}
            x['enc_input'] = autograd.Variable(torch.LongTensor(batch.encoderSeqs))
            x['enc_len'] = batch.encoder_lens
            x['dec_input'] = autograd.Variable(torch.LongTensor(batch.decoderSeqs))
            x['dec_len'] = batch.decoder_lens
            x['dec_target'] = autograd.Variable(torch.LongTensor(batch.targetSeqs))

            predictions = self.model(x)    # batch seq_len outsize

            target_variable = x['dec_target']  #batch seq_lenyou

            # targetlen = target_variable.shape[1]
            # print(predictions.size(), target_variable.size())

            loss = - torch.gather(predictions, 2, torch.unsqueeze(target_variable, 2))
            mask = torch.sign(target_variable.float())
            loss = loss * mask

            loss_mean = torch.mean(loss)

            loss_mean.backward()

            optimizer.step()
            # print(type(loss_mean.data[0]))
            print_loss_total += loss_mean.data[0]
            plot_loss_total += loss_mean.data[0]

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            iter+=1

        showPlot(plot_losses)
