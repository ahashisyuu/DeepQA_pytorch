import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import datetime

class Model(nn.Module):
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, args):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(Model, self).__init__()
        print("Model creation...")

        self.args = args

        self.dtype = 'float32'

        # Placeholders
        # self.encoderInputs  = None
        # self.decoderInputs  = None  # Same that decoderTarget plus the <go>
        # self.decoderTargets = None

        self.embedding = nn.Embedding(self.args['vocabularySize'], self.args['embeddingSize'])

        if self.args['encunit'] == 'lstm':
            self.enc_unit = nn.LSTM(input_size = self.args['embeddingSize'], hidden_size = self.args['hiddenSize'], num_layers = self.args['enc_numlayer'])
        elif self.args['encunit'] == 'gru':
            self.enc_unit = nn.GRU(input_size = self.args['embeddingSize'], hidden_size = self.args['hiddenSize'], num_layers = self.args['enc_numlayer'])

        if self.args['decunit'] == 'lstm':
            self.dec_unit = nn.LSTM(input_size=self.args['embeddingSize'], hidden_size=self.args['hiddenSize'],
                                    num_layers=self.args['dec_numlayer'])
        elif self.args['decunit'] == 'gru':
            self.dec_unit = nn.GRU(input_size=self.args['embeddingSize'], hidden_size=self.args['hiddenSize'],
                                   num_layers=self.args['dec_numlayer'])

        self.out_unit = nn.Linear(self.args['hiddenSize'], self.args['vocabularySize'])
        self.softmax = nn.LogSoftmax()

        self.enc_unit = self.enc_unit.cuda()
        self.dec_unit = self.dec_unit.cuda()
        # self.out_unit = self.out_unit.cuda()
        # self.softmax = self.softmax.cuda()


    def forward(self, x):
        '''
        :param encoderInputs: [batch, enc_len]
        :param decoderInputs: [batch, dec_len]
        :param decoderTargets: [batch, dec_len]
        :return:
        '''

        # print(x['enc_input'])
        self.encoderInputs = x['enc_input']
        self.encoder_lengths = x['enc_len']
        self.decoderInputs = x['dec_input']
        self.decoder_lengths = x['dec_len']
        self.decoderTargets = x['dec_target']

        self.batch_size = self.encoderInputs.size()[0]
        self.enc_len = self.encoderInputs.size()[1]
        self.dec_len = self.decoderInputs.size()[1]

        enc_input_embed = self.embedding(self.encoderInputs).cuda()   # batch enc_len embedsize  ; already sorted in a decreasing order
        dec_input_embed = self.embedding(self.decoderInputs).cuda()   # batch dec_len embedsize
        # dec_target_embed = self.embedding(self.decoderTargets).cuda()   # batch dec_len embedsize

        en_outputs, en_state = self.encoder(enc_input_embed, self.encoder_lengths)
        de_outputs, de_state = self.decoder(en_state, dec_input_embed, self.decoder_lengths)
        return de_outputs

    def encoder(self, inputs, input_len):
        inputs = torch.transpose(inputs, 0,1)
        hidden = (autograd.Variable(torch.randn(self.args['enc_numlayer'], self.args['batchSize'], self.args['hiddenSize'])).cuda(),
                  autograd.Variable(torch.randn(self.args['enc_numlayer'], self.args['batchSize'], self.args['hiddenSize'])).cuda())

        packed_input = nn.utils.rnn.pack_padded_sequence(inputs, input_len)
        packed_out, hidden = self.enc_unit(packed_input, hidden)
        return packed_out, hidden

    def decoder(self, initial_state, inputs, inputs_len ):
        inputs = torch.transpose(inputs, 0,1)
        state = initial_state

        output, out_state = self.dec_unit(inputs, state)
        output = output.cpu()

        output = self.softmax(self.out_unit(output.view(self.batch_size * self.dec_len, self.args['hiddenSize'])))
        output = output.view(self.dec_len, self.batch_size, self.args['vocabularySize'])
        output = torch.transpose(output, 0,1)
        return output, out_state




