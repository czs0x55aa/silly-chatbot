# coding=utf8
import os
import sys
import json
import random
import torch
from model import Seq2Seq, Encoder, Decoder
from masked_cross_entropy import *

with open('config.json') as config_file:
    config = json.load(config_file)

CKPT_PATH = config['TRAIN']['PATH']
USE_CUDA = config['TRAIN']['CUDA']

batch_size = config['TRAIN']['BATCH_SIZE']

def model_evaluate(model, dataset, evaluate_num=10):
    model.train(False)
    total_loss = 0.0
    for _ in range(evaluate_num):
        input_group, target_group = dataset.random_test()
        all_decoder_outputs = model(input_group, target_group, teacher_forcing_ratio=1)
        target_var, target_lens = target_group
        loss = masked_cross_entropy(
            all_decoder_outputs.transpose(0, 1).contiguous(),
            target_var.transpose(0, 1).contiguous(),
            target_lens
        )
        total_loss += loss.data[0]
        format_output(dataset.vocabulary.index2word, input_group, target_group, all_decoder_outputs)
    model.train(True)
    return total_loss / evaluate_num

def format_output(index2word, input_group, target_group, all_decoder_outputs):
    input_var, input_lens = input_group
    target_var, target_lens = target_group
    all_decoder_outputs = all_decoder_outputs.transpose(0, 1)
    input_var = input_var.transpose(0, 1)
    target_var = target_var.transpose(0, 1)
    i = random.randint(0, batch_size-1)
    topv, topi = all_decoder_outputs[i].data.topk(1, dim=1)
    topi = topi.squeeze(1)
    input_length, target_length = input_lens[i], target_lens[i]
    if USE_CUDA:
        inp = input_var[i][:input_length].data.cpu().numpy()
        tar = target_var[i][:target_length].data.cpu().numpy()
        preidct = topi[:target_length].cpu().numpy()
    else:
        inp = input_var[i][:input_length].data.numpy()
        tar = target_var[i][:target_length].data.numpy()
        preidct = topi[:target_length].numpy()
    print('===========>')
    print('[input]:   ' + ' '.join([index2word[x] for x in inp]))
    print('[target]:  ' + ' '.join([index2word[x] for x in tar]))
    print('[preidct]: ' + ' '.join([index2word[x] for x in preidct]))

def build_model(vocab_size, load_ckpt=False, ckpt_epoch=-1):
    hidden_size = config['MODEL']['HIDDEN_SIZE']
    attn_method = config['MODEL']['ATTN_METHOD']
    n_encoder_layers = config['MODEL']['N_ENCODER_LAYERS']
    dropout = config['MODEL']['DROPOUT']
    encoder = Encoder(vocab_size, hidden_size, n_encoder_layers, dropout=dropout)
    decoder = Decoder(hidden_size, vocab_size, attn_method, dropout=dropout)
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        max_length=config['LOADER']['MAX_LENGTH'],
        tie_weights=config['MODEL']['TIE_WEIGHTS']
    )
    print(model)
    if load_ckpt is True and os.path.exists(CKPT_PATH) is True:
        # load checkpoint
        prefix = config['TRAIN']['PREFIX']
        model_path = None
        if ckpt_epoch >= 0:
            model_path = '%s%s_%d' % (CKPT_PATH, prefix, ckpt_epoch)
        else:
            # use last checkpoint
            ckpts = []
            for root, dirs, files in os.walk(CKPT_PATH):
                for fname in files:
                    fname_sp = fname.split('_')
                    if len(fname_sp) == 2:
                        ckpts.append(int(fname_sp[1]))
            if len(ckpts) > 0:
                model_path = '%s%s_%d' % (CKPT_PATH, prefix, max(ckpts))

        if model_path is not None and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print('Load %s' % model_path)

    # print('Seq2Seq parameters:')
    # for name, param in model.state_dict().items():
    #     print(name, param.size())
    if USE_CUDA:
        model = model.cuda()
    return model

def init_path():
    if os.path.exists(CKPT_PATH) is False:
        os.mkdir(CKPT_PATH)

def save_model(model, epoch):
    init_path()
    save_path = '%s%s_%d' % (CKPT_PATH, config['TRAIN']['PREFIX'], epoch)
    torch.save(model.state_dict(), save_path)

def save_vocabulary(vocabulary_list):
    init_path()
    with open(CKPT_PATH + config['TRAIN']['VOCABULARY'], 'w') as file:
        for word, index in vocabulary_list:
            file.write('%s %d\n' % (word, index))

if __name__ == '__main__':
    pass
