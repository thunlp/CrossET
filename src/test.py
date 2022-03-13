import torch
import json

from time import time
from tqdm import tqdm
from random import shuffle, randint
from transformers import BertTokenizer, BertModel

from source.model import noname
from source.parser import parse
from source.sampler import sampler
from source.count import counter
from source.function import *

def test_contrastive(model, data, types, tokenizer):
    model.eval()

    avg_loss = 0
    cnt = 0

    my_sampler = sampler(data, types, batch_size = 32)

    print('begin test')
    for samp in my_sampler.get_batches():
#try:
        if True:

            model_input, mask_tensor, ans, pos = get_input(samp, tokenizer, types)
            model_input = model_input.to('cuda:0')
            mask_tensor = mask_tensor.to('cuda:0')
            sim = get_sim(samp).to('cuda:0')

            model_output,embeddings = model(model_input, pos, mask_tensor = mask_tensor)
            loss = sim_loss_function(embeddings, sim)
            
            avg_loss += loss.item()

            cnt += 1
#except:
#print('error')


    print('avg_loss = ', avg_loss/cnt)

def test(model, datapath, tokenizer, types, config):
    test_data = open(datapath + 'test.json').readlines()

    print('testing : ')

    model.eval()
    cnter9 = counter()
    cnter = counter()

    loss = 0.

    for x in test_data:
        model_input, mask_tensor, ans, pos = get_input([x], tokenizer, types)
        model_input = model_input.to(config.dev)
        mask_tensor = mask_tensor.to(config.dev)
        ans = ans.to(config.dev)

        model_output, embeddings = model(model_input, pos, mask_tensor = mask_tensor)

        loss += loss_function(model_output, ans).item()
        
        cnter.count(model_output.view(-1).tolist(),ans.view(-1).tolist())
        cnter9.count(model_output.view(-1).tolist()[:9],ans.view(-1).tolist()[:9]) #for 9-classification

    print("avg_loss on test : " , loss/len(test_data))
    print("on 130 types:")
    cnter.output()
    print("on 9 types:")
    return cnter9.output()

def main():
    config = parse()
    path = '.'
    datapath = '/data/private/luoyuqi/'


    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tokenizer.add_special_tokens({'additional_special_tokens':["<ent>", "<blank>"]})
    types = loads_from_file(path + '/types.json')

#model = noname(len(tokenizer)).to(config.dev)
    model = torch.load('./model/xcos.pth').to('cuda:0')
#model = torch.load('./model/cSfixf0.pth').to('cuda:0')
    model.activate_bert_fine_tuning(False)

#test(model, datapath, tokenizer, types, config)

    test_data = open(datapath + 'train_data_en.json').readlines()
#test_data = open(datapath + 'distant.json').readlines()
    test_contrastive(model, test_data, types, tokenizer)

    test_data = open(datapath + 'test.json').readlines()
    test_contrastive(model, test_data, types, tokenizer)


if __name__ == '__main__':
    main()
