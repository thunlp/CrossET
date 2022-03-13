import torch
import json
import time
import traceback

from tqdm import tqdm
from random import shuffle, randint
from transformers import BertTokenizer, BertModel

from source.model import noname
from source.parser import parse
from source.count import counter
from source.sampler import sampler
from source.function import *

from test import test

def train(model, data, epoch_num : int, lr : float = 1e-5):
    model.train()

    for i in range(epoch_num, epoch_num+1):
        print("\n===============Epoch.{}===============\n".format(i))

        optim = torch.optim.Adam(params=model.parameters(), lr = lr)
        batch_size = 32
        my_sampler = sampler(data,types,batch_size)

        cnter = counter()
        item_cnt = 0
        avg_loss = 0

        for samp in my_sampler.n_way_k_shot():
#for samp in my_sampler.get_batches():
            optim.zero_grad()

            try:
                model_input, mask_tensor, ans, pos = get_input(samp , tokenizer, types)
                model_input = model_input.to(config.dev)
                mask_tensor = mask_tensor.to(config.dev)
                sim = get_sim(samp).to(config.dev)
                ans = ans.to(config.dev)

                model_output,embeddings = model(model_input, pos, mask_tensor = mask_tensor)
                loss = loss_function(model_output, ans)
                
                cnter.count(model_output.view(-1).tolist(),ans.view(-1).tolist())

                loss /= batch_size
                avg_loss += loss.item()
                loss.backward()

                optim.step()
            
                item_cnt += 1
            except Exception as e:
                traceback.print_exc()

#if (batch_cnt+1) % int(batch_num/3) == 0 :
#if item_cnt == 10 :
        print('avg_loss = ', avg_loss/item_cnt)
        cnter.output()
        cnter.clear()

#torch.save(model.module,path + '/model/epoch{}.pth'.format(i))

def contrastive_train_batch(model,data, epoch_num : int , lr : float = 1e-5):
    model.train()

    for i in range(epoch_num):
        print("\n===============Epoch.{}===============\n".format(i))

        optim = torch.optim.Adam(params=model.parameters(), lr = lr)
        avg_loss = 0
        cnt = 0

        my_sampler = sampler(data, types, batch_size = 24)

        for samp in my_sampler.get_batches():
            try:
                optim.zero_grad()

                model_input, mask_tensor, ans, pos = get_input(samp, tokenizer, types, True)
                model_input = model_input.to(config.dev)
                mask_tensor = mask_tensor.to(config.dev)
                sim = get_sim(samp).to(config.dev)

                model_output, embeddings = model(model_input, pos, mask_tensor = mask_tensor)
                loss = sim_loss_function(embeddings, sim)

#                lmd = 1
#                for c in model.parameters():
#                    loss += lmd * (c**2).sum()
                
                avg_loss += loss.item()
                loss.backward()
                optim.step()

                cnt += 1
            except Exception as e:
                traceback.print_exc()

            if cnt == 100 :
                print('avg_loss = ', avg_loss/cnt)
                avg_loss = 0 
                cnt = 0

        print('avg_loss = ', avg_loss/cnt)
        avg_loss = 0 
        cnt = 0

def main():
    global tokenizer, types, config, path

    config = parse()
    path = '/home/luoyuqi/Entity_zh'
    datapath = '/data/private/luoyuqi/'

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tokenizer.add_special_tokens({'additional_special_tokens':["<ent>","<blank>"]})
    types = loads_from_file(path + '/types.json')

    c_data = open(datapath + 'distant.json','r').readlines()
#c_data = open(datapath + 'data_en.json','r').readlines()

    data = ([] 
#+ open(datapath + 'data_zh.json','r').readlines()
#+ open(datapath + 'data_en.json','r').readlines()
#+ open(datapath + 'train_data_en.json','r').readlines() 
+ open(datapath + 'simple_exp.json','r').readlines() 
#+ open(datapath + 'test.json','r').readlines() 
#+ open(datapath + 'trans_data_zh.json','r').readlines()
#+ open(datapath + 'train_data_zh.json','r').readlines()
    )

#model = noname(len(tokenizer)).to(config.dev)
    model = torch.load('./model/xcos.pth').to(config.dev)
    model = torch.nn.DataParallel(model, device_ids = [0])

    model.module.activate_bert_fine_tuning(True)
    contrastive_train_batch(model, c_data, 5, lr=1e-6)
    torch.save(model.module,path + '/model/x5cos.pth')
    exit()

    model.module.activate_bert_fine_tuning(False)
    for i in range(1):
        train(model, data, i, lr = 1e-2 * (0.9**i))
#        test(model.module, datapath, tokenizer, types, config)
    model.module.activate_bert_fine_tuning(True)

    maxf1 = 0 

    for i in range(20):
#contrastive_train_batch(model, data, 1)
        train(model, data, i, lr = 1e-5)

        model.module.activate_bert_fine_tuning(False)
        maxf1 = max(maxf1, test(model.module, datapath, tokenizer, types, config))
        model.module.activate_bert_fine_tuning(True)
        print('===========')

    print('maxf1 = ',maxf1)

    model = model.module
#torch.save(model,path + '/model/test_cc.pth')

    model.activate_bert_fine_tuning(False)
    test(model, datapath, tokenizer, types, config)

if __name__ == '__main__':
    main()
