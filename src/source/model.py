import torch 
import torch.nn as nn

from transformers import BertTokenizer, BertModel 

class noname(nn.Module):
    
    def __init__(self, len_tokenizer):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.bert.resize_token_embeddings(len_tokenizer)

        self.predict = nn.Linear(768,130,bias=False)
        self.sig = nn.Sigmoid()

        for x in self.bert.parameters():
            x.requires_grad = False
        
    def forward(self, x, pos, mask_tensor = None):
        x = self.bert(x, attention_mask = mask_tensor)
        x = x['last_hidden_state']

        cat_list = []
        n = len(pos)
        for i in range(n):
            cat_list.append(x[i,pos[i].item():pos[i].item()+1,:])
        x = torch.cat(cat_list,0)

        embeddings = x

        x = self.predict(x)
        x = self.sig(x)
        
        return x, embeddings

    def activate_bert_fine_tuning(self, state = True):
        for x in self.bert.parameters():
            x.requires_grad = state
