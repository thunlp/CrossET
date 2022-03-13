import torch 
import torch.nn as nn

from transformers import BertTokenizer, BertModel 

class disc(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.activator = nn.Sigmoid()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.scalar = nn.Sequential(
            nn.Linear(768+130,1000),
            self.activator,
            nn.Linear(1000,1)
        )

        for x in self.bert.parameters():
            x.requires_grad = False
        
    def forward(self, x, y, pos):
        x = self.bert(x)
        x = x['last_hidden_state']
        x = x[0,pos,:]
        
        x = torch.cat(x,y)
        x = self.scalar(x)
        
        return x