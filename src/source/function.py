import torch
import json
import time

def gettime():
    return time.strftime("%m-%d-%H:%M",time.localtime(time.time() + 8*3600))

def loads_from_file(path):
    f = open(path, 'r')
    ret = json.loads(f.read())
    f.close()
    return ret

def get_input(slice, tokenizer, types, isBlank=False):
    model_input = []
    mask_tensor = []
    pos = []
    ans = []
 
    ind = 0
    for line in slice:
        obj = json.loads(line)

        ind += 1
        if isBlank and ind%2==1:
            s = '[CLS]' +  obj['left_context_token'] + ' <ent> <blank>  <ent> ' + obj['right_context_token']
        else:
            s = '[CLS]' +  obj['left_context_token'] + ' <ent> ' + obj['mention_span'] + ' <ent> ' + obj['right_context_token'] 

        s = tokenizer.tokenize(s)
        pos.append(s.index('<ent>'))
        model_input.append(tokenizer.convert_tokens_to_ids(s))
        mask_tensor.append([1]*len(s))

        ans_tmp = [0.] * 130
        for ty in obj['y_str'] : 
            if ty in types:
                ans_tmp[types.index(ty)] = 1.
        ans.append(ans_tmp)
 
    maxlen = max([len(s) for s in model_input])
    model_input = [ (s+[0]*(maxlen-len(s))) for s in model_input]
    mask_tensor = [ (s+[0]*(maxlen-len(s))) for s in mask_tensor]
    
    model_input = torch.tensor(model_input)
    mask_tensor = torch.tensor(mask_tensor)
    ans = torch.tensor(ans)
    pos = torch.tensor(pos)

    return model_input, mask_tensor, ans, pos

def get_sim(slice):
    n = len(slice)

    objs = []
    for line in slice:
        objs.append(json.loads(line))

    ret = torch.zeros(n,n)
    for i in range(n):
        for j in range(i+1):
            for ty in objs[i]['y_str']:
                if ty in objs[j]['y_str']:
                    ret[i][j] = 1
                    ret[j][i] = 1
                    break
    return ret


def loss_function(model_output, ans):
#return torch.nn.CrossEntropyLoss()(model_output, ans)

    model_output = model_output*0.99999 + 0.000009

    return - torch.sum(
        torch.log(model_output) * ans
        + torch.log(1-model_output) * (1-ans)
    )

def is_sim(x, y):
    x = json.loads(x)
    y = json.loads(y)
    for ty in x['y_str']:
        if ty in y['y_str']:
            return True
    return False

def sim_loss_function(model_output, sim): #model_output : n*dim, sim : n*n
    M = torch.nn.functional.normalize(model_output) * 2
#M = model_output
    M = M.matmul(M.t())
    M = M - torch.eye(M.size()[0]).to(M.device.type)*1000

    M=torch.exp(M)
    return (-torch.log((M*sim).sum(0)+1e-30) + torch.log( (M*(1-sim)).sum(0)+1e-30)).sum()

    M = M.softmax(0)
    M = (M*sim).sum(0)
    return - (M * 0.999999 + 0.0000009).log().sum()
