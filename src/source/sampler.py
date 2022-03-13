import json
import time

from random import shuffle, randint
from tqdm import tqdm

class sampler:
    def __init__(self, data, types, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.types = types
        self.n = len(data)
        self.ind = {}
        self.keys = []

        shuffle(self.data)

        for i in range(self.n):
            for ty in json.loads(self.data[i])['y_str']:
                self.ind.setdefault(ty, []).append(i)

        for k in self.ind:
            self.keys.append(k)
            self.ind[k] = list(set(self.ind[k])) #去重
    
    def get_batches(self): 
        j = 0
        lst_time = 0

        tmp = 1

        while j < self.n/tmp : 

            samp = []
            while len(samp) < self.batch_size and j < self.n/tmp:

                i = j

                tyid = json.loads(self.data[i])['y_str'][0]
                if(len(self.ind[tyid]) > 1):
                    i_sim = self.ind[tyid][randint(0, len(self.ind[tyid])-1)]
                    while i_sim == i: 
                        i_sim = self.ind[tyid][randint(0, len(self.ind[tyid])-1)]

                    if(self.data[i] == self.data[i_sim]):
                        pass
                    else:
                        samp.append(self.data[i])
                        samp.append(self.data[i_sim])
                j += 1

            if len(samp) < self.batch_size:
                break

            yield samp

            if time.time() - lst_time > 5 :
                print(str(i) + '/' + str(self.n/tmp))
                lst_time = time.time()

    def get_random_numbers(self,k,m):
        tmp = []
        for i in range(m):
            tmp.append(i)
        shuffle(tmp)
        for i in range(k):
            yield tmp[i]

    def n_way_k_shot(self):
        ''' fake
        '''
        lst_time = 0
        m = int(self.n / self.batch_size)
        p = [0]*len(self.ind)
        for i in range(m):
#            samp = []
#            for jj in self.get_random_numbers(self.batch_size>>1,len(self.ind)):
#                j = self.keys[jj]
#                if len(self.ind[j]) == 0:
#                    continue
#                for k in range(2):
#                    samp.append(self.data[self.ind[j][p[jj]%len(self.ind[j])]])
#                    p[jj] += 1 
            samp=self.data[i*self.batch_size:(i+1)*self.batch_size]
            yield samp

            if time.time() - lst_time > 3 :
                print(str(i) + '/' + str(m))
                lst_time = time.time()
