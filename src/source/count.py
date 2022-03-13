class counter:
    def __init__(self):
        self.clear()

    def clear(self):
        self.cnt = [[0,0],[0,0]]

    def count(self,out, ans):
        for i in range(len(out)):
            self.cnt[ans[i] > 0.99][out[i] > 0.4] += 1
    
    def output(self):
        try:
            p = self.cnt[1][1]*1.0 / (self.cnt[0][1] + self.cnt[1][1])
            r = self.cnt[1][1]*1.0 / sum(self.cnt[1])
# print('accuracy = ', (self.cnt[0][0] + self.cnt[1][1])*1.0 / (sum(self.cnt[0]) + sum(self.cnt[1])) )
            print('precision = ',  p)
            print('recall = ',  r)
            f1 = 2/(1/r+1/p)
            print('f1 = ', f1)
            print('')
            return f1
        except:
            pass
