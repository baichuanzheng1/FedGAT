import random
from random import shuffle
import os

cwd=os.getcwd()

def load5foldData(obj):

    if 'Twitter' in obj:
        labelPath = os.path.join(cwd,"data/" +obj+"/"+ obj + "_label_All.txt")
        labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
        print("loading tree label" )
        NR,F,T,U = [],[],[],[]
        l1=l2=l3=l4=0
        labelDic = {}
        for line in open(labelPath):          
            r = random.uniform(0,1)
            if r<1:
                line = line.rstrip()
                label, eid = line.split('\t')[0], line.split('\t')[2]
                labelDic[eid] = label.lower()
                if label in labelset_nonR:
                    NR.append(eid)
                    l1 += 1
                if labelDic[eid] in labelset_f:
                    F.append(eid)
                    l2 += 1
                if labelDic[eid] in labelset_t:
                    T.append(eid)
                    l3 += 1
                if labelDic[eid] in labelset_u:
                    U.append(eid)
                    l4 += 1
        print(len(labelDic))
        print(l1,l2,l3,l4)
        random.shuffle(NR)
        random.shuffle(F)
        random.shuffle(T)
        random.shuffle(U)

        fold0_x_test,fold0_x_train=[],[]

        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        leng3 = int(l3 * 0.2)
        leng4 = int(l4 * 0.2)
        
        # 每次取一个作为测试集剩下的为训练集
        fold0_x_train.extend(NR[leng1:])
        fold0_x_train.extend(F[leng2:])
        fold0_x_train.extend(T[leng3:])
        fold0_x_train.extend(U[leng4:])
        fold0_x_test.extend(NR[0:leng1])
        fold0_x_test.extend(F[0:leng2])
        fold0_x_test.extend(T[0:leng3])
        fold0_x_test.extend(U[0:leng4])
        
    if obj == "Weibo":
        labelPath = os.path.join(cwd,"data/Weibo/weibo_id_label.txt")
        print("loading weibo label:")
        F, T = [], []
        l1 = l2 = 0
        labelDic = {}
        for line in open(labelPath):
            line = line.rstrip()
            eid,label = line.split(' ')[0], line.split(' ')[1]
            labelDic[eid] = int(label)
            if labelDic[eid]==0:
                F.append(eid)
                l1 += 1
            if labelDic[eid]==1:
                T.append(eid)
                l2 += 1
        print(len(labelDic))
        print(l1, l2)
        random.shuffle(F)
        random.shuffle(T)

        fold0_x_test,fold0_x_train=[],[]
        
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        fold0_x_test.extend(F[0:leng1])
        fold0_x_test.extend(T[0:leng2])
        fold0_x_train.extend(F[leng1:])
        fold0_x_train.extend(T[leng2:])
        
    fold0_test = list(fold0_x_test)
    shuffle(fold0_test)
    fold0_train = list(fold0_x_train)
    shuffle(fold0_train)
    
    return list(fold0_test),list(fold0_train)