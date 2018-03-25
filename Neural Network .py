
# coding: utf-8

# In[76]:


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/ (1+np.exp(-z))

def derivative_sig(z):
    return z*(1-z)

def file_h(file_name, mode):
    
    arr=[]
    with open(file_name,mode) as handle:
        for line in handle:
            a,b,c,d,e = map(float,line.strip().split(","))
            arr.append((a,b,c,d,e))
    
    list=np.array(arr)        

    return list


data=file_h("nn data.txt","r")
m=len(data)
X=data[:, :-1]
Y=data[:, -1]


features=4
input_l=features     #inputlayer
hidden=8           #hiddenlayer
a2=1               #outputlayer

w_hidden=np.random.uniform(size=(hidden,features))
b_hidden=np.random.uniform(size=(1,hidden))
w_out=np.random.uniform(size=(a2,hidden))
b_out=np.random.uniform(size=(a2,1))
alpha=0.01

for i in range(240):
    h1=np.dot(w_hidden,X.T)  #forward propagation
    z1=b_hidden.T+h1
    a1=sigmoid(z1)
    
    h2=np.dot(w_out,a1)
    z2=h2+b_out
    output=sigmoid(z2)

    dz2=output-Y
    dw2=np.dot(dz2,a1.T)/m
    db2=np.sum(dz2,axis=1,keepdims=True)/m
    
    deri=derivative_sig(a1)
    dz1 = np.dot(w_out.T,dz2) *deri  
    dw1=np.dot(dz1,X)/m
    db1=np.sum(dz1,axis=1,keepdims=True)/m
   
    w_out=w_out-alpha*dw2
    b_out=b_out-alpha*db2.T
    w_hidden=w_hidden-alpha*dw1
    b_hidden=b_hidden-alpha*db1.T
      
print("Loss : ",output)

print("w1:",w_hidden)
print("w2:",w_out)

