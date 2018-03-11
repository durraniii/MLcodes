import numpy as np

def file_h(file_name, mode):
    
    arr=[]
    with open(file_name,mode) as handle:
        for line in handle:
            a,b,c,d,e = map(float,line.strip().split(","))
            arr.append((a,b,c,d,e))
    
    list=np.array(arr)        

    return list

def sigmoid(z):
    return 1/ (1+np.exp(-z))

data=file_h("asign2 data.txt","r")
m=len(data)
X=data[:, :-1]
Y=data[:, -1]


 
def shapes(we,tr):    

    prod=np.dot(we,tr)
    z=np.array(prod)
    
    A=sigmoid(z)    
    dz=A-Y         #dz
   
    dzt=dz.T    
    db=np.sum(dz)/m   #db                   
    
    dw=np.dot(tr,dzt)/m  #dw
    
    return dw,db
    
def calculations():
    alpha=0.01
    b=1
    TX=X.T
    y=Y.reshape(1,m)
    TY=y.T
 
    w=np.array([np.ones((4,),dtype=int)])     #weight
    
    for i in range (500):
        a,b1=shapes(w,TX)
        w=w-alpha*a.T
        b=b-alpha*b1   
        print(w)
        
calculations()