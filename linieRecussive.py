import numpy as np
#Define the parameters of the line equation
true_w=np.array([23.2,-45.32]).reshape(2,1)
true_b=np.array([-78.9])

n_batch=1000
#init data
x=np.random.random([n_batch,2])
y=np.dot(x,true_w)
y+=np.random.random([n_batch,1])/100
# Randomly extract values for a given batch size
def data_itr(batch_size):
    for i in range(1,n_batch,batch_size):
        index=np.random.randint(n_batch-1,size=batch_size)
        yield x[index],y[index]
#Randomly init the parameters of the line equation
w=np.random.randn(2,1)
b=np.random.random(1)
batch_size=100
# Learning rate
lr=0.1

for i in range(100):
    for xs,ys in data_itr(batch_size):
        # Calculate y 
        yt=np.dot(xs,w)+b
        dout=yt-ys
        # Calculate variance
        loss=dout**2/2
        # Calculate gradient
        dw=np.dot(xs.T,dout)/batch_size
        db=np.sum(dout,axis=0)/batch_size
        # Optimization parameter
        w-=dw*lr
        b-=db*lr
    print("i=%d, loss=%f"%(i,np.mean(loss)))
print([w,b])
