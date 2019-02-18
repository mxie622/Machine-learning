from matplotlib import pyplot as plt
import  random
# %matplotlib inline


def data():
    x = range(10)
    y = [(2*i+4) for i in x]
    for i in range(10):
        y[i] = y[i]+random.randint(0,8)-4
    return x,y


def SGD(x,y):
    error0 = 0
    step_size = 0.001
    esp = 1e-6
    #a = random.randint(0,4)
    #b = random.randint(0,8)
    a = 1.2
    b = 3.5
    m = len(x)

    n = 0
    while True:
        i = random.randint(0,m-1)

        sum0 = a * x[i] + b - y[i]
        sum1 = (a * x[i] + b - y[i])*x[i]
        error1 = (a * x[i] + b - y[i])**2

        a = a - sum1*step_size/m
        b = b - sum0*step_size/m
        print('a=%f,b=%f,error=%f'%(a,b,error1))

        if abs(error1-error0)<esp:  
            break
        error0 = error1
        n = n+1
        if n%20==0:                
                print('ITERATE%d'%n)
        if (n>500):
            break
    return a,b
if __name__ == '__main__':
    x,y = data()
    a,b = SGD(x,y)
    X = range(10)
    Y = [(a*i+b) for i in X]

    plt.scatter(x,y,color='red')
    plt.plot(X,Y)
    plt.show()

