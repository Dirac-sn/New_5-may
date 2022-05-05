
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
                     
                          

def FourierCoeff(func,n,L,d=4):
    n_arr = np.arange(0,n)
    a_n=np.zeros(n)
    b_n=np.zeros(n)

    uppr_limit = 2*L
    coef_div = L

    calc_a_n = np.vectorize(lambda k: (1/coef_div)*integrate.quad(lambda x : np.cos(k*np.pi*x/L)*func(x),
    a=0,b=uppr_limit,epsrel=0.5*10**(-d))[0])
    calc_b_n = np.vectorize(lambda k: (1/coef_div)*integrate.quad(lambda x : np.sin(k*np.pi*x/L)*func(x),
    a=0,b=uppr_limit,epsrel=0.5*10**(-d))[0])
    
    
    a_n = calc_a_n(n_arr)
    b_n = calc_b_n(n_arr)

    a_n[0] /= 2
    return(a_n,b_n)

def Partials(func,n,L):
    cosnx = lambda x,i : np.cos(i*np.pi*x/L)
    sinnx = lambda x,i : np.sin(i*np.pi*x/L)

    a_i,b_i = FourierCoeff(func,n,L)
    
    na = np.arange(0,len(a_i))
    nb = np.arange(1,len(b_i))
    
    return (np.vectorize(lambda x :np.sum(cosnx(x,na)*(a_i))+np.sum(sinnx(x,nb)*(b_i[1:]))))


def Table(func,N,L):
    n_l = np.arange(1,N,1)
    a_an = np.zeros(N+1)
    a_cal,b_cal = FourierCoeff(func,N, L)
    a_an = [1/3]
    
    func = lambda n: (4*(-1)**n)/(10*n**2)
    for i in range(len(n_l)):
        a_an.append(func(n_l[i]))
        
    print([0]*N)
    print(b_cal)    
    
    data = {'an_analytical':a_an,'an_cal':a_cal,'bn_analytical':[0]*N,'bn_cal':b_cal}
    df = pd.DataFrame(data)
   
    return df
    
def func(x):
    
    if -1 <= x and x <= 1:
       return x**2
    elif x > 1 :
        return func(x-2)
    elif x < -1:
        return func(x+2)
     
func = np.vectorize(func)



def func_plot(x0,x1,step,N_f,Func_an):
    m = np.floor(np.log2(N_f))
    N = np.logspace(1,m,base = 2,num = int(m))
    
    x = np.arange(x0,x1+step,step)
    ls = ['*','o','v','x','d','1','2','o','v','x','*']
    sol = []
    for n in N:
        sol.append(Partials(Func_an,int(n),L=1)(x))
        
    fig,ax = plt.subplots()
    for i in range(len(N)):
        ax.plot(x,sol[i],ls = '--',marker = ls[i],label = N[i])
    ax.plot(x,Func_an(x)) 
    ax.legend()
    ax.set_title('For the given periodic function')
    ax.set_xlabel('x')
    ax.set_ylabel('func')
    plt.savefig('plot.png')
    plt.show()
        
func_plot(-2,2,0.2,32,func)    
    
print(Table(func,5, 1))       
    
print(Table(func,10, 1))

T = Partials(func,32, 1)([0.5,1])
y = func(0.5)

data = {'x':[0.5,1],'Y analytical':[y,500],'Y calc':T}
df = pd.DataFrame(data)
print(df)



    

    

