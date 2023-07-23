from matplotlib.pyplot import*
from numpy import zeros, array,log,dot,sqrt,conjugate,real,imag
from numpy import arange,transpose
from scipy.linalg import eigh,eig
from copy import deepcopy


#Array storing values of the electric field associated with the different vertices
ef=zeros(6)+1
ef[2]=-1;ef[3]=-1;ef[5]=-1

#Array storing values of the charge associated with the different vertices
ch=zeros(6)
ch[4]=1;ch[5]=-1

#Given a state, returns the list of valid states with one more site
def add_site(st):
    n=st[-1]
    lst=[]
    if n==0:
        lst=[st+[0],st+[1],st+[5]]
    elif n==1:
        lst=[st+[0],st+[1],st+[5]]
    elif n==2:
        lst=[st+[2],st+[3],st+[4]]  
    elif n==3:
        lst=[st+[2],st+[3],st+[4]]
    elif n==4:
        lst=[st+[0],st+[1],st+[5]]
    else:
        lst=[st+[2],st+[3],st+[4]]
    return lst

#Returns the possible single vertex transitions
def transitions_single(n):
    if n<2:
        return [1-n] 
    elif n<4:
        return [5-n]
    return []

#Returns the possible two vertex transitions
def transitions_pair(p):
    if p==[1,1]:
        return [5,4]
    elif p==[2,2]:
        return [4,5]
    elif p==[1,5]:
        return [5,2]
    elif p==[1,1]:
        return [5,4]
    elif p==[2,2]:
        return [4,5]
    elif p==[2,4]:
        return [4,1]
    elif p==[4,1]:
        return [2,4]
    elif p==[4,5]:
        return [2,2]
    elif p==[5,2]:
        return [1,5]
    elif p==[5,4]:
        return [1,1]
    return []

def st_2_M(st):
    return int(''.join(map(str,st)))

#Building states dependent on the specified boundary conditions
#The expected values of the links are -1, 0 and 1, with 0 meaning pbc
def build_states(L,link_left=0,link_right=0):
    M2m={}
    if link_left==0:
        st_lst=[[0],[1],[2],[3],[4],[5]]
        for i in range(L):
            new_lst=[]
            for st in st_lst:
                new_lst+=add_site(st)
            st_lst=deepcopy(new_lst)
        new_lst=[]
        count=0
        for st in st_lst:
            if st[-1]==st[0]:
                new_lst+=[st[:-1]]
                M2m[st_2_M(new_lst[-1])]=count
                count+=1
    else:
        if link_left==1:
            st_lst=[[0],[1],[5]]
        else:
            st_lst=[[2],[3],[4]]
        for i in range(L):
            new_lst=[]
            for st in st_lst:
                new_lst+=add_site(st)
            st_lst=deepcopy(new_lst)
            new_lst=[]
            count=0
            allowed=[[2,3,5],[0,1,4]][(link_right+1)//2]
            for st in st_lst:
                if st[-1] in allowed:
                    new_lst+=[st]
                    M2m[st_2_M(new_lst[-1])]=count
                    count+=1
    return new_lst,M2m

def Hamiltonian(t,m,L,st_lst,M2m,bc='pbc'):
    N=len(st_lst)
    H=zeros((N,N))
    for i,st in enumerate(st_lst):
        for j,n in enumerate(st):
            if n<4:
                H[i,i]+=m*(-1)**n
            trans=transitions_single(n)
            if trans!=[]:
                M=st_2_M(st[:j]+trans+st[j+1:])
                H[i,M2m[M]]=-t
            trans=transitions_pair([st[j],st[(j+1)%L]])
            if trans!=[] and j<L-1:
                M=st_2_M(st[:j]+trans+st[j+2:])
                H[i,M2m[M]]=-t
            elif trans!=[] and bc=='pbc':
                M=st_2_M([trans[-1]]+st[1:-1]+[trans[0]])
                H[i,M2m[M]]=-t
    return H    


def mass_term(m,L,st_lst):
    N=len(st_lst)
    H=zeros((N,N))
    for i,st in enumerate(st_lst):
        for j,n in enumerate(st):
            if n<4:
                H[i,i]+=m*(-1)**n
    return H

def shannon_st(st):
    s=0
    for a in st:
        if abs(a)>0.000000001:
            s-=abs(a**2)*log(abs(a**2))
    return s

def Efield(st_lst):
    N=len(st_lst)
    E=zeros((N,N))
    for i,st in enumerate(st_lst):
        for n in st:
            E[i,i]+=ef[n]
    return E

def charge_at_x(x,st_lst):
    N=len(st_lst)
    C=zeros((N,N))
    for i,st in enumerate(st_lst):
        C[i,i]+=ch[st[x]]
    return C

#In order for pair creation to occur between x and y (assuming x<y) one of two sitations must occur:
###-> x and y are in {0,1} and electric field must be altered accross the boundary
###-> x and y are in {2,3} and electric field must be betwwen x and y
def creat_pair_pm(x,y,st_lst,M2m):
    N=len(st_lst)
    L=len(st_lst[0])
    C=zeros((N,N))
    for i,st in enumerate(st_lst):
        if st[x] in [0,1] and st[y] in [0,1]:
            new_st=zeros(L,int)
            new_st[x+1:y]=st[x+1:y]
            new_st[x]=4
            new_st[y]=5
            valid=True
            for z in range(x):
                if st[z]>3:
                    valid=False
                    break
                new_st[z]=st[z]+2
            if valid:
                for z in range(y+1,L):
                    if st[z]>3:
                        valid=False
                        break
                    new_st[z]=st[z]+2
            if valid:
                M=st_2_M(new_st)
                C[M2m[M],i]=1
        elif st[x] in [2,3] and st[y] in [2,3]:
            new_st=zeros(L,int)
            new_st[:x]=st[:x]
            new_st[y+1:]=st[y+1:]
            new_st[x]=4
            new_st[y]=5
            valid=True
            for z in range(x+1,y,1):
                if st[z]>3:
                    valid=False
                    break
                new_st[z]=st[z]-2
            if valid:
                M=st_2_M(new_st)
                C[M2m[M],i]=1
    return C
