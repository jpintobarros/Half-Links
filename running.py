from functions import*
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("N",type=int, help="Total number of sites")
parser.add_argument("jobid",type=int, help="The job id will determine the fermion mass")
parser.add_argument("Njobs",type=int, help="Total number of jobs")
args = parser.parse_args()

#The number of gauge invariant vertices
L=args.N//2

#We investigate masses from 0 to 1 according to jobid and Njobs
m=args.jobid/args.Njobs

##### PERIODIC BOUNDARY CONDITIONS #####

base,dic=build_states(L,0,0)
H=Hamiltonian(1,m,L,base,dic,bc='pbc')

#Diagonalization
w,v=eigh(H)

#Shannon entropy
sh=[shannon_st(v[:,i]) for i in range(len(v))]

#Level Statistics
Nstates=len(w)
lev_stat=[]
s0=w[1]-w[0]
for i in range(1,Nstates-1):
    s=w[i+1]-w[i]
    if abs(s)<.00000001 or abs(s0)<.00000001:
        continue
    r=s/s0
    lev_stat+=[min(r,1/r)]
    s0=s

#Superposition with defined states
sup_st_dic={}
for i in range(4):
    ind=dic[int(str(i)*L)]
    sup_st_dic[str(i)]=[abs(v[ind,i])**2 for i in range(Nstates)]
ind=dic[int('45'*(L//2))]
sup_st_dic['45']=[abs(v[ind,i])**2 for i in range(Nstates)]


with open('L{}_m{:.2f}.pkl'.format(L,m), 'wb') as file:
    pickle.dump(w, file)
    pickle.dump(sh, file)
    pickle.dump(lev_stat, file)
    pickle.dump(sup_st_dic,file)