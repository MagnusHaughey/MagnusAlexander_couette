from scipy import *
from scipy.linalg import eig, solve, inv, norm
import os
import pickle

##--------------------------

M = 250

##--------------------------

epsilon = 1.0/24.0
JSa = 0.985497788 

# Viscosity ratio 
vra = 0.0330112265
bbeta = vra/(1.0+vra)

nus = 25.0
nup = 25.0
nul = 80.0

Pe = 10000.
Na = 1.0e-4

Wi = 10.62
DT = 0.001

MaxIter = 8000000

##--------------------------

II = identity(M,dtype='d')

cbar = ones(M,dtype='d')
cbar[0] = 2.0
cbar[M-1] = 2.0

ygl = zeros(M,dtype='d')
for m in range(M):
    ygl[m] = cos(pi*m/(M-1))

rr = zeros(M,dtype='d')
for m in range(M):
    rr[m] = 0.5*epsilon*(ygl[m]+1.0)+1.0

_over_rr = 1.0/rr
_over_rr2 = 1.0/(rr*rr)

D1 = zeros((M,M),dtype='d')
for l in range(M):
    for j in range(M):
        if l != j:
            D1[l,j] = cbar[l]*((-1)**(l+j))/(cbar[j]*(ygl[l]-ygl[j]))

for j in range(1,M-1):
    D1[j,j] = -0.5*ygl[j]/(1.0-ygl[j]*ygl[j])

D1[0,0] = (2.0*(M-1)*(M-1)+1.0)/6.0
D1[M-1,M-1] = -D1[0,0]

D1 = 2*D1/epsilon
D2 = dot(D1,D1)


# Operators for T

Lstar = identity(M,dtype='d') - 0.25*DT*( D2 + dot(_over_rr*II,D1) )/Pe
Lnplus1 = identity(M,dtype='d') - 0.5*DT*( D2 + dot(_over_rr*II,D1) )/Pe

Lstar[0] = zeros(M,dtype='d'); Lstar[M-1] = zeros(M,dtype='d')
Lnplus1[0] = zeros(M,dtype='d'); Lnplus1[M-1] = zeros(M,dtype='d')

Lstar[0,0] = 1.0; Lstar[M-1,M-1] = 1.0 
Lnplus1[0,0] = 1.0; Lnplus1[M-1,M-1] = 1.0 


def EQS(Trr,Trt,Ttt,T):

    _x1 = (1.0/T) - 1.0

    Es = exp(nus*_x1)
    Ep = exp(nup*_x1)
    El = exp(-nul*_x1)

    _Mat = D2 + dot( (_over_rr-nus*dot(D1,T)/(T*T) )*II , D1 )
    _Mat += ( nus*dot(D1,T)*_over_rr/(T*T) -_over_rr2 )*II
    _Mat *= bbeta

    _rhsvec = -(1.0-bbeta)*( dot(D1,Trt) + 2.0*Trt*_over_rr )/Es

    _Mat[0] = zeros(M,dtype='d')
    _Mat[M-1] = zeros(M,dtype='d')
    _Mat[0,0] = 1.0
    _Mat[M-1,M-1] = 1.0
    _rhsvec[0] = 0.0
    _rhsvec[M-1] = epsilon*(2.0+epsilon)

    vel = solve(_Mat,_rhsvec)    
    gd = dot(D1,vel) - vel*_over_rr


    dTdt = (1.0/Pe)*( dot(D2,T) + _over_rr*dot(D1,T) ) + bbeta*(Na/Pe)*Es*gd*gd + (1.0-bbeta)*(Na/Pe)*Trt*gd


    RHSrr = T*El*Trr/Wi + (1.0-JSa)*Trt*gd - Trr*dTdt/T

    RHStt = T*El*Ttt/Wi - (1.0+JSa)*Trt*gd - Ttt*dTdt/T

    RHSrt = T*El*Trt/Wi + 0.5*( (1.0-JSa)*Ttt - (1.0+JSa)*Trr )*gd - Trt*dTdt/T - T*gd*Ep*El/Wi

    RHSenergy = bbeta*(Na/Pe)*Es*gd*gd + (1.0-bbeta)*(Na/Pe)*Trt*gd


    return -RHSrr, -RHSrt, -RHStt, RHSenergy





##-------------------------------------------
##
##    MAIN CODE
##
##-------------------------------------------

#f = open('init.txt','r')
#xstart = pickle.load(f)
#f.close()


Srr = zeros(M,dtype='d')
Srt = zeros(M,dtype='d')
Stt = zeros(M,dtype='d')
Temp = ones(M,dtype='d')

path = 'FIELDS_{}_{}_{}_{}_{}_{}_{}_{}'.format(DT, MaxIter, nus, nup, nul, Pe, Na, Wi)
try: 
    os.makedirs('{}/VELOCITY'.format(path))
    os.makedirs('{}/STRESS_RR'.format(path))
    os.makedirs('{}/STRESS_TT'.format(path))
    os.makedirs('{}/STRESS_RT'.format(path))
    os.makedirs('{}/TEMPERATURE'.format(path))
except OSError:
    if not os.path.isdir(path):
        raise

f = open('trace.txt','w')

for j in range(0,MaxIter):

    # Predictor step

    Frr, Frt, Ftt, FT = EQS(Srr,Srt,Stt,Temp)

    Srr_star = Srr + 0.5*DT*Frr
    Srt_star = Srt + 0.5*DT*Frt
    Stt_star = Stt + 0.5*DT*Ftt

    rhsvector = Temp + (0.25/Pe)*DT*dot( D2 + dot(_over_rr*II,D1),Temp ) + 0.5*DT*FT
    rhsvector[0] = 1.0
    rhsvector[M-1] = 1.0
    Temp_star = solve(Lstar,rhsvector)


    # Corrector step

    Frr, Frt, Ftt, FT = EQS(Srr_star,Srt_star,Stt_star,Temp_star)

    Srr = copy(Srr + DT*Frr)
    Srt = copy(Srt + DT*Frt)
    Stt = copy(Stt + DT*Ftt)

    rhsvector = Temp + (0.5/Pe)*DT*dot( D2 + dot(_over_rr*II,D1),Temp ) + DT*FT
    rhsvector[0] = 1.0
    rhsvector[M-1] = 1.0
    Temp = solve(Lnplus1,rhsvector)


    if j%100 == 0:

        print(j)

        f.write('%f %40.38f\n'%(j*DT,norm(Srt)))
        f.flush()


        _Mat = D2 + dot( (_over_rr-nus*dot(D1,Temp)/(Temp*Temp) )*II , D1 )
        _Mat += ( nus*dot(D1,Temp)*_over_rr/(Temp*Temp) -_over_rr2 )*II
        _Mat *= bbeta

        _rhsvec = -(1.0-bbeta)*( dot(D1,Srt) + 2.0*Srt*_over_rr )*exp(-nus*((1.0/Temp) - 1.0))

        _Mat[0] = zeros(M,dtype='d')
        _Mat[M-1] = zeros(M,dtype='d')
        _Mat[0,0] = 1.0
        _Mat[M-1,M-1] = 1.0
        _rhsvec[0] = 0.0
        _rhsvec[M-1] = epsilon*(2.0+epsilon)

        vel = solve(_Mat,_rhsvec)    

        ### Write data to files ###

        _filename_vel = 'FIELDS_{}_{}_{}_{}_{}_{}_{}_{}/VELOCITY/{}'.format(DT, MaxIter, nus, nup, nul, Pe, Na, Wi, j)
        file_vel = open(_filename_vel,'w')
        for m in range(M):
            file_vel.write('%f %20.18f\n'%(rr[m],vel[m]))
        file_vel.close()

        _filename_Srr = 'FIELDS_{}_{}_{}_{}_{}_{}_{}_{}/STRESS_RR/{}'.format(DT, MaxIter, nus, nup, nul, Pe, Na, Wi, j)
        file_Srr = open(_filename_Srr,'w')
        for m in range(M):
            file_Srr.write('%f %20.18f\n'%(rr[m],Srr[m]))
        file_Srr.close()

        _filename_Stt = 'FIELDS_{}_{}_{}_{}_{}_{}_{}_{}/STRESS_TT/{}'.format(DT, MaxIter, nus, nup, nul, Pe, Na, Wi, j)
        file_Stt = open(_filename_Stt,'w')
        for m in range(M):
            file_Stt.write('%f %20.18f\n'%(rr[m],Stt[m]))
        file_Stt.close()

        _filename_Srt = 'FIELDS_{}_{}_{}_{}_{}_{}_{}_{}/STRESS_RT/{}'.format(DT, MaxIter, nus, nup, nul, Pe, Na, Wi, j)
        file_Srt = open(_filename_Srt,'w')
        for m in range(M):
            file_Srt.write('%f %20.18f\n'%(rr[m],Srt[m]))
        file_Srt.close()

        _filename_temp = 'FIELDS_{}_{}_{}_{}_{}_{}_{}_{}/TEMPERATURE/{}'.format(DT, MaxIter, nus, nup, nul, Pe, Na, Wi, j)
        file_temp = open(_filename_temp,'w')
        for m in range(M):
            file_temp.write('%f %20.18f\n'%(rr[m],Temp[m]))
        file_temp.close()

f.close()


# vel = dot(MMM,Trt) + MMMvec

# f=open('vel.txt','w')
# for m in range(M):
#     f.write('%f %20.18f\n'%(rr[m],vel[m]))
# f.close()



# f=open('field.pickle','w')
# pickle.dump(xxx,f)
# f.close()
