# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 08:51:03 2020

@author: mjflores
"""

#==================================================
# Implementacion de   ELM 
# Caso: v.a. normal y  uniforme
# Autor: MFlores, 2020
#==================================================
#  Datos
#  i |  x_i    | y_i
#-------------------
# 1    (-2,-2)   (1,0)
# 2    (-1,-1)   (1,0)
# 3    (+1,+1)   (0,1)
# 4    (+2,+2)   (0,1)
# 4    (-2,-4)   (1,0)
# 4    (+2,+4)   (0,1)

#-------------------------------------------

# Notacion
# N = Tamano datos de entrenamiento
# L = Numero de nodos de la capa oculta
# d = Dimension del vector x_i
# m = Numero de clases en  t_i
#
#==================================================

# Referencia
# An insight into Extreme Learning Machines: Random 
# Neurons, Random Features and Kernels
# by Guang-Bin Huang, 2014

import matplotlib.pyplot as plt
import random as rn
import numpy as np

#-------------------------------------------

def generar_a_b(row,col):
    a = np.zeros((row,col))
    rg = 15.0
    for i in range(row):
        for j in range(col):
            a[i,j] = rn.uniform(-10.90,10.90)
            #a[i,j] = rn.uniform(0.0,4*rg)
            #a[i,j] = rn.gauss(0.0,1*rg)
    b = [rn.uniform(0.0,4*rg) for _ in range(L)]           
    return a, b
#-------------------------------------------
    
def generar_H(a,b,X,N,L):
    H = np.zeros((N,L))
    for i in range(N):
        for j in range(L):
            H[i,j] = funcion_Sigmoid(a[j,:],b[j],X[i,:])
    return H, H.transpose()
#-------------------------------------------

def funcion_Sigmoid(a,b,x):
    aux = np.dot(a,x) + b
    return 1.0/(1.0+np.exp(-aux))
#-------------------------------------------

def funcion_Fourier(a,b,x):
    aux = np.dot(a,x) + b
    return np.sin(aux)
#-------------------------------------------

def funcion_Hardlimit(a,b,x):
    aux = np.dot(a,x)- b
    if aux >=0:
        return 1.0
    else:
        return 0.0
#-------------------------------------------

def funcion_Gaussian(a,b,x):
    aux  = np.linalg.norm(x-a)
    return np.exp(-b*aux*aux)
#-------------------------------------------

def funcion_Multiquadrics(a,b,x):
    aux  = np.linalg.norm(x-a)
    return np.sqrt(aux*aux + b*b)

#-------------------------------------------  
#-------------------------------------------    
#def Yi_2_Ti(Y):
#    r = len(Y)
#    mx = np.max(Y)
#    #print("Maximo: ",mx)
#    T = np.zeros((r,mx))
#    for i in range(r):        
#        for j in range(mx):
#            print(j)
#            if Y[i] ==j:
#                T[j] = j+1
#    return T
#
##-------------------------------------------
#Y = np.array([[1],
#              [1],
#              [2],
#              [2],
#              [2],
#              [3]])
#TT = Yi_2_Ti(Y)
#print(TT)
    
#=================================================
#=================================================
# Datos de entrenamiento 

X = np.array([[-2,-2],
              [-1,-1],
              [1,1],
              [2,2],
              [-2,-4],
              [2,4]])

T = np.array([[1,0],
              [1,0],
              [0,1],
              [0,1],
              [1,0],
              [0,1]])

#-------------------------------------------
plt.scatter(X[:,0], X[:,1])
plt.title('Datos')
plt.show()
#-------------------------------------------
L   = 500  # numero de capas ocultas 5e06 no puede operar
N,d = X.shape
m   = T.shape[1]

a, b    = generar_a_b(L,d)
H, H_tr = generar_H(a,b,X,N,L)

C = .10

print('Sigmoid',funcion_Sigmoid(np.array([1,2]),1,np.array([2,2])))
print('Fourier',funcion_Fourier(np.array([1,2]),1,np.array([2,2])))
print('HardLimit',funcion_Hardlimit(np.array([1,2]),1,np.array([2,2])))
print('Gausian',funcion_Gaussian(np.array([1,2]),1,np.array([2,2])))
print('Multiquadrics',funcion_Multiquadrics(np.array([1,2]),1,np.array([2,2])))

# Version cuando N grande
I1   = np.identity(L)/C
aux1 = np.linalg.inv((I1 + np.matmul(H_tr,H)))
aux2 = np.matmul(H_tr,T)
beta1 = np.matmul(aux1,aux2) 
# print(beta1)
# print("-------------------------------")


# Version cuando N pequeño
I2   = np.identity(N)/C
aux1 = np.linalg.inv((I2 + np.matmul(H,H_tr)))
aux2 = np.matmul(H_tr,aux1)
beta2 = np.matmul(aux2,T) 
# print(beta2)

def fx(xnew):
    hx = np.zeros(L)
    for j in range(L):
        hx[j] = funcion_Sigmoid(a[j,:],b[j],xnew)
        
    print(hx.shape)
    fx1 = np.matmul(hx,beta1)
    return fx1
    
#-------------------------------------------
#xnew = np.array([[-2,-3],[3,3]])
#xnew = np.array([-3,-3])
xnew = np.array([+3,+3])
hx = np.zeros(L)
for j in range(L):
    hx[j] = funcion_Sigmoid(a[j,:],b[j],xnew)
    
print(hx.shape)
fx1 = np.matmul(hx,beta1)
fx2 = np.matmul(hx,beta2)
print(fx1)
print(fx2)