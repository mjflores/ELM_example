# -*- coding: utf-8 -*-
"""
        Basic implementation of ELM

@date: 31/01/2020
@author: mjflores
"""

#==================================================
# Implementacion de   ELM 
# Autor: MFlores, Enero 2020
#==================================================
#  Datos
#  i |  x_i    | y_i
#-------------------
# 1    (-2,-2)   (1,0)
# 2    (-1,-1)   (1,0)
# 3    (-2,-4)   (1,0)
# 4    (-3,-3)   (1,0)
# 5    (-2,-3)   (0,1)
# 6    (+1,+1)   (0,1)
# 7    (+2,+2)   (0,1)
# 8    (+2,+4)   (0,1)
# 9    (+3,+3)   (0,1)
# 10   (+2,+3)   (0,1)


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

def generar_a_b(row_d,col_L):
    a = np.random.uniform(-1,1,size=(row_d,col_L))
    b = np.random.uniform(-1,1,size=(col_L,1))
    return a, b
#-------------------------------------------
    
def generar_H(a,b,X,N,L):
    H = np.zeros((N,L))
    for i in range(N):
        for j in range(L):
            #print(a[:,j])    
            H[i,j] = funcion_Sigmoid(a[:,j],b[j],X[i,:])
    return H, H.transpose()
#-------------------------------------------

def funcion_Sigmoid(a,b,x):
    aux = np.dot(a,x) + b
    return 1.0/(1.0+np.exp(-aux))   
#=================================================
#=================================================
# Datos de entrenamiento 

X = np.array([[-2,-2],
              [-1,-1],
              [-2,-4],
              [-3,-3],
              [-2,-3],
              [1,1],
              [2,2],
              [2,4],
              [3,3],
              [2,4],])

T = np.array([[1,0],
              [1,0],
              [1,0],
              [1,0],
              [1,0],
              [0,1],
              [0,1],
              [0,1],
              [0,1],
              [0,1]])
#-------------------------------------------

plt.scatter(X[:,0], X[:,1])
plt.title('Datos para ELM')
plt.show()
#-------------------------------------------
L   = 70  # numero de capas ocultas 5e06 no puede operar
N,d = X.shape
m   = T.shape[1]

a, b    = generar_a_b(d,L)
#print("Dimension a: ",a.shape)
H, H_tr = generar_H(a,b,X,N,L)

C = .10

beta = []
# Version cuando N grande
if N>L: 
    # Eq (11)
    I1   = np.identity(L)/C
    aux1 = np.linalg.inv((I1 + np.matmul(H_tr,H)))
    aux2 = np.matmul(H_tr,T)
    beta1 = np.matmul(aux1,aux2) 
    beta = beta1
# Version cuando N pequeño
else:
    I2   = np.identity(N)/C
    aux1 = np.linalg.inv((I2 + np.matmul(H,H_tr)))
    aux2 = np.matmul(H_tr,aux1)
    beta2 = np.matmul(aux2,T) 
    beta = beta2
#-------------------------------------------
def fx(xnew):
    hx = np.zeros(L)
    for j in range(L):
        hx[j] = funcion_Sigmoid(a[:,j],b[j],xnew)
        
    #print(hx.shape)
    fx1 = np.matmul(hx,beta1)
    return fx1
    
#-------------------------------------------

xnew1 = np.array([-3,-3])
xnew2 = np.array([+0,+2])
    
fx1 = fx(xnew1)
fx2 = fx(xnew2)
print("fx1 = ",fx1)
print("fx2 = ",fx2)
