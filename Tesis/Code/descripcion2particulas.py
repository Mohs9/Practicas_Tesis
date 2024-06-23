#####################################################################
'''
En este programa se busca exhibir el mapa de probabilidades y el estado resultante luego de la medición, a partir de un estado inicial y un observable dado en un sistema de dos partículas.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from numpy import linalg as LA
import random
import array_to_latex as a2l

#Se define una función que un estado lo vuelva un operador proyector
def proyector(state):
    return np.outer(state,state)

##número de partículas
n=2


class observable:
    def __init__(self, matrix):
        self.obs = matrix
        #Calcular los valores propios y vectores propios de cada uno de los observables
        self.eigenvectors= eigh(matrix)[1].T
        self.eigenvalues=eigh(matrix)[0]
        self.len=len(self.obs)
#El observable que se desea medir en el sistema 1
obs1= observable(np.array([[1, 0], [0,-1]]))
#El observable que se desea medir en el sistema 2
obs2= observable(np.array([[0, 1], [1, 0]]))

#Calcular el observable del sistema conjunto y sus valores y vectores propios
observables= observable(np.kron(obs1.obs, obs2.obs))





## El estado inicial dado que se quiere medir
rho_inicial=np.array([[1/6,1/6,0,0],[1/6,1/6,0,0],[0,0,1/3,1/3],[0,0,1/3,1/3]])


####DESCRIPCIÓN##################################################################


###Definir el valor esperado###
def valor_esperado_fm(state,p):
    '''Valor esperado del observable, devuelve un número'''
    obs=np.kron(obs1.obs,obs2.obs)
    SobsS=np.kron(obs2.obs,obs1.obs)
    nuevo_obs=p*obs+(1-p)*SobsS
    valor= np.trace(np.matmul(state,nuevo_obs))
    return valor

###Obtener los efectos par A y B no degenerados####
def efectos_fm(output1, output2,p=0.5):
    #Tomar primero los vectores propios  de A y luego los vectores propios de B
    for i in range(len(obs1.eigenvectors)):
        if obs1.eigenvalues[i]==output1:
            operador_proyector1= proyector(obs1.eigenvectors[i])
    for i in range(len(obs2.eigenvectors)):
        if obs2.eigenvalues[i]==output2:
            operador_proyector2= proyector(obs2.eigenvectors[i])
    operador_proyector=np.kron(operador_proyector1,operador_proyector2)
    Soperador_proyectorS=np.kron(operador_proyector2,operador_proyector1)
    fuzzy_operator=p*operador_proyector+(1-p)*Soperador_proyectorS
    return fuzzy_operator

print(efectos_fm(1,1,1/4))

####Mapeo de probabilidades###################################################
def mapeo_fm(output1, output2, state,p):
    '''Realizar un mapeo de probabilidades, toma como entrada, 
    la salida, el estado inicial y la probabilidad de intercambio de partículas.
    Devuelve la probabilidad de obtener esa salida en una medición difusa.'''
    #inicializar el efecto
    #efecto=np.array([[0 for j in range(len(eigenvectors))] for i in range(len(eigenvectors))])
    probabilidad= np.trace(np.matmul(efectos_fm(output1,output2, p), state))
    return probabilidad


##############Obtener el estado posterior a la medición.
def estado_final(output1,output2, state, p):
    '''Realizar un mapeo de estados, toma como entrada, 
    la salida, el estado inicial y la probabilidad de intercambio de partículas.
    Devuelve el estado posterior dado la salida en una medición difusa.'''
    #Sacar la raiz cuadrada de cada efecto dependiendo la salida
    eigenvalues, eigenvectors=eigh(efectos_fm(output1, output2,p))
    #Sacarle la raiz a los valores propios
    eigenvalues=[round(eigenvalues[i],6) for i in range(len(eigenvalues))]
    eigenvalues_nuevos= np.sqrt(eigenvalues)
    #Inicializar el operador de Kraus
    kraus_operator=np.array([[0 for j in range(len(eigenvectors))] for i in range(len(eigenvectors))])
    #Sumar los proyectores del efecto multiplicado por los nuevos valores propios
    for i in range(len(eigenvectors)):
        proyectores=proyector(eigenvectors.T[i])
        kraus_operator=np.add(kraus_operator,eigenvalues_nuevos[i]*proyectores)
    estadoFinal= np.matmul(np.matmul(kraus_operator,state),kraus_operator)
    return estadoFinal/np.trace(estadoFinal)

print(estado_final(1,1,rho_inicial,1/4))
######
# ################################################################################
#####Gráficas#####################################

def graficar_distribucion(p, ax):
    ##data
    eigval=np.array([])
    for out1 in obs1.eigenvalues:
        for out2 in obs2.eigenvalues:
            eigval=np.append(eigval,out1*out2)

    mapeos=np.array([])
    for out1 in obs1.eigenvalues:
        for out2 in obs2.eigenvalues:
            prob=mapeo_fm(out1,out2,rho_inicial,p)
            mapeos=np.append(mapeos,prob)
    X=eigval
    nueva_X=set(X)
    nueva_Y=np.array([])
    for i in nueva_X:
        #crear lista de indices donde estan los repetidos
        indices=np.where(X == i)[0]
        #inicializar el valor de la probabilidad
        prob=0
        for j in indices:
            prob+=mapeos[j]
        nueva_Y=np.append(nueva_Y,prob)



    ######
    ax.stem(np.array(list(nueva_X)), nueva_Y,  linefmt='-', markerfmt='black', basefmt='black')
    ax.vlines(x = valor_esperado_fm(rho_inicial,p), ymin = 0, ymax = 1,
           colors = 'purple',
           label = 'Valor esperado:'+str(round(valor_esperado_fm(rho_inicial,p),5))
           +' con 'r'$p$='+str(round(p,5)),
           ls='--')
    
    #axis settings  
    ax.set_ylim([0, 1])
    ax.grid(linewidth=0.2)
    ax.set_ylabel("Probabilidad de obtener la salida")
    ax.set_xlabel('Salidas de la medición')
    for x,y in zip(np.array(list(nueva_X)),nueva_Y): 
        #ax.annotate(y,(x,y-4))
        ax.legend(loc=9)
        ax.annotate(str(round(y,3)), xy=(x,y), xytext=(0,5), textcoords='offset points',ha='center')

    return 0

def mostrar_graficas(ncol=2, nrow=3):
    if ncol>1 and nrow>1:
        fig, axes = plt.subplots(ncol, nrow, figsize=(nrow*5, ncol*5))
        for i, axe in enumerate(axes.flat):
            graficar_distribucion(p=1/(i+1), ax=axe)
        plt.show()
    else:
        graficar_distribucion(p=0.25,ax=plt.subplot())
        plt.show()


####GRAFICAS EN 3D PARA MOSTRAR CADA COMBINACIÓN DE a_j*b_k####
##Esta gráfica solo tiene sentido para sistemas de dos partículas.
def graficar_distribucion_3d(p):
    ##data
    z=np.array([])
    x=[]
    y=np.array([])
    for i in range(len(obs1.eigenvalues)):
        x.extend([obs1.eigenvalues[i]]*len(obs1.eigenvalues))
    x=np.array(x) 

    for i in range(len(obs2.eigenvalues)):
        for j in range(len(obs2.eigenvalues)):
            y=np.append(y,obs2.eigenvalues[j])  

    for out1 in obs1.eigenvalues:
        for out2 in obs2.eigenvalues:
            map= mapeo_fm(out1,out2,rho_inicial,p)
            z=np.append(z,map)

    ######
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.stem(x,y,z, basefmt='None')
    #axis settings  
    ax.set_zlim([0, 1])
    ax.grid(linewidth=0.1)
    ax.set_zlabel("Probabilidades")
    ax.set_ylabel("Salidas del observable $\sigma_x$")
    ax.set_xlabel('Salidas del observable $\sigma_z$')
    for x,y,z in zip(x,y,z): 
        #ax.annotate(y,(x,y-4))
        ax.text(x,y,z, str(round(z,3)))
    plt.savefig(f'../Images/ejemplo1.pdf', format='pdf')
    #ax.set_title('Mapeo de resultados en un sistema de dos partículas')
    plt.show()
    return 0


##################Instrumento###########################################

#############PRIMER INSTRUMENTO########################################
###Definir el operador difuso##### 
### Solo para dos partículas coon observables de dos dimensiones
def fuzzy_operator(state,p):
    '''Aplicar el operador difuso a un estado inicial devuelve otro operador'''
    SWAP=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    rho_posterior= p*state+(1-p)*np.matmul(np.matmul(SWAP,state),SWAP)
    return rho_posterior



def sis_clasico_1ins(out1,out2):
    for i in range(len(obs1.eigenvalues)):
        if obs1.eigenvalues[i]==out1:
            operador_proyector1= proyector(obs1.eigenvectors[i])
    for i in range(len(obs2.eigenvalues)):
        if obs2.eigenvalues[i]==out2:
            operador_proyector2= proyector(obs2.eigenvectors[i])
    operador_proyector=np.kron(operador_proyector1,operador_proyector2)
    return operador_proyector

def sis_cuantico_1ins(out1,out2, rho,p):
    proy=sis_clasico_1ins(out1,out2)
    canal=np.matmul(proy, np.matmul(fuzzy_operator(rho,p),proy))
    return canal


def primer_instrumento(rho, p):
    '''Repica el instrumento cuantico dado por el ensamble de un sistema clasico
    y uno cuantico'''
    instrumento=np.array([[0 for vec in observables.eigenvectors] for vec in observables.eigenvectors])
    instrumento=np.kron(instrumento,instrumento)
    for out1 in obs1.eigenvalues:
        for out2 in obs2.eigenvalues:
            selectivo=np.kron(sis_clasico_1ins(out1,out2), sis_cuantico_1ins(out1,out2,rho, p))
            instrumento=np.add(instrumento,selectivo)
    return instrumento


#print(primer_instrumento(rho_inicial,0.25))

def segundo_instrumento(rho, p):
    '''Repica el instrumento cuantico dado por el ensamble de un sistema clasico
    y uno cuantico'''
    instrumento=np.array([[0 for vec in observables.eigenvectors] for vec in observables.eigenvectors])
    instrumento=np.kron(instrumento,instrumento)
    for vector in observables.eigenvectors:
        sis_clasico=fuzzy_operator(proyector(vector),p)
        sis_cuantico=np.matmul(proyector(vector), np.matmul(rho,proyector(vector)))
        selectivo=np.kron(sis_clasico, sis_cuantico)
        instrumento=np.add(instrumento,selectivo)
    return instrumento
########################################