#####################################################################
'''
En este programa se busca exhibir el mapa de probabilidades y el estado resultante luego de la medición, a partir de un estado inicial y un observable dado en un sistema de varias partículas.
'''

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif' 
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
from numpy import linalg as LA
import random
import array_to_latex as a2l
import functools as ft
from itertools import permutations 

#Se define una función que un estado lo vuelva un operador proyector
def proyector(state):
    return np.outer(state,state)

#Es una función que ayudará en las gráficas. Devolverá una lista de todas las posibles salidas de los n observables.
def cross_lists(lista1, lista2):
    '''Input:Dos listas, la primera debe ser un arreglo de arreglos, la segunda solo un arreglo.'''
    all_outputs=[]
    for i in range(len(lista1)):
        for j in lista2:
            flag=lista1[i]#recorre los valores propios del primer observable
            flag=np.append(flag, j)#Va anexando los valores propios del segundo observable al primero.
            all_outputs.append(flag)
    return all_outputs

#print(cross_lists([[1],[-1]],[1,-1]))


##número de partículas
n=2

class observable:
    def __init__(self, matrix):
        self.obs = matrix
        #Calcular los valores propios y vectores propios de cada uno de los observables
        self.eigenvectors= eigh(matrix)[1].T
        self.eigenvalues=eigh(matrix)[0]
        self.len=len(self.obs)
#####################Escribir los observables##################
#El observable que se desea medir en el sistema 1
obs1= observable(np.array([[1, 0], [0,-1]]))
#El observable que se desea medir en el sistema 2
obs2= observable(np.array([[0, 1], [1, 0]]))
#El observable que se desea medir en el sistema 3
obs3= observable(np.array([[1, 0], [0, -1]]))

#Una lista donde se encuentren todos los observables
observables=[obs1,obs2]


#En la siguiente lista se encuentran los observables
operadores=[]
for obser in observables:
    operadores.append(obser.obs)

## Escribir el estado inicial dado que se quiere medir
state1=[[1,0],[0,0]] 
state2=[[0.5,0.5],[0.5,0.5]]
state3=[[1,0],[0,0]]
states=[state1,state2]
rho_inicial=ft.reduce(np.kron,states)


###DESCRIPCIÓN##################################################################
###Definir el operador difuso##### 
def fuzzy_operator(estados,probabilidades):
    '''Aplicar el operador difuso a un estado inicial devuelve otro operador'''
    perm= permutations(estados)
    suma=0*ft.reduce(np.kron,estados)
    count=0
    for i in list(perm):
        operador_permutado=ft.reduce(np.kron,i)
        suma= np.add(suma,probabilidades[count]*operador_permutado)
        count+=1 
    return suma




##Una lista de probabilidades aleatorias (pueden cambiar)
##falta corregir

def generar_probabilidades(m):
    # Generar una lista de n números aleatorios
    probabilidades = [random.random() for _ in range(m)]
    # Calcular la suma de los números aleatorios
    suma_total = sum(probabilidades)
    # Normalizar los números para que sumen 1
    probabilidades = [p/suma_total for p in probabilidades]
    return probabilidades

probas=generar_probabilidades(np.math.factorial(n))
#prueba
probas=[0.25,0.75]
###Definir el valor esperado###
def valor_esperado_fm(probabilidades):
    '''Valor esperado del observable, devuelve un número'''
    suma=fuzzy_operator(operadores,probabilidades)
    valor= np.trace(np.matmul(suma, rho_inicial))
    return valor

print(valor_esperado_fm(probas))


###Obtener los efectos para observables####
outs=[1,1]
def efectos_fm(salidas, probabilidades):
    #Tomar primero los vectores propios de cada uno de los observables
    operadores_proyeccion=[]
    count=0
    for obser in observables:
        operador=np.zeros([len(obser.eigenvalues),len(obser.eigenvalues)])
        for i in range(len(obser.eigenvalues)):
            if obser.eigenvalues[i]==salidas[count]:
                #para degenerados deberíamos sumar los de proyeccion
                operador=np.add(operador,proyector(obser.eigenvectors[i]))
        operadores_proyeccion.append(operador)
        count+=1
    perm=permutations(operadores_proyeccion)
    suma=0*ft.reduce(np.kron,operadores_proyeccion)
    count=0
    for i in list(perm):
        operador_permutado=ft.reduce(np.kron,i)
        suma= np.add(suma,probabilidades[count]*operador_permutado)
        count+=1 

    return suma
print(efectos_fm(outs,probas))
####Mapeo de probabilidades###################################################
def mapeo_fm(salidas,estado,probabilidades):
    '''Realizar un mapeo de probabilidades, toma como entrada, 
    la salida, el estado inicial y la probabilidad de intercambio de partículas.
    Devuelve la probabilidad de obtener esa salida en una medición difusa.'''
    probabilidad= np.trace(np.matmul(efectos_fm(salidas,probabilidades), estado))
    return probabilidad

#print(mapeo_fm(outs,probas))

##############Obtener el estado posterior a la medición.
def estado_final(salidas,estado_inicial, probabilidades):
    '''Realizar un mapeo de estados, toma como entrada, 
    la salida, el estado inicial y la probabilidad de intercambio de partículas.
    Devuelve el estado posterior dado la salida en una medición difusa.'''
    #Sacar la raiz cuadrada de cada efecto dependiendo la salida
    eigenvalues, eigenvectors=eigh(efectos_fm(salidas,probabilidades))
    eigenvalues = np.abs(eigenvalues).round(5)
    #Sacarle la raiz a los valores propios
    eigenvalues_nuevos= np.sqrt(eigenvalues)
    #Crear el operador de kraus
    kraus_operator=eigenvectors.dot(np.diag(eigenvalues_nuevos)).dot(eigenvectors.T.conj())
    kraus_operator=kraus_operator.round(5)
    #Crear el estado final
    estadoFinal= kraus_operator.dot(estado_inicial).dot(kraus_operator.T.conj())
    return estadoFinal/np.trace(estadoFinal)
print(estado_final(outs,rho_inicial,probas))
#a2l.to_ltx(estado_final(outs,rho_inicial,probas), frmt = '{:6.4f}', arraytype = 'pmatrix')
######################################################################################
#####Gráficas#####################################

def graficar_distribucion(probabilidades, ax):
    ##data
 
    ##lista de todos las listas con los valores propios de los observables
    eigenvalues=[obs.eigenvalues for obs in observables]
    #La primera lista de valores propios la volvemos una lista de listas
    eigenvalues[0]=obs1.eigenvalues.reshape(len(obs1.eigenvalues),1)
    #Hacemos recursiva la funcion de cross_list para obtener una lista que tenga todas las listas con  las posibles salidas.
    all_outs=ft.reduce(cross_lists,eigenvalues)

    X=[]#el eje X
    #Con este ciclo multiplicamos los elementos de cada lista, es obtener los eigenvalores (aunque esten repetidos.)
    for lista in all_outs:
        multiplicacion=np.prod(lista)
        X.append(multiplicacion)

    Y=[]# el arreglo del eje Y
    #Con este ciclo obtenemos cada uno de los mapeos para cada posible salida en el eje X
    for lista in all_outs:
        mapeo= mapeo_fm(lista,rho_inicial,probabilidades)
        Y.append(mapeo)
    # Esto es para graficar aunque el observable este degenerado.
    nueva_X=set(X)
    nueva_Y=[]
    for i in nueva_X:
        #crear lista de indices donde están los repetidos
        indices=np.where(X == i)[0]
        #inicializar el valor de la probabilidad
        prob=0
        for j in indices:
            prob+=Y[j]
        nueva_Y.append(prob)
    #print(nueva_X,nueva_Y)
    def Labels(probabilidades):
        s=''
        for i in range(len(probabilidades)):
            valor=i+1
            string= '\n $p_{%s}=%s$,'%(valor,round(probabilidades[i],3))
            s+=string
        return s
    
    

    ######
    valorEsperado= str(round(valor_esperado_fm(probabilidades),3))
    ax.stem(np.array(list(nueva_X)), np.array(nueva_Y),  linefmt='-', markerfmt='black', basefmt='black')
    ax.vlines(x = valor_esperado_fm(probabilidades), ymin = 0, ymax = 1,     colors = 'purple', label='Valor Esperado ='+valorEsperado,
           ls='--')
    #ax.text(valor_esperado_fm(probabilidades), 0.99, valorEsperado , color='r', ha='right', va='top', rotation=90)
    props = dict(boxstyle='round', facecolor='grey', alpha=0.15)
    #ax.text(1.03, 0.98, Labels(probabilidades), transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

    #axis settings  
    ax.set_ylim([0, 1])
    ax.grid(linewidth=0.2)
    ax.set_ylabel("Probabilidad de obtener la salida")
    ax.set_xlabel('Salidas de la medición')
    for x,y in zip(np.array(list(nueva_X)),nueva_Y): 
        #ax.annotate(y,(x,y-4))
        ax.legend(loc='upper center')
        ax.annotate(str(round(y,3)), xy=(x,y), xytext=(0,5), textcoords='offset points',ha='center')

    return 0

def mostrar_graficas(ncol=2, nrow=3):
    if ncol>1 and nrow>1:
        fig, axes = plt.subplots(ncol, nrow, figsize=(nrow*5, ncol*5))
        for i, axe in enumerate(axes.flat):
            proba=generar_probabilidades(np.math.factorial(n))
            graficar_distribucion(proba, ax=axe)
        plt.show()
    else:
        graficar_distribucion(p=0.25,ax=plt.subplot())
        plt.show()

#mostrar_graficas(2,3)



##################Instrumento###########################################


#############PRIMER INSTRUMENTO########################################

def sis_clasico_1ins(salidas):
    operadores_proyeccion=[]
    count=0
    for obser in observables:
        operador=np.array([[0 for j in range(len(obser.eigenvalues))] for k in range(len(obser.eigenvalues))])
        for i in range(len(obser.eigenvectors)):
            if obser.eigenvalues[i]==salidas[count]:
                operador=np.add(operador,proyector(obser.eigenvectors[i]))
        operadores_proyeccion.append(operador)
        count+=1
    operador_proyector=ft.reduce(np.kron,operadores_proyeccion)
    return operador_proyector

def sis_cuantico_1ins(estados,salidas,probabilidades):
    proy=sis_clasico_1ins(salidas)
    canal=np.matmul(proy, np.matmul(fuzzy_operator(estados,probabilidades),proy))
    return canal



def primer_instrumento(estados, probabilidades):
    '''Replica el instrumento cuantico dado por el ensamble de un sistema clasico
    y uno cuantico'''
    instrumento=np.array([[0 for i in range(len(obs1.obs)**n)] for i in range(len(obs1.obs)**n)])
    instrumento=np.kron(instrumento,instrumento)

    eigenvalues=[obs.eigenvalues for obs in observables]
    #La primera lista la volvemos una lista de listas
    eigenvalues[0]=obs1.eigenvalues.reshape(len(obs1.eigenvalues),1)
    #Hacemos recursiva la funcion de cross_list para obtener una lista que tenga todas las listas con  las posibles salidas.
    all_outs=ft.reduce(cross_lists,eigenvalues)

    for salidas in all_outs:
        selectivo= np.kron(sis_clasico_1ins(salidas), sis_cuantico_1ins(estados,salidas,probabilidades))
        instrumento=np.add(instrumento,selectivo)
    return instrumento

#print(primer_instrumento(states, probas))
#######################################
