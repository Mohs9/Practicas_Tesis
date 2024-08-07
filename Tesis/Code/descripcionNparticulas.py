#####################################################################
'''
En este programa se busca exhibir el mapa de probabilidades y el estado
resultante luego de la medición, a partir de un estado inicial y un observable 
dado en un sistema de varias partículas.

Para ejecutar las rutinas principales de manera adecuada del programa,
en primer lugar, se debe brindar el número de partículas con las que se desea
trabajar en la variable N. Luego, el observable de forma matricial. Si el 
observable es factorizable se deben brindar cada uno de los operadores que 
lo conforman. Asimismo el estado inicial debe  proporcionarse en forma 
matricial, si este factorizable se debe dar cada uno de los factores que lo 
conforman.

Posteriormente, para obtener el mapeo de probabilidades debe ejecutarse
la rutina "mapeo_fm(salidas,estado,probabilidades )", el cual necesita la 
lista que contenga las salidas de cada uno de la medición de los observables 
y la lista que contenga las probabilidades de intercambio. Para obtener el 
estado posterior a la medición se debe ejecutar la función 
"estado_final(salidas, probabilidades)".

Adicionalmente es posible graficar la distribución de probabilidad, con la 
función de mostrar_graficas(1,1). 
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

print(cross_lists([[1],[-1]],[1,-1]))


##número de partículas
N=2

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

####Si el operador es no factorizable, puede sustituirse en la variable OBS
OBS=ft.reduce(np.kron,operadores)



## Escribir el estado inicial dado que se quiere medir
state1=[[1/3,0],[0,2/3]] 
state2=[[0.5,0.5],[0.5,0.5]]
state3=[[1,0],[0,0]]
states=[state1,state2]
rho_inicial=ft.reduce(np.kron,states)

###DESCRIPCIÓN##################################################################

######PRIMERO DEBEN GENERARSE LOS OPERADORES DE PERMUTACIÓN3#################

####definimos la función permutacion que servira para construir las matrices de permutación utilizando la base computacional.
def PERM(lista,no_fila):
    '''Esta función recibe una lista que tiene los números que indican como se permutaran los valores y un número entero que en binario representa un bra de la base computacional. Devuelve un entero que en binario representa un ket en la base computacional'''
    #Usar un if para tomar un número binario con N dígitos
    if len(bin(no_fila)[2:])<=N:
        #Convertimos el numero de fila a binario
        numero_binario=str(bin(no_fila)[2:].zfill(N))
    #Escribir el número en una lista para poder cambiar su orden
    ket_binario=list(numero_binario)
    #Inicializar el nuevo ket al que se le aplica la permutación
    nueva_ket=[0 for i in range(N)]
    #Realizar la permutacion indicada por la lista.
    for i in range(N):
      nueva_ket[lista[i]]=ket_binario[i]
    #Inicializamos el número en base 10 que indicara el numero de columna
    no_columna= 0
    #Convertimos a base 10
    for i in range(N):
        no_columna+=int(nueva_ket[i])*(2**((N-1)-i))
    return no_columna

#print(PERM((1,0,2),7))


def operador_permutacion(lista):
    #Inicializar una matriz
    Pi=[[0 for i in range(2**N)] for i in range(2**N)]
    #Colocar unos en la filas y columnas correspondientes a lo que indica la función perm
    for fila in range(2**N):
        columna=PERM(lista,fila)
        Pi[fila][columna]=1
    return np.array(Pi)

#print(matrices_permutacion((1,0,2)))

lista_original=list(range(N))
#Crear las diferentes permutaciones
diferentes_ordenes=list(permutations(lista_original))
todos_operadores_permutacion=[] #El array que contenga los N! operadores de permutación
for lista in diferentes_ordenes:
    todos_operadores_permutacion.append(operador_permutacion(lista)) #Agregarlos todos.

print(todos_operadores_permutacion)
##Una lista de probabilidades aleatorias (pueden cambiar)
def generar_probabilidades(m):
    # Generar una lista de n números aleatorios
    probabilidades = [random.random() for _ in range(m)]
    # Calcular la suma de los números aleatorios
    suma_total = sum(probabilidades)
    # Normalizar los números para que sumen 1
    probabilidades = [p/suma_total for p in probabilidades]
    return probabilidades

probas=generar_probabilidades(np.math.factorial(N))
#prueba
probas=[0.25,0.75]


###Definir el operador difuso##### 
def fuzzy_operator(estado,probabilidades):
    '''Aplicar el operador difuso a un estado inicial devuelve otro operador'''
    suma=np.zeros([2**N,2**N])
    count=0
    for operador in todos_operadores_permutacion:
        termino= np.matmul(np.matmul(operador, estado), operador.T)
        suma= np.add(suma,probabilidades[count]*termino)
        count+=1 
    return suma

#print(fuzzy_operator(rho_inicial,probas))




###Definir el valor esperado###
def valor_esperado_fm(probabilidades):
    '''Valor esperado del observable, devuelve un número'''
    suma=fuzzy_operator(rho_inicial,probabilidades)
    valor= np.trace(np.matmul(suma, OBS))
    return valor

#print(valor_esperado_fm(probas))


###Obtener los efectos para observables FACTORIZABLES####
outs=[1,1]
def efectos_fm(salidas, probabilidades):
    #Tomar primero los vectores propios de cada uno de los observables
    operadores_proyeccion=[]
    count=0
    #Se ejecuta un ciclo for que recorra los observables de cada partícula
    for obser in observables:
        operador=np.zeros([len(obser.eigenvalues),len(obser.eigenvalues)])
        #El ciclo recorre todos los eigenvalues y busca el que se la salida
        for i in range(len(obser.eigenvalues)):
            if obser.eigenvalues[i].round(3)==salidas[count]:
                #En este caso convertimos el vector propio correspondiente a la
                #salida en un operador proyector
                operador=np.add(operador,proyector(obser.eigenvectors[i]))
        operadores_proyeccion.append(operador)
        count+=1
    #Ejecuta el producto de kronecker de los operadores de proyeccion de cada
    #operador
    operador_proyeccion=ft.reduce(np.kron,operadores_proyeccion)
    suma=0*ft.reduce(np.kron,operadores_proyeccion)
    count=0
    #Se le aplica los operadores de permutacion
    for operador in todos_operadores_permutacion:
        termino=np.matmul(np.matmul(operador.T,operador_proyeccion),operador)
        suma= np.add(suma,probabilidades[count]*termino)
        count+=1 

    return suma
print(efectos_fm(outs,probas))
#a2l.to_ltx(efectos_fm(outs,probas), frmt = '{:6.4f}', arraytype = 'pmatrix')
####Mapeo de probabilidades###################################################
def mapeo_fm(salidas,probabilidades):
    '''Realizar un mapeo de probabilidades, toma como entrada, 
    la salida, el estado inicial y la probabilidad de intercambio de partículas.
    Devuelve la probabilidad de obtener esa salida en una medición difusa.'''
    probabilidad= np.trace(np.matmul(efectos_fm(salidas,probabilidades), rho_inicial))
    return probabilidad

#print(mapeo_fm(outs,probas))

##############Obtener el estado posterior a la medición.
def estado_final(salidas, probabilidades):
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
    estadoFinal= kraus_operator.dot(rho_inicial).dot(kraus_operator.T.conj())
    return estadoFinal/np.trace(estadoFinal)
#print(estado_final([1,1,1],probas))
#a2l.to_ltx(estado_final(outs,rho_inicial,probas), frmt = '{:6.4f}', arraytype = 'pmatrix')
######################################################################################



#####Gráficas para observables factorizables #####################################
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
            proba=generar_probabilidades(np.math.factorial(N))
            graficar_distribucion(proba, ax=axe)
        plt.show()
    else:
        graficar_distribucion(probas,ax=plt.subplot())
        plt.show()

#mostrar_graficas(1,1)











##################Instrumento###########################################


#############PRIMER INSTRUMENTO operadores factorizables ########################################

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

def sis_cuantico_1ins(estado,salidas,probabilidades):
    proy=sis_clasico_1ins(salidas)
    canal=np.matmul(proy, np.matmul(fuzzy_operator(estado,probabilidades),proy))
    return canal



def primer_instrumento(estado, probabilidades):
    '''Replica el instrumento cuantico dado por el ensamble de un sistema clasico
    y uno cuantico'''
    instrumento=np.array([[0 for i in range(len(obs1.obs)**N)] for i in range(len(obs1.obs)**N)])
    instrumento=np.kron(instrumento,instrumento)

    eigenvalues=[obs.eigenvalues for obs in observables]
    #La primera lista la volvemos una lista de listas
    eigenvalues[0]=obs1.eigenvalues.reshape(len(obs1.eigenvalues),1)
    #Hacemos recursiva la funcion de cross_list para obtener una lista que tenga todas las listas con  las posibles salidas.
    all_outs=ft.reduce(cross_lists,eigenvalues)

    for salidas in all_outs:
        selectivo= np.kron(sis_clasico_1ins(salidas), sis_cuantico_1ins(estado,salidas,probabilidades))
        instrumento=np.add(instrumento,selectivo)
    return instrumento

#print(primer_instrumento(states, probas))
#######################################





###############Efectos para observables NO factorizables ############################
def efectos_no_factorizable(salida, probabilidades):
    #Tomar primero los vectores propios del observable
    operador_proyector=np.zeros([2**N,2**N])
    eigenvalues, eigenvectors= eigh(OBS) 
    for i in range(len(eigenvalues)):
        if salida==eigenvalues[i].round(3):
            #para degenerados deberíamos sumar los de proyeccion
            operador_proyector=np.add(operador_proyector,proyector(eigenvectors[i]))
    suma=0*np.zeros([2**N,2**N])
    count=0
    for operador in todos_operadores_permutacion:
        termino=np.matmul(np.matmul(operador.T,operador_proyector),operador)
        suma= np.add(suma,probabilidades[count]*termino)
        count+=1 
    return suma

####Mapeo de probabilidades para observables NO FACTORIZABLES###################################################
def mapeo_no_factorizables(salida,estado,probabilidades):
    '''Realizar un mapeo de probabilidades, toma como entrada, 
    la salida, el estado inicial y la probabilidad de intercambio de partículas.
    Devuelve la probabilidad de obtener esa salida en una medición difusa.'''
    probabilidad= np.trace(np.matmul(efectos_no_factorizable(salida,probabilidades), estado))
    return probabilidad

##############Obtener el estado posterior a la medición cuando el observable no es factorizable.
def estado_final_nf(salida,estado_inicial, probabilidades):
    '''Realizar un mapeo de estados, toma como entrada, 
    la salida, el estado inicial y la probabilidad de intercambio de partículas.
    Devuelve el estado posterior dado la salida en una medición difusa.'''
    #Sacar la raiz cuadrada de cada efecto dependiendo la salida
    eigenvalues, eigenvectors=eigh(efectos_no_factorizable(salida,probabilidades))
    eigenvalues = np.abs(eigenvalues).round(5)
    #Sacarle la raiz a los valores propios
    eigenvalues_nuevos= np.sqrt(eigenvalues)
    #Crear el operador de kraus
    kraus_operator=eigenvectors.dot(np.diag(eigenvalues_nuevos)).dot(eigenvectors.T.conj())
    kraus_operator=kraus_operator.round(5)
    #Crear el estado final
    estadoFinal= kraus_operator.dot(estado_inicial).dot(kraus_operator.T.conj())
    return estadoFinal/np.trace(estadoFinal)
#print(estado_final_nf(-1,rho_inicial,probas))


##################Instrumento###########################################


#############PRIMER INSTRUMENTO operadores factorizables ########################################

def sis_clasico_1ins_nf(salida):
    operador_proyeccion=np.zeros([2**N,2**N])
    eigenvalues, eigenvectors=eigh(OBS)
    for i in range(len(eigenvalues)):
        if eigenvalues[i].round(3)==salida:
            operador_proyeccion=np.add(operador_proyeccion,proyector(eigenvectors[i]))
    return operador_proyeccion

def sis_cuantico_1ins_nf(estado,salidas,probabilidades):
    proy=sis_clasico_1ins(salidas)
    canal=np.matmul(proy, np.matmul(fuzzy_operator(estado,probabilidades),proy))
    return canal



def primer_instrumento_nf(estado, probabilidades):
    '''Replica el instrumento cuantico dado por el ensamble de un sistema clasico
    y uno cuantico'''
    instrumento=np.array([[0 for i in range(len(obs1.obs)**N)] for i in range(len(obs1.obs)**N)])
    instrumento=np.kron(instrumento,instrumento)
    eigenvalues, eigenvectors=eigh(OBS)
    for salida in eigenvalues:
        selectivo= np.kron(sis_clasico_1ins_nf(salida), sis_cuantico_1ins_nf(estado,salida,probabilidades))
        instrumento=np.add(instrumento,selectivo)
    return instrumento

#print(primer_instrumento(states, probas))
#######################################
