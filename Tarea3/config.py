#En este documento se crearan todas las funciones a utilizar en el programa principal
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from numba import njit

#La longitud reducida esta dado por la función
#L_bin(concentración reducida, número de partículas)
@njit
def L_bi(n,N,dim):
    if dim==2:
        return (np.sqrt(N/n))
    else:
        return ((N/n)**(1./3))
    
#Creamos una función que nos diga si dos circulos se intersectan
@njit
def intersec(x0, y0, z0, x1, y1, z1, r):
    d = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2)
    if d <= 2*r:
        return 0
    else:
        return 1

#Esta función es general para 2 o 3 dimensiones
@njit
def caja(s,n,N,dim):
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    rd.seed(s)
    
    #Llamamos a la funciones declaradas para calcular la longitud, Área/Volumen y Radio
    L = L_bi(n,N,dim)/2.
    r = 0.5
    
    #Calculamos el valor de la primera partícula de forma directa ya que no hay otra posición con cual compararla
    x[0] = rd.uniform(-L+0.5,L-0.5)
    y[0] = rd.uniform(-L+0.5,L-0.5)
    #Incorporamos un condicional para el caso de 2 o 3 dimensiones
    if dim==3:
        z[0] = rd.uniform(-L+0.5,L-0.5)
    i=1
    #Calculamos la posición de las demas partículas
    while i < N:
        k = 0
        #kk = 0
        x[i] = rd.uniform(-L+0.5,L-0.5)
        y[i] = rd.uniform(-L+0.5,L-0.5)
        if dim==3:
            z[i] = rd.uniform(-L+0.5,L-0.5)
            
        #En este ciclo comparamos la posición prueba de nuestra partícula con las posiciones de las partículas anteriores
        for j in range(i):
            #Llamamos a la función que nos calcula la intersección entre dos partículas, si hay intersección nos regresa un cero
            #Cada IF revisa si existe una intersección con la partícula y las vecinas, incluyendo a las que se encuentran en las celdas imagen
            if ( intersec( x[i], y[i], z[i], x[j], y[j], z[j], r ) == 0):
                k = k +1
        
        if k == 0:                    
        #Si el valor no se repitio guardamos el valor prueba en nuestra lista
        #Aumentamos el valor de este contador para pasar a la siguiente partícula
            i=i+1
        elif k != 0:                    
            s=s+0.1
            rd.seed(s)
        #else:
        #    kk = kk + 1
            
        #if kk == 1.0E+5:
        #    sys.exit()
        
    if dim==2:
        v = np.column_stack((x,y))
        return v
        #return x, y
    elif dim==3:
        #return x, y, z
        v = np.column_stack((x,y,z))
        return v


@njit
def cajaP(s,n,N):
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    rd.seed(s)
    
    #Llamamos a la funciones declaradas para calcular la longitud, Área/Volumen y Radio
    L = L_bi(n,N,dim)/2.
    r = 0.5
    
    #Calculamos el valor de la primera partícula de forma directa ya que no hay otra posición con cual compararla
    x[0] = rd.uniform(-L+0.5,L-0.5)
    y[0] = rd.uniform(-L+0.5,L-0.5)
    z[0] = rd.uniform(-L+0.5,L-0.5)
    print(0)     #####
    i=1
    #Calculamos la posición de las demas partículas
    while i < N:
        k = 0
        #kk = 0
        x[i] = rd.uniform(-L+0.5,L-0.5)
        y[i] = rd.uniform(-L+0.5,L-0.5)
        if dim==3:
            z[i] = rd.uniform(-L+0.5,L-0.5)
            
        #En este ciclo comparamos la posición prueba de nuestra partícula con las posiciones de las partículas anteriores
        for j in range(i):
            #Llamamos a la función que nos calcula la intersección entre dos partículas, si hay intersección nos regresa un cero
            #Cada IF revisa si existe una intersección con la partícula y las vecinas, incluyendo a las que se encuentran en las celdas imagen
            if ( intersec( x[i], y[i], z[i], x[j], y[j], z[j], r ) == 0):
                k = k +1
        
        if k == 0:                    
        #Si el valor no se repitio guardamos el valor prueba en nuestra lista
        #Aumentamos el valor de este contador para pasar a la siguiente partícula
            if (i+1)%100==0:
                print(i+1) #imprimimos en pantalla en que partícula vamos BORRAR
            i=i+1
        elif k != 0:                    
            s=s+0.1
            rd.seed(s)
        
    v = np.column_stack((x,y,z))
    return v



#Definimos una función que nos genere circulo
def circl(x,y,n,N,dim,i, clr):
    circle = plt.Circle( (x[i], y[i] ), 0.5, color = clr) #fill=Flase
    return circle

#función que calcula la energía de la configuración
@njit
def sumup(N,L,rCut, x, y):
    V=0.
    for i in range(N-1):
        rxi = x[i]
        ryi = y[i]
        
        for j in range(i+1,N):
            rxij = rxi - x[j]
            ryij = ryi - y[j]
            #condición de imagen mínima
            rxij = rxij - L * round(rxij/L)
            ryij = ryij - L * round(ryij/L)
            
            rijsq = np.sqrt(rxij*rxij + ryij*ryij)
            
            if rijsq < rCut:
                if rijsq <= 1.:
                    Vij = 1.0E+10
                else:
                    Vij = 0
                V = V + Vij
    return V#, rxi, ryi

#función que calcula la energía de la configuración en 3d
@njit
def sumup3d(N,L,rCut, x, y, z):
    V=0.
    for i in range(N-1):
        rxi = x[i]
        ryi = y[i]
        rzi = z[i]
        
        for j in range(i+1,N):
            rxij = rxi - x[j]
            ryij = ryi - y[j]
            rzij = rzi - z[j]
            #condición de imagen mínima
            rxij = rxij - L * round(rxij/L)
            ryij = ryij - L * round(ryij/L)
            rzij = rzij - L * round(rzij/L)
            
            rijsq = np.sqrt(rxij*rxij + ryij*ryij + rzij*rzij)
                               
            if rijsq < rCut:
                if rijsq <= 1.:
                    Vij = np.inf
                elif rijsq > 1. and rijsq < 1.25 :
                    Vij = -1.0E+10
                elif rijsq >= 1.25:
                    Vij = 0
                    
                V = V + Vij
    return V

#función que obtine la energía de la j-ésima partícula
@njit
def energia(x,y,rxi, ryi, N, L, rCut, j):
    V = 0.
    for i in range(N):
        if j != i:
            rxij = rxi - x[i]
            ryij = ryi - y[i]
            #condición de imagen mínima
            rxij = rxij - L * round(rxij/L)
            ryij = ryij - L * round(ryij/L)
            
            rijsq = np.sqrt(rxij*rxij + ryij*ryij)
            
            if rijsq < rCut:
                if rijsq <= 1.:
                    Vij = 1.0E+10
                else:
                    Vij = 0.
                V = V + Vij
    return V

#función que obtine la energía de la j-ésima partícula 3d
@njit
def energia3d(x,y,z, rxi, ryi, rzi, N, L, rCut, j):
    V = 0.
    for i in range(N):
        if j != i:
            rxij = rxi - x[i]
            ryij = ryi - y[i]
            rzij = rzi - z[i]
            #condición de imagen mínima
            rxij = rxij - L * round(rxij/L)
            ryij = ryij - L * round(ryij/L)
            rzij = rzij - L * round(rzij/L)
            
            rijsq = np.sqrt(rxij*rxij + ryij*ryij + rzij*rzij)
            
            if rijsq < rCut:
                if rijsq <= 1.:
                    Vij = np.inf
                elif rijsq > 1. and rijsq < 1.25 :
                    Vij = -1.0E+10
                elif rijsq >= 1.25:
                    Vij = 0
                V = V + Vij
    return V

#Creamos una función que nos calcule las nuevas posiciones de nuestras partículas
@njit
def movimiento(x, y, N, nStep, L, rCut, drMax, iRatio, iPrint, cc, V, iTraza, Vlrc):
    acatma = 0.
    xTraza = np.zeros(nStep)
    yTraza = np.zeros(nStep)
    Vn = np.zeros(nStep)

    for i in range(nStep):
        for j in range(N):
            xOld = x[j]
            yOld = y[j]
            Vold = energia(x,y,xOld, yOld, N, L, rCut, j)
            #print(j,Vold)
            xNew = xOld + ((2.*rd.uniform(0,1) - 1.) *drMax)
            yNew = yOld + ((2.*rd.uniform(0,1) - 1.) *drMax)
            Vnew = energia(x,y,xNew, yNew, N, L, rCut, j)

            #condición de imagen mínima
            xNew = xNew - L * round(xNew/L)
            yNew = yNew - L * round(yNew/L)

            dV = Vnew - Vold

            if dV < 75.:
                if dV <= 0. :
                    V = V + dV
                    x[j] = xNew
                    y[j] = yNew

                    acatma = acatma + 1
                elif np.exp(-dV) > rd.uniform(0,1):
                    V = V + dV
                    x[j] = xNew
                    y[j] = yNew

                    acatma = acatma + 1.

            if j==iTraza:
                xTraza[i] = x[j]
                yTraza[i] = y[j]
                #k = k +1

        Vn = (V + Vlrc)/float(N)

        if i % iRatio == 0:
            ratio = acatma/float((N*iRatio))

            if ratio > cc:
                drMax = drMax*1.05
            else:
                drMax = drMax*0.95
            acatma = 0.

        if i % iPrint == 0:
            print (i, ratio, drMax, Vn,"\n")

    return x, y, xTraza, yTraza

#Creamos una función que nos calcule las nuevas posiciones de nuestras partículas en 3d
@njit
def movimiento3d(x, y, z, N, nStep, L, rCut, drMax, iRatio, iPrint, cc, V, iTraza, Vlrc):
    acatma = 0.
    xTraza = np.zeros(nStep)
    yTraza = np.zeros(nStep)
    zTraza = np.zeros(nStep)
    vTraza = np.zeros(nStep+1)
    vTraza[0] = V
    #Vn = np.zeros(nStep+1)
    #Vn[0] = V

    for i in range(nStep):
        for j in range(N):
            xOld = x[j]
            yOld = y[j]
            zOld = z[j]
            Vold = energia3d(x,y,z,xOld, yOld,zOld, N, L, rCut, j)
            #print(j,Vold)
            xNew = xOld + ((2.*rd.uniform(0,1) - 1.) *drMax)
            yNew = yOld + ((2.*rd.uniform(0,1) - 1.) *drMax)
            zNew = zOld + ((2.*rd.uniform(0,1) - 1.) *drMax)
            Vnew = energia3d(x,y,z,xNew, yNew,zNew, N, L, rCut, j)

            #condición de imagen mínima
            xNew = xNew - L * round(xNew/L)
            yNew = yNew - L * round(yNew/L)
            zNew = zNew - L * round(zNew/L)


            dV = Vnew - Vold

            if dV < 75.:
                if dV <= 0. :
                    V = V + dV
                    x[j] = xNew
                    y[j] = yNew
                    z[j] = zNew

                    acatma = acatma + 1
                elif np.exp(-dV) > rd.uniform(0,1):
                    V = V + dV
                    x[j] = xNew
                    y[j] = yNew
                    z[j] = zNew

                    acatma = acatma + 1.

            if j==iTraza:
                xTraza[i] = xNew
                yTraza[i] = yNew
                zTraza[i] = zNew

        Vn = (V + Vlrc)/float(N)
            

        if i % iRatio == 0:
            ratio = acatma/float((N*iRatio))

            if ratio > cc:
                drMax = drMax*1.05
            else:
                drMax = drMax*0.95
            acatma = 0.

        if i % iPrint == 0:
            print (i, ratio, drMax, Vn,"\n")
        vTraza[i+1] = Vn

    return x, y, z, xTraza, yTraza, zTraza, vTraza


















