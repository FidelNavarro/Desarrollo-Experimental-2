#En este documento se crearan todas las funciones a utilizar en el programa principal
import numpy as np
import random as rd
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
                elif rijsq > 1.0 :
                    #Vij= 0 #-5000
                    Vij = 580*(np.e)**(-0.149*rijsq) / rijsq
                    #Vij = 4*((rCut/rijsq)**12-(rCut/rijsq)**6)

                    
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
                elif rijsq > 1.0 :
                    #Vij = 0 #-5000
                    Vij = 580*(np.e)**(-0.149*rijsq) / rijsq
                    #Vij = 4*((rCut/rijsq)**12-(rCut/rijsq)**6)

                V = V + Vij
    return V

#Creamos una función que nos calcule las nuevas posiciones de nuestras partículas en 3d
@njit
def movimiento3d(x, y, z, N, nStep, L, rCut, drMax, iRatio, iPrint, cc, V, iTraza, Vlrc, nener):
    acatma = 0.
    #nn2 = 1000
    #nn3 = 3500
    ki2 = 0
    xTraza = np.zeros(nStep)
    yTraza = np.zeros(nStep)
    zTraza = np.zeros(nStep)
    vTraza = np.zeros(nStep)
    
    Cx = np.zeros((N, nStep))
    Cy = np.zeros((N, nStep))
    Cz = np.zeros((N, nStep))
    
    #vTraza[0] = V
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
            
            #condición de imagen mínima
            xNew = xNew - L * round(xNew/L)
            yNew = yNew - L * round(yNew/L)
            zNew = zNew - L * round(zNew/L)

            Vnew = energia3d(x,y,z,xNew, yNew,zNew, N, L, rCut, j)


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
                
            if i > nener:
                Cx[j][ki2] = x[j]
                Cy[j][ki2] = y[j]
                Cz[j][ki2] = z[j]

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
        vTraza[i] = Vn
        
        if i > nener:
            ki2 = ki2 + 1
        #    for k in range(N):
        #        Cx[k, ki2] = x[k]
        #        Cy[k, ki2] = y[k]
        #        Cz[k, ki2] = z[k]
       

    return x, y, z, xTraza, yTraza, zTraza, vTraza, Cx, Cy, Cz, ki2


@njit
def gdr(Cx, Cy, Cz, rCut, L, nStep, n, N, ki2):
    #ntmax = ki2
    #nn2 = 1000
    #nn3 = 3500
    nhist = np.zeros(int(nStep))
    deltar = 0.05
    maxbin = int(L/2/deltar)
    
    
    for i in range(N):
        #print(i)
        for j in range(N):
            if i != j :
                for k in range(ki2):
                    xL0 = Cx[i,k]
                    xLT = Cx[j,k]
                    xL0T = xL0-xLT

                    yL0 = Cy[i,k]
                    yLT = Cy[j,k]
                    yL0T = yL0-yLT

                    zL0 = Cz[i,k]
                    zLT = Cz[j,k]
                    zL0T = zL0-zLT
                    
                    xL0T = xL0T-L*round(xL0T/L)
                    yL0T = yL0T-L*round(yL0T/L)
                    zL0T = zL0T-L*round(zL0T/L)
                    
                    R0T = np.sqrt(xL0T**2 + yL0T**2 + zL0T**2)
                    
                    nBin = int(R0T/deltar)+1
                    if nBin <= maxbin :
                        nhist[nBin] = nhist[nBin] + 1
                        
                    #print(nhist[nBin])
    
    c1 = 4.0*(np.pi)*n/3.0
    
    gdrTA = np.zeros(maxbin)
    rt = np.zeros(maxbin)
    for i in range(maxbin):
        rl = float(i-1)*deltar
        ru = rl + deltar
        rt[i] = rl + deltar/2.0
        c2 = c1*(ru**3 - rl**3)
        gdrTA[i] = float(nhist[i])/float(ki2)/float(N)/c2
        
        #print(i, rt[i], gdrTA[i], nhist[i])
        
    return rt, gdrTA

@njit
def pote(rt, gdr, n):
    #xig = np.zeros(len(rt))
    #xig = 0.0
    #isig = np.zeros(len(rt))
    for i in range(len(rt)):
        if rt[i] > (1.0-0.01) and rt[i] > (1.0 + 0.01):
            xog = gdr[i]
            isig = i
    
    print("ICONTACTO = ", isig, ", GDR DE CONTACTO = ", xog)
    
    #expresion exacta para la presion de un sistema de hs
    phs = n + 2.0 * np.pi * n * xog / 3.0
    
    print("rho = ", n, ", Phs = ", phs)
        
    








