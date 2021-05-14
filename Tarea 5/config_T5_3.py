#En este documento se crearan todas las funciones a utilizar en el programa principal
import numpy as np
import random as rd
from numba import njit

@njit
def gauss():
    a = rd.uniform(0,1)
    b = rd.uniform(0,1)
    R = np.sqrt(-2.*np.log(a))*np.cos(2.*np.pi*b)
    return R

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
    r=0.5
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

#Esta subrutina calcula la fuerza de interacción y la energía de la configuración para yukawa
@njit
def fuerzaYukawa(x,y,z,N,L,rCut):
    E = 0
    fx = np.zeros(N)
    fy = np.zeros(N)
    fz = np.zeros(N)
    
    for i in range(N-1):
        fxi = fx[i]
        fyi = fy[i]
        fzi = fz[i]
        
        for j in range(i+1, N):
            xij = x[i] - x[j]
            yij = y[i] - y[j]
            zij = z[i] - z[j]

            xij = xij - L*round(xij/L)
            yij = yij - L*round(yij/L)
            zij = zij - L*round(zij/L)

            rij = (xij**2 + yij**2 + zij**2)**0.5

            #if rij<=1.0 :
                #print(i,j)

            if rij < rCut and rij>1 :
                #Yukawa
                #zk = 0.149
                #A = 556
                #K = A*np.exp(zk)
                #U = np.exp(-zk*rij)
                #U2 = K*U*(zk*rij + 1.0)/rij**3
                #E = (K*U)/rij + E

                A = 556
                zk = 0.149
                K = A*np.exp(zk)
                E_pot = K*np.exp(-zk*rij)/rij
                E = E_pot + E
                U2 = E_pot*(zk*rij+1.)/rij**2

                fxij = xij*U2
                fyij = yij*U2
                fzij = zij*U2

                fxi = fxi + fxij
                fyi = fyi + fyij
                fzi = fzi + fzij
                
                fx[j] = fx[j] - fxij
                fy[j] = fy[j] - fyij
                fz[j] = fz[j] - fzij
        
        fx[i] = fxi
        fy[i] = fyi
        fz[i] = fzi
    
    return E, fx, fy, fz
    
#Creamos una función que nos calcule las nuevas posiciones de nuestras partículas en 3d
@njit
def movimientoYukawa(x,y,z,fx,fy,fz,L,N,rCut,nStep,dT,nener,iFrec1, iFrec2):
    k1 = 0
    k2 = 0
    var = np.sqrt(2.0*dT)
    
    V = np.zeros(nStep)
    
    Cx = np.zeros((N,nStep))
    Cy = np.zeros((N,nStep))
    Cz = np.zeros((N,nStep))
    
    Cxr = np.zeros((N,nStep))
    Cyr = np.zeros((N,nStep))
    Czr = np.zeros((N,nStep))
    
    xr = x#np.zeros(N)
    yr = z#np.zeros(N)
    zr = z#np.zeros(N)
    
    for l in range(nStep):
        for i in range(N):
            xRandom = float(gauss())
            yRandom = float(gauss())
            zRandom = float(gauss())
            
            x[i] = x[i] + fx[i]*dT + var*xRandom
            y[i] = y[i] + fy[i]*dT + var*yRandom
            z[i] = z[i] + fz[i]*dT + var*zRandom
            
            x[i] = x[i] - L*round(x[i]/L)
            y[i] = y[i] - L*round(y[i]/L)
            z[i] = z[i] - L*round(z[i]/L)
            
            xr[i] = xr[i] + fx[i]*dT + var*xRandom
            yr[i] = yr[i] + fy[i]*dT + var*yRandom
            zr[i] = zr[i] + fz[i]*dT + var*zRandom
            
            if l>nener and l%iFrec1==0.0 : 
            #for i in range(N):
                Cx[i][k1] = x[i]
                Cy[i][k1] = y[i]
                Cz[i][k1] = z[i]
                
            #k1 = k1 + 1
        
            if l>nener and l%iFrec2==0.0 : 
            #for i in range(N):
                Cxr[i][k2] = xr[i]
                Cyr[i][k2] = yr[i]
                Czr[i][k2] = zr[i]
            
        if l>nener and l%iFrec1==0.0 :
            k1 = k1 + 1
        if l>nener and l%iFrec2==0.0 : 
            k2 = k2 + 1
            
        fx = np.zeros(N)
        fy = np.zeros(N)
        fz = np.zeros(N)
        E, fx, fy, fz = fuerzaYukawa(x, y, z, N, L, rCut)
                
        V[l] = E
    
    return x,y,z,Cx,Cy,Cz,Cxr,Cyr,Czr,V,k1,k2





@njit
def gdr(Cx, Cy, Cz, rCut, L, nStep, n, N, ki2):

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
                        
                    #print(i,j,k)
    
    c1 = 4.0*(np.pi)*n/3.0
    
    gdrTA = np.zeros(maxbin)
    rt = np.zeros(maxbin)
    for i in range(maxbin):
        rl = float(i)*deltar
        ru = rl + deltar
        rt[i] = rl + deltar/2.0
        c2 = c1*(ru**3 - rl**3)
        gdrTA[i] = float(nhist[i])/float(ki2)/float(N)/c2
        
        #print(i, rt[i], gdrTA[i], nhist[i])
        
    return rt, gdrTA

#@njit
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
    

@njit
def wdt(Cx, Cy, Cz, N, ki, dT, iFrec):
    tim = dT*float(iFrec)
    
    dif = np.zeros(ki)
    time = np.zeros(ki)
    Wt = np.zeros(ki)
    
    for k in range(1,ki):
        #ntmax = ki-k
        R = 0
    
        for i in range(N):
            #for j in range(k):
            
            r0 = ( ( Cx[i][k-1] )**2 + ( Cy[i][0] )**2 + ( Cz[i][k-1] )**2 )**0.5
            r = ( ( Cx[i][k] )**2 + ( Cy[i][k] )**2 + ( Cz[i][k] )**2 )**0.5
            R = (r - r0)**2 + R
        
        time[k] = tim*float(k)
        Wt[k] = (R)/float(N)#/6.0
        dif[k] = Wt[k]/time[k]
    
    return time, Wt, dif

@njit
def wdt2(Cx,Cy,Cz,N,ki,dT,iFrec):
    tim = dT*float(iFrec)
    
    dif = np.zeros(ki)
    time = np.zeros(ki)
    Wt = np.zeros(ki)
    
    for k in range(1,ki):
        #ntmax = ki-k
        R = 0
    
        for i in range(N):
            
            x = Cx[i][k] - Cx[i][k-1]
            y = Cy[i][k] - Cy[i][k-1]
            z = Cz[i][k] - Cz[i][k-1]
            
            R = x**2 + y**2 + z**2 + R
        
        time[k] = tim*float(k)
        Wt[k] = (R)/float(N)
        dif[k] = Wt[k]/time[k]
    
    return time, Wt, dif

@njit
def wdt3(Cxr, Cyr, Czr, N, ki2, dT, iFrec):
    tim = float(iFrec)*dT
    wt = np.zeros(ki2-1)
    time = np.zeros(ki2-1)
    dif = np.zeros(ki2-1)
    
    for i in range(ki2-1):
        ntmax = ki2 - i
        wtx = 0.0
        wty = 0.0
        wtz = 0.0
        
        for j in range(N):
            for h in range(ntmax):
                wtx = wtx + ( Cxr[j][h+i] - Cxr[j][h] )**2
                wty = wty + ( Cyr[j][h+i] - Cyr[j][h] )**2
                wtz = wtz + ( Czr[j][h+i] - Czr[j][h] )**2
                
        time[i] = tim*float(i+1)
        wt[i] = (wtx + wty + wtz)/float(ntmax)
        dif[i] = wt[i]/time[i]
    
    return time, wt, dif


@njit
def fuerzaGauss(x,y,z,N,L,rCut,n):
    E = 0
    fx = np.zeros(N)
    fy = np.zeros(N)
    fz = np.zeros(N)
    
    for i in range(N-1):
        fxi = fx[i]
        fyi = fy[i]
        fzi = fz[i]
        
        for j in range(i+1, N):
            xij = x[i] - x[j]
            yij = y[i] - y[j]
            zij = z[i] - z[j]

            xij = xij - L*round(xij/L)
            yij = yij - L*round(yij/L)
            zij = zij - L*round(zij/L)

            rij = (xij**2 + yij**2 + zij**2)**0.5

            #if rij<=1.0 :
                #print(i,j)

            if rij < rCut :
                #Gauss
                eps = 100
                k = 1.0
                rho = 1 #L*(n/float(N))**(1./3.)
                
                E_pot = eps*np.exp(-k*(rij/rho)**2)
                E = E_pot + E
                U2 = E_pot*(2.0*k)/rho**2

                fxij = xij*U2
                fyij = yij*U2
                fzij = zij*U2

                fxi = fxi + fxij
                fyi = fyi + fyij
                fzi = fzi + fzij
                
                fx[j] = fx[j] - fxij
                fy[j] = fy[j] - fyij
                fz[j] = fz[j] - fzij
        
        fx[i] = fxi
        fy[i] = fyi
        fz[i] = fzi
    
    return E, fx, fy, fz

@njit
def movimientoGauss(x,y,z,fx,fy,fz,L,N,rCut,nStep,dT,nener,iFrec1, iFrec2, n):
    k1 = 0
    k2 = 0
    var = np.sqrt(2.0*dT)
    
    V = np.zeros(nStep)
    
    Cx = np.zeros((N,nStep))
    Cy = np.zeros((N,nStep))
    Cz = np.zeros((N,nStep))
    
    Cxr = np.zeros((N,nStep))
    Cyr = np.zeros((N,nStep))
    Czr = np.zeros((N,nStep))
    
    xr = x#np.zeros(N)
    yr = z#np.zeros(N)
    zr = z#np.zeros(N)
    
    for l in range(nStep):
        for i in range(N):
            xRandom = float(gauss())
            yRandom = float(gauss())
            zRandom = float(gauss())
            
            x[i] = x[i] + fx[i]*dT + var*xRandom
            y[i] = y[i] + fy[i]*dT + var*yRandom
            z[i] = z[i] + fz[i]*dT + var*zRandom
            
            x[i] = x[i] - L*round(x[i]/L)
            y[i] = y[i] - L*round(y[i]/L)
            z[i] = z[i] - L*round(z[i]/L)
            
            xr[i] = xr[i] + fx[i]*dT + var*xRandom
            yr[i] = yr[i] + fy[i]*dT + var*yRandom
            zr[i] = zr[i] + fz[i]*dT + var*zRandom
            
            if l>nener and l%iFrec1==0.0 : 
            #for i in range(N):
                Cx[i][k1] = x[i]
                Cy[i][k1] = y[i]
                Cz[i][k1] = z[i]
                
            #k1 = k1 + 1
        
            if l>nener and l%iFrec2==0.0 : 
            #for i in range(N):
                Cxr[i][k2] = xr[i]
                Cyr[i][k2] = yr[i]
                Czr[i][k2] = zr[i]
            
        if l>nener and l%iFrec1==0.0 :
            k1 = k1 + 1
        if l>nener and l%iFrec2==0.0 : 
            k2 = k2 + 1
            
        fx = np.zeros(N)
        fy = np.zeros(N)
        fz = np.zeros(N)
        E, fx, fy, fz = fuerzaGauss(x,y,z,N,L,rCut,n)
                
        V[l] = E
    
    return x,y,z,Cx,Cy,Cz,Cxr,Cyr,Czr,V,k1,k2



