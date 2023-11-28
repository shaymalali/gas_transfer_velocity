#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:56:37 2023

@author: shaymaalali
"""

############################################

#function to calculate the solubility using the equation from Reichle & Deike (2020)
# ln(K0)=a1+a2(100/T0)+a3*ln(T0/100)+S0(b1+b2(T0/100)+b3(t0/100)^2)
# T0 is the sea surface temperature in Kelvin 
#S0 is the surface salinity in g/kg 

def calcSolubility(swh,time,salinity,sst):
    
    import pandas as pd
    import numpy as np
    import math
    a1=-58.0931
    a2=90.5069
    a3=22.2490
    b1=0.027766
    b2=-0.025888
    b3=0.0050578
    
    for i in range(len(time)):
        
        #get the swh and SST
        swh1=swh[:,:,i]
        sst1=sst[:,:,i]
        
        #get the sea surface salinity
        date=time[i]
        month=pd.to_datetime(date).month
        month=month-1
        sss=salinity[:,:,month]
        
        j1=a1
        j2=(a2*(100/sst1))
        j3=(a3*np.log(sst1/100))
        j4=(b1+(b2*(sst1/100)))+(b3*((sst1/100)**2))
        j5=sss*j4
        jj=j1+j2+j3+j5
        jj=np.exp(jj)
        
        if i == 0:
            k0=jj
        elif i==1:
            k0=np.stack([k0,jj],axis=2)
        else:
            k0=np.append(k0,np.atleast_3d(jj),axis=2)
    
    return k0;

############################################

#function to calculate the transfer velocity from Reichl & Deike (2020)
#Kw is the sum of non-bubble (kwnb) and non-bubble (kwb) components
#kwnb=Anb*ustar*(Sc/660)^(-1/2)
#where Anb=1.55*10^-4 (Fairral et all (2011))
#and Kwb=(Ab/K0*R*T0)*ustar^(5/3)*(gHs)^(2/3)*(Sc/660)^(-1/2)
#where Ab=1*10^-5
# R is the ideal gas constant with units m3.atm/K.mol
#T0=SST in Celcius
#g=Gravity

def calcTransferVelocity(swh,sst,ustar,k0,time):
    
    import numpy as np
    
    
    #Coefficients
    Anb=1.55e-5
    Ab=1e-5
    g=9.8
    R=8.206e-5
    
    for i in range(len(time)):
        #convert temperature to Celcius 
        sst_c=sst[:,:,i]-273
        #Schmidt Number Coefficients
        A=2116.8
        B=(-136.25)*sst_c
        C=4.7353*(sst_c**2)
        D=(-0.092307)*(sst_c**3)
        E=0.000755*(sst_c**4)
        
        tmp_sc=A+B+C+D+E
        if i == 0:
            Sc=tmp_sc
        elif i ==1:
            Sc=np.stack([Sc,tmp_sc],axis=2)
        else:
            Sc=np.append(Sc,np.atleast_3d(tmp_sc),axis=2)
        
        ## calculate the non-bubble component of the transfer velocity
        tmp_kwnb=Anb*ustar[:,:,i]*(tmp_sc/660)**(-1/2)
        tmp_kwnb=tmp_kwnb*360000 #convert from m/s to cm/hr
        
        if i == 0:
            Kwnb=tmp_kwnb
        elif i ==1:
            Kwnb=np.stack([Kwnb,tmp_kwnb],axis=2)
        else:
            Kwnb=np.append(Kwnb,np.atleast_3d(tmp_kwnb),axis=2)
        
        ## calculate the bubble component of the transfer velocity
        
        tmp_ko=k0[:,:,i]*(1/0.001) #convert to mol.(m3.atm)**(-1)
        
        l=(Ab/(tmp_ko*R*sst[:,:,i]))
        ll=ustar[:,:,i]**(5/3)
        l2=(g*swh[:,:,i])**(2/3)
        ll2=(tmp_sc/660)**(-1/2)
        
        tmp_kwb=l*ll*l2*ll2
        tmp_kwb=tmp_kwb*360000 #convert from m/s to cm/hr
        
        if i == 0:
            Kwb=tmp_kwb
        elif i ==1:
            Kwb=np.stack([Kwb,tmp_kwb],axis=2)
        else:
            Kwb=np.append(Kwb,np.atleast_3d(tmp_kwb),axis=2)
        
        ## calculate the total transfer velocity
        
        tmp_kw=tmp_kwnb+tmp_kwb
        
        if i == 0:
            Kw=tmp_kw
        elif i ==1:
            Kw=np.stack([Kw,tmp_kw],axis=2)
        else:
            Kw=np.append(Kw,np.atleast_3d(tmp_kw),axis=2)
    
    return Sc,Kwnb,Kwb,Kw;
        
        
        
        
        
        
        