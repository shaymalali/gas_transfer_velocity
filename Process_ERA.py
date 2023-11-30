#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib as mlb 
import netCDF4 as nc
import pandas as pd
from scipy import interpolate
import xarray as xr
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm 

#############################################
## read in the ocean ERA data
#includes SWH (m) on 0.5x0.5 grid

fname='/Users/shaymaalali/Desktop/Python_Codes/adaptor.mars.internal-1695330000.1561894-16987-4-caaef19f-1ae7-4640-a0d0-49febfea2b9b.nc'
ds=xr.open_dataset(fname)
era_waves_lon=ds['longitude'].values
era_waves_lat=ds['latitude'].values
era_waves_time=ds['time'].values
era_waves_swh=ds['swh'].values

#transpose the data so dimensions are lat x lon x time
era_waves_swh=np.transpose(era_waves_swh,[1,2,0])
#add map projection
map_crs=ccrs.PlateCarree()
data_crs=ccrs.PlateCarree()
fig=plt.figure(figsize=(10,8))

ax = fig.add_subplot(111,projection=map_crs) #specify map projection
ax.coastlines()

cax=ax.contourf(era_waves_lon,era_waves_lat,era_waves_swh[:,:,30],transform=data_crs)
plt.colorbar(cax,label='Significant Wave Height (m)',orientation='horizontal')
plt.title('Significant Wave Height')

#############################################
## read in the SST and atmospheric ERA data
#atmospheric data is on a 0.25x0.25 grid and needs to be interpolated to match the ocean ERA data

fname = '/Users/shaymaalali/Desktop/Python_Codes/adaptor.mars.internal-1701204028.2774477-26018-4-d9483db0-f09b-4c24-a3ff-ffd392982dd7.nc'
ds = xr.open_dataset(fname)
era_atmo_lon = ds['longitude'].values
era_atmo_lat = ds['latitude'].values
era_atmo_sst = ds['sst'].values
era_atmo_time = ds['time'].values
era_atmo_ustar = ds['zust'].values
era_atmo_u10=ds['u10'].values
era_atmo_v10=ds['v10'].values

#calculate the wind speed using the u10 and v10 values
era_atmo_wind=np.sqrt(((era_atmo_v10)**2)+((era_atmo_u10)**2))

#transpose the data so dimensions are lat x lon x time
era_atmo_sst = np.transpose(era_atmo_sst, [1, 2, 0])
era_atmo_ustar = np.transpose(era_atmo_ustar, [1, 2, 0])
era_atmo_wind=np.transpose(era_atmo_wind,[1,2,0])

#interpolate the SST and atmospheric data to match the 0.5x0.5 ERA Ocean grid
for i in range(len(era_atmo_time)):
    sst = np.squeeze(era_atmo_sst[:, :, i])
    ustar = np.squeeze(era_atmo_ustar[:, :, i])
    wind=np.squeeze(era_atmo_wind[:,:,i])

    nan_map = np.zeros_like(sst)
    nan_map[np.isnan(sst)] = 1
    nan_map2 = np.zeros_like(ustar)
    nan_map2[np.isnan(ustar)] = 1
    nan_map3=np.zeros_like(wind)
    nan_map3[np.isnan(wind)]=1

    filled_z = sst.copy()
    filled_z[np.isnan(sst)] = 0

    filled_z2 = ustar.copy()
    filled_z2[np.isnan(ustar)] = 0
    
    filled_z3 = wind.copy()
    filled_z3[np.isnan(wind)] = 0

    f = interpolate.interp2d(
        era_atmo_lon, era_atmo_lat, filled_z, kind='linear')
    f_nan = interpolate.interp2d(
        era_atmo_lon, era_atmo_lat, nan_map, kind='linear')
    
    f2 = interpolate.interp2d(
        era_atmo_lon, era_atmo_lat, filled_z2, kind='linear')
    f2_nan = interpolate.interp2d(
        era_atmo_lon, era_atmo_lat, nan_map2, kind='linear')
    
    f3 = interpolate.interp2d(
        era_atmo_lon, era_atmo_lat, filled_z3, kind='linear')
    f3_nan = interpolate.interp2d(
        era_atmo_lon, era_atmo_lat, nan_map3, kind='linear')

    tmp1 = f(era_waves_lon, era_waves_lat)
    nan_new = f_nan(era_waves_lon, era_waves_lat)
    tmp1[nan_new > 0.5] = np.nan
    tmp1 = tmp1[::-1, :]

    tmp2 = f2(era_waves_lon, era_waves_lat)
    nan_new2 = f2_nan(era_waves_lon, era_waves_lat)
    tmp2[nan_new2 > 0.5] = np.nan
    tmp2 = tmp2[::-1, :]
    
    tmp3 = f3(era_waves_lon, era_waves_lat)
    nan_new3 = f3_nan(era_waves_lon, era_waves_lat)
    tmp3[nan_new2 > 0.5] = np.nan
    tmp3 = tmp3[::-1, :]

    if i == 0:
        era_waves_sst = tmp1
        era_waves_ustar = tmp2
        era_waves_wind=tmp3
    elif i == 1:
        era_waves_sst = np.stack([era_waves_sst, tmp1], axis=2)
        era_waves_ustar = np.stack([era_waves_ustar, tmp2], axis=2)
        era_waves_wind=np.stack([era_waves_wind,tmp3],axis=2)
    else:
        era_waves_sst = np.append(era_waves_sst, np.atleast_3d(tmp1), axis=2)
        era_waves_ustar = np.append(era_waves_ustar, np.atleast_3d(tmp2), axis=2)
        era_waves_wind=np.append(era_waves_wind,np.atleast_3d(tmp3),axis=2)

#plot the interpolated and non-interpolated values
fig = plt.figure()
fig.suptitle('Comparing the interpolated and non-interpolated values')
map_crs = ccrs.PlateCarree()
data_crs = ccrs.PlateCarree()

ax1 = fig.add_subplot(221, projection=map_crs)
ax1.coastlines()
cax = ax1.contourf(era_atmo_lon, era_atmo_lat,
                   era_atmo_ustar[:, :, 5], transform=data_crs, cmap=cm.jet)
plt.colorbar(cax, label='Friction Velocity (m/s)', orientation='vertical')
plt.title('Non-interpolated values of Friction Velocity')

ax2 = fig.add_subplot(222, projection=map_crs)
ax2.coastlines()
cax = ax2.contourf(era_waves_lon, era_waves_lat,
                   era_waves_ustar[:, :, 5], transform=data_crs, cmap=cm.jet)
plt.colorbar(cax, label='Friction Velocity (m/s)', orientation='vertical')
plt.title('Interpolated values of Friction Velocity')

ax3 = fig.add_subplot(223, projection=map_crs)
ax3.coastlines()
cax = ax3.contourf(era_atmo_lon, era_atmo_lat,
                   era_atmo_sst[:, :, 1], transform=data_crs, cmap=cm.jet)
plt.colorbar(cax, label='Sea Surface Temperature', orientation='vertical')
plt.title('Non-interpolated values of Sea Surface Temperature')

ax4 = fig.add_subplot(224, projection=map_crs)
ax4.coastlines()
cax = ax4.contourf(era_waves_lon, era_waves_lat,
                   era_waves_sst[:, :, 0], transform=data_crs, cmap=cm.jet)
plt.colorbar(cax, label='Sea Surface Temperature', orientation='vertical')
plt.title('Interpolated values of Sea Surface Temperature')



#############################################

# read in the Sea Surface Salinity
import glob
direc=glob.glob('/Users/shaymaalali/Desktop/Python_Codes/OISSS_*.nc')
direc.sort()

X,Y=np.meshgrid(era_waves_lon,era_waves_lat) # to interpolate the SSS grid

for file in range(len(direc)):
    if file == 0:
      fname=direc[file]
      ds=xr.open_dataset(fname)
      SSS_lat=ds['latitude'].values
      SSS_lon=ds['longitude'].values
      SSS_time=ds['time'].values
      SSS=ds['sss'].values
      SSS=np.squeeze(SSS)
      
      #convert lon from [-180 to 180] to [0 360]
      idx=np.where(SSS_lon < 0)
      SSS_lon[idx]=SSS_lon[idx]+360
      ix=np.argsort(SSS_lon)
      
      SSS_lon=SSS_lon[ix]
      SSS=SSS[:,ix]
     
      
      # flip the latitudes so it ranges from [90 -90] instead of [-90 90]
      SSS_lat=np.flip(SSS_lat)
      SSS=SSS[::-1,:]
      #interpolate the SSS values to be on same 0.5x0.5 grid as ERA-5
      nan_map = np.zeros_like(SSS)
      nan_map[ np.isnan(SSS) ] = 1
      
      filled_z=SSS.copy()
      filled_z[np.isnan(SSS)]=0
      
      f = interpolate.interp2d(SSS_lon,SSS_lat,filled_z,kind='linear')
      f_nan=interpolate.interp2d(SSS_lon,SSS_lat,nan_map,kind='linear')
      
      tmp=f(era_waves_lon,era_waves_lat)
      nan_new=f_nan(era_waves_lon,era_waves_lat)
      tmp[nan_new>0.5]=np.nan
      tmp=tmp[::-1,:]
      #plot the interpolated and non interpolated SS
      fig=plt.figure()
      fig.suptitle('Non interpolated vs interpolated SSS values')
      map_crs=ccrs.PlateCarree()
      data_crs=ccrs.PlateCarree()
      
     
      ax1 = fig.add_subplot(211,projection=map_crs)
      ax1.coastlines()
      cax=ax1.contourf(SSS_lon,SSS_lat,np.squeeze(SSS),transform=data_crs,cmap=cm.jet)
      plt.colorbar(cax,label='Sea Surface Salinity',orientation='vertical')
      plt.title('Non-interpolated SSS values')
      
      ax2 = fig.add_subplot(212,projection=map_crs)
      ax2.coastlines()
      cax=ax2.contourf(era_waves_lon,era_waves_lat,np.squeeze(tmp),transform=data_crs,cmap=cm.jet)
      plt.colorbar(cax,label='Sea Surface Salinity',orientation='vertical')
      plt.title('Interpolated SSS values')
      
      SSS_interp=tmp
    else:
       fname=direc[file]
       ds=xr.open_dataset(fname)
       tmp=ds['sss'].values
       tmp=np.squeeze(tmp)
       
       tmp=tmp[::-1,:]
       #interpolate the SSS values to be on same 0.5x0.5 grid as ERA-5
       nan_map = np.zeros_like(tmp)
       nan_map[ np.isnan(tmp) ] = 1
       
       filled_z=tmp.copy()
       filled_z[np.isnan(tmp)]=0
       
       f = interpolate.interp2d(SSS_lon,SSS_lat,filled_z,kind='linear')
       f_nan=interpolate.interp2d(SSS_lon,SSS_lat,nan_map,kind='linear')
       
       tmp2=f(era_waves_lon,era_waves_lat)
       nan_new=f_nan(era_waves_lon,era_waves_lat)
       tmp2[nan_new>0.5]=np.nan
       tmp2=tmp2[::-1,:]
       
       if file ==1:
          SSS_interp=np.stack([SSS_interp,tmp2],axis=2)
       else:
          SSS_interp=np.append(SSS_interp,np.atleast_3d(tmp2),axis=2)
          
#############################################

#calculate the transfer velocities using the transfer velocity parametrization from Reichl & Deike (2020)

from TransferVelocity_Funcs import calcSolubility
from TransferVelocity_Funcs import calcTransferVelocity
from TransferVelocity_Funcs import calcWanKw

K0=calcSolubility(era_waves_swh,era_waves_time,SSS_interp,era_waves_sst)
Sc, Kwnb, Kwb, Kw =calcTransferVelocity(era_waves_swh,era_waves_sst,era_waves_ustar,K0,era_waves_time)
Sc, Kw_Wan =calcWanKw(era_waves_wind, era_waves_sst, era_waves_time)

#plot the transfer velocity 
map_crs=ccrs.PlateCarree()
data_crs=ccrs.PlateCarree()
fig=plt.figure(figsize=(10,8))

ax = fig.add_subplot(111,projection=map_crs) #specify map projection
ax.coastlines()

cax=ax.contourf(era_waves_lon,era_waves_lat,Kw[:,:,30],transform=data_crs)
plt.colorbar(cax,label='Transfer Velocity (cm/hr)',orientation='horizontal')
plt.title('Transfer Velcoity (cm/hr)')

#############################################

#compare and contrast the transfer velocities

#calculate the global annual mean gas transfer velocity for both paramterizations
#compare the bubble and non-bubble transfer velocity annual averages
#Calculate the global monthly mean and then convert the arrays to a Pandas Dataframe
kw_mean=[]
kwb_mean=[]
kwnb_mean=[]
kwWan_mean=[]
time=[]
for i in range(len(era_waves_time)):
    
    kw_tmp=np.nanmean(Kw[:,:,i])
    kwb_tmp=np.nanmean(Kwb[:,:,i])
    kwnb_tmp=np.nanmean(Kwnb[:,:,i])
    kwWan_tmp=np.nanmean(Kw_Wan[:,:,i])
    time_tmp=pd.to_datetime(era_waves_time[i])
    
    kw_mean.append(kw_tmp)
    kwb_mean.append(kwb_tmp)
    kwnb_mean.append(kwnb_tmp)
    kwWan_mean.append(kwWan_tmp)
    time.append(time_tmp)
    
monthlymean=pd.DataFrame({'time':time,'KW':list(kw_mean),'KWB':list(kwb_mean),'KWNB':list(kwnb_mean),'KW_Wan':list(kwWan_mean)},columns=['time','KW','KWB','KWNB','KW_Wan'])

#plot the monthly means 

monthlymean.plot(y=['KW','KWB','KWNB','KW_Wan'],figsize=(10,6),title='Monthly Averages of Gas Transfer Velocity',ylabel='(cm/hr)')

#calculate the annual mean and plot it 
annualmean = monthlymean.groupby(monthlymean["time"].dt.year)[monthlymean.columns[1:]].agg("mean")

annualmean.plot(y=['KW','KWB','KWNB','KW_Wan'],figsize=(10,6),title='Annual Averages of Gas Transfer Velocity',ylabel='(cm/hr)')
