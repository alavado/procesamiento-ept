#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 08:07:16 2019

@author: marce
"""

#%%
import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal, misc
import matplotlib.pyplot as plt
import sys

import math

def processAll(imu, emg, ti, tf, c, tir, tfr, cr):

    def rmsValue(array):
        array = np.array(array)
        n = len(array)
        squre = 0.0
        root = 0.0
        mean = 0.0
        
        #calculating Squre
        for i in range(0, n):
            squre += (array[i] ** 2)
        #Calculating Mean
        mean = (squre/ (float)(n))
        #Calculating Root
        root = math.sqrt(mean)
        return root


    #%% carga de señales

    imu.t = (imu.t - imu.t[0])/1000 #transformacion a segundos
    lSho = imu.iloc[:,[0,1,2,3]]  #x: abd/add; y: flex/ext; z: rotExt/rotInt 
    lElb = imu.iloc[:,[0,4,5,6]]    #y: prono/supinacion; z: flex/exte
    lWri = imu.iloc[:,[0,7,8,9]]  #x: flex/ext; y:prono/supinacion; z: radializacion/uln; 

    #%% Funciones deteccion mov forzado

    #Hombro
    def flexSho(df):
        time = df.iloc[:,0]
        ang = df.iloc[:,2]
        mas60 =     time[(ang > 60)]
        en60y90 =   time[(ang > 60) & (ang < 90)]
        mas90 =     time[(ang > 90)]
        return mas60,en60y90,mas90

    def abdSho(df):
        time = df.iloc[:,0]
        ang = df.iloc[:,1]
        mas60 =     time[(ang < -60)]
        en60y90 =   time[(ang < -60) & (ang > -90)]
        mas90 =     time[(ang < -90)]
        return mas60,en60y90,mas90

    def rotSho(df):
        time = df.iloc[:,0]
        ang = df.iloc[:,3]
        mas40 = time[(ang > 40) | (ang < -40)]
        return mas40


    #Codo
    def proElb(df):
        time = df.iloc[:,0]
        ang = df.iloc[:,2]
        mas70 = time[(ang < -70)] #Revisar con datos 
        return mas70

    def supElb(df):
        time = df.iloc[:,0]
        ang = df.iloc[:,2]
        mas70 = time[(ang > 70)] #Revisar con datos 
        return mas70

    def extWri(df):
        time = df.iloc[:,0]
        ang = df.iloc[:,1]
        mas45 = time[(ang > 45)]
        mas30 = time[(ang > 30)]
        return mas45, mas30

    #Muneca (mas extension 45)
    def flexWri(df):
        time = df.iloc[:,0]
        ang = df.iloc[:,1]
        mas30 = time[(ang < -30)]
        return mas30

    def radWri(df):
        time = df.iloc[:,0]
        ang = df.iloc[:,3]
        mas10 = time[(ang > 10)] #Revisar con datos 
        return mas10

    def ulnWri(df):
        time = df.iloc[:,0]
        ang = df.iloc[:,3]
        mas10 = time[(ang < -10)] #Revisar con datos 
        return mas10



    def proWri(df):
        time = df.iloc[:,0]
        ang = df.iloc[:,2]
        mas70 = time[(ang < -70)] #Revisar con datos 
        return mas70

    def supWri(df):
        time = df.iloc[:,0]
        ang = df.iloc[:,2]
        mas70 = time[(ang > 70)] #Revisar con datos 
        return mas70

    #%% Funcion deteccion mov mantenido

    def mantenido(df,col):
        time = df.iloc[:,0]
        ang = df.iloc[:,col]
        inicio = time.iloc[0]
        rango = np.std(np.abs(np.diff(ang)))
        ind = []
        
        for i in np.arange(0,len(time)):
            inicio = time.iloc[i]
            intervalo = (time >= inicio) & (time < inicio+4.1)
            sel_ang = ang[intervalo]
            sel_time = time[intervalo]
            diferencia = np.abs(np.diff(sel_ang))
            if np.sum(diferencia < rango) == len(diferencia):
                ind.append(i)
        
        time = time[ind]
        ang  = ang[ind]

        return ang,time

    #%% Funcionpara obtener t inicial y t final de evento detectado
    def tiempo(x):
        lista = []
        
        if(x.empty):
            lista = [0]
        else:
            inicial = x.index[0]
            final = x.index[-1]
            for i in range(len(x)):
                if(i == len(x)-1):
                    final = x.index[i]
                    tf = x[final]
                    ti = x[inicial]
                    lista.append([ti,tf])
                elif(x.index[i]+1 != x.index[i+1]):
                    final = x.index[i]
                    tf = x[final]
                    ti = x[inicial]
                    lista.append([ti,tf])
                    inicial = x.index[i+1]
        return lista

    #%% Funcionpara obtener tiempo del evento
    def tiempo_exp(x):
        lista = []
        
        if(x.empty):
            lista = [0]
        else:
            inicial = x.index[0]
            final = x.index[-1]
            for i in range(len(x)):
                if(i == len(x)-1):
                    final = x.index[i]
                    tf = x[final]
                    ti = x[inicial]
                    lista.append(tf-ti)
                elif(x.index[i]+1 != x.index[i+1]):
                    final = x.index[i]
                    tf = x[final]
                    ti = x[inicial]
                    lista.append(tf-ti)
                    inicial = x.index[i+1]
        return lista

    #%% Codigo antiguo, incluye place holder de repetitivo
    #rep_ph = [[3,3.5],[5,5.5]]
    #flex_hombro = [tiempo(flexSho(lSho)[0]),tiempo(mantenido(lSho,2)[1]),rep_ph]
    #abd_hombro  = [tiempo(abdSho(lSho)[0]),tiempo(mantenido(lSho,1)[1]),rep_ph]
    #rot_hombro  = [tiempo(rotSho(lSho)),tiempo(mantenido(lSho,3)[1]),rep_ph]
    #ext_mun     = [tiempo(extWri(lWri)),tiempo(mantenido(lWri,1)[1]),rep_ph]
    #flex_mun    = [tiempo(flexWri(lWri)),tiempo(mantenido(lWri,1)[1]),rep_ph]
    #sup_mun     = [tiempo(supWri(lElb)),tiempo(mantenido(lElb,2)[1]),rep_ph]
    #prono_mun   = [tiempo(proWri(lElb)),tiempo(mantenido(lElb,2)[1]),rep_ph]
    #rad_mun     = [tiempo(extWri(lWri)),tiempo(mantenido(lWri,2)[1]),rep_ph]
    #uln_mun     = [tiempo(extWri(lWri)),tiempo(mantenido(lWri,2)[1]),rep_ph]


    #%% Calculo de listas (t_inicial, t_final) de segmentos detectados

    # flex_hombro = [tiempo(flexSho(lSho)[0]),tiempo(mantenido(lSho,2)[1])]
    # abd_hombro  = [tiempo(abdSho(lSho)[0]),tiempo(mantenido(lSho,1)[1])]
    # rot_hombro  = [tiempo(rotSho(lSho)),tiempo(mantenido(lSho,3)[1])]

    # sup_codo     = [tiempo(supWri(lElb)),tiempo(mantenido(lElb,2)[1])]
    # prono_codo   = [tiempo(proWri(lElb)),tiempo(mantenido(lElb,2)[1])]
    # ext_mun30     = [tiempo(extWri(lWri)[1]),tiempo(mantenido(lWri,1)[1])]

    # ext_mun45     = [tiempo(extWri(lWri)[0]),tiempo(mantenido(lWri,1)[1])]
    # flex_mun    = [tiempo(flexWri(lWri)),tiempo(mantenido(lWri,1)[1])]
    # rad_mun     = [tiempo(radWri(lWri)),tiempo(mantenido(lWri,3)[1])]
    # uln_mun     = [tiempo(ulnWri(lWri)),tiempo(mantenido(lWri,3)[1])]

    #%% resumen
    # hombro  = [flex_hombro,abd_hombro,rot_hombro]
    # codo    = [ext_mun30,flex_mun,sup_codo,prono_codo]
    # muneca  = [ext_mun45,flex_mun,rad_mun,uln_mun]
    # mano    = [flex_mun]

    #%% Tiempos de exposicion
    hombro_forzado = sum([sum(tiempo_exp(flexSho(lSho)[0])),
                        sum(tiempo_exp(abdSho(lSho)[0])),
                        sum(tiempo_exp(rotSho(lSho)))])

    hombro_mantenido = sum([sum(tiempo_exp(mantenido(lSho,2)[1])),
                            sum(tiempo_exp(mantenido(lSho,1)[1])),
                            sum(tiempo_exp(mantenido(lSho,3)[1]))])

    codo_forzado = sum([sum(tiempo_exp(supWri(lElb))),
                    sum(tiempo_exp(proWri(lElb))),
                    sum(tiempo_exp(extWri(lWri)[1])),
                    sum(tiempo_exp(flexWri(lWri)))])

    codo_mantenido = sum([sum(tiempo_exp(mantenido(lElb,2)[1])),
                        sum(tiempo_exp(mantenido(lWri,1)[1]))])

    muneca_forzado = sum([sum(tiempo_exp(extWri(lWri)[0])),
                        sum(tiempo_exp(flexWri(lWri))),
                        sum(tiempo_exp(radWri(lWri))),
                        sum(tiempo_exp(ulnWri(lWri)))])

    muneca_mantenido = sum([sum(tiempo_exp(mantenido(lWri,1)[1])),
                            sum(tiempo_exp(mantenido(lWri,3)[1]))])

    #%% EMG


    #procesamiento tiempo
    tiempo_total = sum(np.diff(emg.t))
    emg['tiempo2'] = np.linspace(0, tiempo_total, num=len(emg))
    sample_rate = len(emg)/tiempo_total

    #%%

    gain = 1 + 49400/47
    emgs = (((emg.iloc[:,1:5] * 3.3) / 1023) / gain)
    emgs = emgs - np.tile(np.mean(emgs),(len(emgs),1))

    hfrec = 10 #highpass cutoff frequency
    lfrec = 450 #lowpass cutoff frequency
    nfrec = sample_rate/2  #Niquist frequency
    order = 5 #orden del filtro
    b1,a1 = sp.signal.butter(order,hfrec/nfrec,'highpass') #Filtrado Pasa ALTO
    b2,a2 = sp.signal.butter(order,lfrec/nfrec,'lowpass')  #Filtrado Pasa Bajo
    emgs = sp.signal.filtfilt(b1,a1,emgs, padtype=None)
    emgs = sp.signal.filtfilt(b2,a2,emgs, padtype=None)

    emgs = sp.signal.hilbert(emgs)
    emgs = np.abs(emgs)

    # order = 1
    # frame_size = 31
    # emgs = sp.signal.savgol_filter(emgs, window_length=31, polyorder=1)

    emgs = pd.DataFrame(emgs)
    emgs.columns = ['emg1','emg2','emg3','emg4']
    emgs['t'] = emg.tiempo2
    #%%
    #ventana referencia (selector de ventana y canal en app web)
    inicio_ref = float(tir)
    final_ref = float(tfr)
    canal = int(cr)

    senal = emgs.iloc[:,canal]
    time = emgs.t

    ventana_ref = senal[(time>=inicio_ref) & (time<=final_ref)]

    mean_ref = np.mean(ventana_ref)
    rms_ref = rmsValue(ventana_ref)
    duracion = final_ref - inicio_ref
    maxV = np.max(ventana_ref)

    #%% ventana a analizar
    inicio_sel = float(ti)
    final_sel = float(tf)
    canal = int(c)

    senal = emgs.iloc[:,canal]
    time = emgs.t

    ventana_sel = senal[(time>=inicio_sel) & (time<=final_sel)]
    tiempo_sel = time[(time>=inicio_sel) & (time<=final_sel)]

    mean_sel = np.mean(ventana_sel)
    rms_sel = rmsValue(ventana_sel)
    duracion_sel = final_sel - inicio_sel
    maxV_sel = np.max(ventana_sel)

    #%% porcentajes
    mean_cvm = mean_sel/mean_ref*100
    rms_cvm = rms_sel/rms_ref*100

    #tiempo de rms > 30% de CVM
    porcentaje_tiempo = len(tiempo_sel[senal > rms_ref*0.3])/len(tiempo_sel)*100
    tiempo_sobre30 = sum(tiempo_exp(tiempo_sel[senal > rms_ref*0.3]))

    ret = {}
    # variables imu
    ret['hombro_forzado'] = hombro_forzado
    ret['hombro_mantenido'] = hombro_mantenido
    ret['codo_forzado'] = codo_forzado
    ret['codo_mantenido'] = codo_mantenido
    ret['muneca_forzado'] = muneca_forzado
    ret['muneca_mantenido'] = muneca_mantenido

    # variables emg
    ret['duracion_sel'] = duracion_sel
    ret['mean_sel'] = mean_sel
    ret['rms_sel'] = rms_sel
    ret['maxV_sel'] = maxV_sel
    ret['mean_cvm'] = mean_cvm
    ret['rms_cvm'] = rms_cvm
    ret['porcentaje_tiempo'] = porcentaje_tiempo
    ret['tiempo_sobre30'] = tiempo_sobre30

    return ret


