#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("/home/yc424/.local/lib/python3.9/site-packages")
sys.path.append("/home/aad473/.local/lib/python3.9/site-packages")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymmwr as pm
from datetime import datetime
from datetime import date
from datetime import timedelta
from scipy.integrate import solve_ivp
from scipy.stats import gamma, nbinom
from scipy.special import loggamma
from copy import copy
from scipy.optimize import minimize, Bounds
from sys import argv
from scipy.stats import nbinom
from matplotlib.pyplot import cm
locationIndex = int(argv[1])
add_beta=0

state_names = ["District of Columbia", "Puerto Rico","Virgin Islands","Florida","Alabama", "Alaska","Arkansas", "Arizona", "California", "Colorado", "Connecticut", "Delaware", "Georgia", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]

state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Puerto Rico": "PR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Virgin Islands": "VI",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}



# In[32]:

data_folder = '/projects/math_cheny/FluSight/Data'
#data_folder='G:/My Drive/papers/FluSight/2022QualityCheck'
path_to_results='G:/My Drive/papers/FluSight/2022QualityCheck/mcmc_results/'
path_to_results_N='/projects/math_cheny/FluSight/2023_2024_test/mcmc_results/'
path_to_results_N_plus1='/projects/math_cheny/FluSight/2023_2024/mcmc_N3_2022only_addBeta/'



ili_all=pd.read_csv(data_folder+'/ILINet.csv', skiprows=1)
states_name = list(set(np.array(ili_all['REGION'])))
#states_name.remove('Florida')
states_name.remove('Commonwealth of the Northern Mariana Islands')
#states_name.remove('Puerto Rico')
#states_name.remove('Virgin Islands')
states_name.remove('New York City')
#states_name.remove('District of Columbia')
states_name.sort()

if add_beta==0:
    path_to_results=path_to_results_N
else:
    path_to_results=path_to_results_N_plus1
national=0
if locationIndex==53: #National data
    locationIndex=52
    national=1
targetState = state_names[locationIndex]

targetIndex = states_name.index(targetState)
#best_fit_parameters=pd.read_csv(path_to_results+'best_fit_parameters_'+targetState+'.csv')

covid=pd.read_csv(data_folder+'/rows.csv?accessType=DOWNLOAD')

#covid=pd.read_csv(data_folder+'/COVID-19_Reported_Patient_Impact_and_Hospital_Capacity_by_State_Timeseries.csv')

HI=covid[['date','state','previous_day_admission_influenza_confirmed','total_patients_hospitalized_confirmed_influenza_coverage']].sort_values(['state','date'])
state_HI_data = HI[HI['state']=='NY']

L = len(np.array(state_HI_data['previous_day_admission_influenza_confirmed'].values))


index = -1

targetSeason = 2023
onset = pm.epiweek_to_date(pm.Epiweek(targetSeason, 26))
end = pm.epiweek_to_date(pm.Epiweek(targetSeason+1, 47))

season2021Data = np.nan*np.zeros((len(states_name), (end-onset).days))
season2021dateArray = [[] for i in range(len(states_name))]

for name in states_name:
    
    index += 1
    state_HI_data = HI[HI['state']==state_to_abbrev[name]]
        
    dateArray = [ datetime.strptime(state_HI_data['date'].values[i], '%Y/%m/%d')-timedelta(days=1) for i in range(len(state_HI_data))]
    binary = [onset <= buff.date() < end for buff in dateArray] # previous day 
    
    tSpan = np.array([(dateArray[i]-dateArray[0]).days for i in range(len(state_HI_data))])
    
    season2021Data[index,:np.sum(binary)] = state_HI_data['previous_day_admission_influenza_confirmed'][binary]
    season2021dateArray[index] = np.array(dateArray)[binary]
    
def dateArray_to_timeArray(dateArray):
    
    timeArray = np.zeros(len(dateArray))
    timeArray[:] = np.array([(date-dateArray[0]).days for date in dateArray], dtype='int64')
    return timeArray.astype('int64')

Y22 = season2021Data[targetIndex,:]
X22 = dateArray_to_timeArray(season2021dateArray[targetIndex])

    



def dateArray_to_timeArray(dateArray):
    
    timeArray = np.zeros(len(dateArray))
    timeArray[:] = np.array([(date-dateArray[0]).days for date in dateArray], dtype='int64')

    return timeArray.astype('int64')


par = {'beta':0.5,
       'gamma':0.244,
       'betaMax':0.002,
       'betaMax':0.0025,
       'theta':0.,
       'I0':0.0001,
       'S0':0.8,
       'fD':1000,
       'onset':150,
       'r':10,
       'delta':1.0/2/365}


# In[55]:


bounds = [[0, np.inf],
          [0, np.inf],
          [0, np.inf],
          [0, np.inf],
          [-np.pi/8, np.pi/8],
          [0, 1],
          [0, 1],
          [0, np.inf],
          [0, np.inf],
          [0, np.inf],
          [0, np.inf]] 


# In[56]:


def par_to_array(par):
    
    return np.array([par['beta'],
                     par['gamma'],
                     par['betaMax1'],
                     par['betaMax2'],
                     par['theta'],
                     par['I0'],
                     par['S0'],
                     par['fD'],
                     par['onset'],
                     par['r'],
                     par['delta']])

def array_to_par(nparray):
    
    par = {'beta':nparray[0],
       'gamma':nparray[1],
       'betaMax1':nparray[2],
       'betaMax2':nparray[3],
       'theta':nparray[4],
       'I0':nparray[5],
       'S0':nparray[6],
       'fD':nparray[7],
       'onset':nparray[8],
       'r':nparray[9],
       'delta':nparray[10]}
    
    return par


def array_assign_to_par(nparray):
    
    par['beta'],par['gamma'],par['betaMax1'],par['betaMax2'],par['theta'],par['I0'],par['S0'],par['fD'],par['onset'],par['r'],par['delta']=nparray
    
    return par


def IS(alpha,predL,predU):
    
    return lambda y: (predU-predL)+2/alpha*(y<predL)*(predL-y) + 2/alpha*(y>predU)*(y-predU)

def WIS(yObs, qtlMark,predQTL):
    
    # checking if the qtlMark is well-defined
    ifWellDefined = np.mod(len(qtlMark),2)!=0
    
    NcentralizedQT = (len(qtlMark)-1)//2 + 1
    
    alphaList = np.zeros(NcentralizedQT)
    weightList = np.zeros(NcentralizedQT)
    
    for i in range(NcentralizedQT):
        
        ifWellDefined = ifWellDefined & (np.abs(-1.0+qtlMark[i]+qtlMark[-1-i])<1e-8)
        alphaList[i] = 1-(qtlMark[-1-i]-qtlMark[i])
        weightList[i] = alphaList[i]/2
        
    if ifWellDefined:
        
        #print(alphaList)
        #print(qtlMark)
        #print(NcentralizedQT)
        
        output = 1.0/2*np.abs(yObs-predQTL[NcentralizedQT-1])
        
        for i in range(NcentralizedQT-1):
            
            output += weightList[i]*IS(alphaList[i],predQTL[i],predQTL[-1-i])(yObs)
            
            #print(alphaList[i], predQTL[i],predQTL[-1-i])
            
        return output/(NcentralizedQT-1/2)
    
        
    else:
        
        print('Check the quantile marks: either no median defined, or not in symmetric central QTL form.')
        

def transformQTL(inputQTL, weights):
    
    outputQTL = np.zeros_like(inputQTL)
    
    outputQTL[0] = inputQTL[0]*weights[0]
        
    for i in range(1,len(outputQTL)):
        
        outputQTL[i] = outputQTL[i-1] + weights[i]*(inputQTL[i]-inputQTL[i-1])
        
    return outputQTL

def lossTransformQTL(Yseries,inferredQTL,weights,qtlMark):
    
    NQTL, NT = inferredQTL.shape
    
    if len(Yseries)<=NT:

        sumWIS = 0
        counter = 0
        
        for i in range(len(Yseries)):

            if ~np.isnan(Yseries[i]):
                
                transformedQTLval = transformQTL(inferredQTL[:,i], weights)
                #transformedQTLval = inferredQTL[:,i]
                sumWIS += WIS(Yseries[i], qtlMark, transformedQTLval)
                counter += 1
                
        return sumWIS/counter 
    
    else:
        
        print('time series length is longer than that of the QTL series.')
        
qtlMark = np.array([0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990])


    
targetState = state_names[locationIndex]

# Loading the state definition
targetIndex = states_name.index(targetState)
if national==1: #National data

    for locationIndex in range(len(state_names)-1): #Y22 is the last state, skip it
        
        targetState = state_names[locationIndex]
        
        targetIndex = states_name.index(targetState)
    
        Y22 = Y22+season2021Data[targetIndex,:]
    
    targetState='National'   
    
prefix=path_to_results+targetState



nWeek=int(np.floor(len(X22)/7)) 
Y_weekly=np.zeros((nWeek))
tskip=np.remainder(len(X22), 7) #If the number of days are not divisible by 7, skip the first few days
for j in range(nWeek):

    #Convert daily data to weekly data
    Y_weekly[j]=np.sum(Y22[j*7+tskip:(j+1)*7+tskip])


inferredQTL = np.genfromtxt(prefix+'-percentiles-weekly.txt')
NQTL, NT = inferredQTL.shape
ylen=NT-4

weights = np.ones(NQTL)
lbound, ubound = np.zeros(NQTL), np.inf*np.ones(NQTL)
bounds = Bounds(lbound, ubound)

output = minimize(lambda weights: lossTransformQTL(Y_weekly[-ylen:],inferredQTL,weights,qtlMark), weights, bounds=bounds, method='Nelder-Mead')

optimized_weights = output.x

fullTransformedQTL = np.zeros_like(inferredQTL)

for i in range(NT):
    fullTransformedQTL[:,i] = transformQTL(inferredQTL[:,i], optimized_weights)


np.savetxt(prefix+'-percentiles-weekly-calibrated.txt',fullTransformedQTL)      

    

