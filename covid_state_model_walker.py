#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Python Model

# In[4]:


import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.integrate import odeint
from diffeqpy import de
from datetime import datetime


# ## Parameters

# Population parameters

# In[5]:


agepars_meanage_in=np.arange(5,95,10)
agepars_highage=np.arange(9,99,10)
agepars_lowage=np.arange(0,90,10)

#Data from 2018 census
population_N= 10666108

population_agefrac_in = [0.126,0.137,0.139,0.132,0.130,0.129,0.104,0.061,0.036,0.007]
myInt = sum(population_agefrac_in)
population_agefrac = [x / myInt for x in population_agefrac_in]

agepars_meanage= [a * b for a, b in zip(agepars_meanage_in, population_agefrac)]
population_meanage = sum(agepars_meanage)


# Check if population data sums to ~1.00

# In[6]:


x = (sum(population_agefrac))
y = format(x,'.5f')
yy = format(1,'.5f')
bool(y==yy)


# Basic parameters

# In[7]:


pars_gamma_e=1/4;	#Transition to infectiousness
pars_gamma_a=1/6;	#Resolution rate for asymptomatic
pars_gamma_s=1/6;	#Resolution rate for symptomatic
pars_gamma_h=1/10;	#Resolution rate in hospitals
pars_beta_a=4/10;	#Transmission for asymptomatic
pars_beta_s=8/10;	#Transmission for symptomatic

pars_p=[0.95,0.95,0.90,0.8,0.7,0.6,0.4,0.2,0.2,0.2]			#Fraction asymptomatic

pars_overall_p= sum([a * b for a, b in zip(pars_p, population_agefrac)])

pars_Itrigger = 500000/population_N #Trigger at 5000 total cases, irrespective of type


# Age stratification

# In[8]:


agepars_hosp_frac_in=[0.1,0.3,1.2,3.2,4.9,10.2,16.6,24.3,27.3,27.3]
agepars_hosp_frac = [x / 100 for x in agepars_hosp_frac_in]

agepars_hosp_crit_in=[5,5,5,5,6.3,12.2,27.4,43.2,70.9,70.9]
agepars_hosp_crit = [x / 100 for x in agepars_hosp_crit_in]

agepars_crit_die= 0.5*np.ones(len(agepars_meanage)+1) ## CHECK
agepars_num_ages = len(agepars_meanage);

N=agepars_num_ages;
agepars_S_ids= (1,N)
agepars_E_ids= ((N+1),(2*N))
agepars_Ia_ids=((2*N+1),(3*N))
agepars_Is_ids=((3*N+1),(4*N))
agepars_Ihsub_ids=((4*N+1),(5*N))
agepars_Ihcri_ids=((5*N+1),(6*N))
agepars_R_ids=((6*N+1),(7*N))
agepars_D_ids=((7*N+1),(8*N))
agepars_Hcum_ids=((8*N+1),(9*N))

agepars_IFR_2= [a * b * c * d for a, b, c, d in zip(population_agefrac, agepars_hosp_frac, agepars_hosp_crit, agepars_crit_die)]
pp = [a-b for a, b in zip(np.ones(len(pars_p)), pars_p)]
agepars_IFR_1= [a*b for a,b in zip(agepars_IFR_2,pp)]
agepars_IFR = sum(agepars_IFR_1)


# Epidemiological parameters

# In[9]:


pars_Ra=pars_beta_a/pars_gamma_a;
pars_Rs=pars_beta_s/pars_gamma_s;


x = [a-b for a, b in zip(np.ones(len(pars_p)), pars_p)] #1-pars_p
y = [a*b for a,b in zip(x,population_agefrac)] #(1-pars_p*pop_agefrac)
z = [a*pars_Rs for a in y] #(1-pars_p*pop_agefrac*pars_Rs)
m = [a*b*pars_Ra for a,b in zip(pars_p,population_agefrac)] #(pars_p*pop_agefrac*pars_Ra)
pars_R0 = [a*b for a,b in zip(z,m)]

#pars_bau=pars;


# ## Initial Conditions

# Population initial conditions

# In[23]:


#SEIaIS (open) and then I_ha I_hs and then R (open) and D (cumulative) age stratified

# Vector for each age group has structure - [agefrac (S), E, Ia, Is, Ihsub, Ihcrit, R, D, Hcum]
# Ages 9-19
outbreak_y0_9_19=([population_agefrac[0],0,0,0,0,0,0,0,0])
outbreak_y0_9_19=[a*population_N for a in outbreak_y0_9_19]
outbreak_y0_9_19=[a/population_N for a in outbreak_y0_9_19]


# Ages 19-29
outbreak_y0_19_29=([population_agefrac[1],0,0,0,0,0,0,0,0])
outbreak_y0_19_29=[a*population_N for a in outbreak_y0_19_29]
outbreak_y0_19_29[0] = outbreak_y0_19_29[0]-1
outbreak_y0_19_29[1]= 1
outbreak_y0_19_29=[a/population_N for a in outbreak_y0_19_29]

# Ages 29-39
outbreak_y0_29_39=([population_agefrac[2],0,0,0,0,0,0,0,0])
outbreak_y0_29_39=[a*population_N for a in outbreak_y0_29_39]
outbreak_y0_29_39=[a/population_N for a in outbreak_y0_29_39]

# Ages 39-49
outbreak_y0_39_49=([population_agefrac[3],0,0,0,0,0,0,0,0])
outbreak_y0_39_49=[a*population_N for a in outbreak_y0_39_49]
outbreak_y0_39_49=[a/population_N for a in outbreak_y0_39_49]

# Ages 49-59
outbreak_y0_49_59=([population_agefrac[4],0,0,0,0,0,0,0,0])
outbreak_y0_49_59=[a*population_N for a in outbreak_y0_49_59]
outbreak_y0_49_59=[a/population_N for a in outbreak_y0_49_59]

# Ages 59-69
outbreak_y0_59_69=([population_agefrac[5],0,0,0,0,0,0,0,0])
outbreak_y0_59_69=[a*population_N for a in outbreak_y0_59_69]
outbreak_y0_59_69=[a/population_N for a in outbreak_y0_59_69]

# Ages 69-79
outbreak_y0_69_79=([population_agefrac[6],0,0,0,0,0,0,0,0])
outbreak_y0_69_79=[a*population_N for a in outbreak_y0_69_79]
outbreak_y0_69_79=[a/population_N for a in outbreak_y0_69_79]

# Ages 79-89
outbreak_y0_79_89=([population_agefrac[7],0,0,0,0,0,0,0,0])
outbreak_y0_79_89=[a*population_N for a in outbreak_y0_79_89]
outbreak_y0_79_89=[a/population_N for a in outbreak_y0_79_89]

# Ages 89-99
outbreak_y0_89_99=([population_agefrac[8],0,0,0,0,0,0,0,0])
outbreak_y0_89_99=[a*population_N for a in outbreak_y0_89_99]
outbreak_y0_89_99=[a/population_N for a in outbreak_y0_89_99]


# In[ ]:





# ### Data and dates from GA survey - I am still fiddling with this

# In[41]:


outbreak_pTime=365;
outbreak_pNear=30;
outbreak_pshift=0;


# In[42]:


# Set timings for plots
now = datetime.now()

#tickdates = [(now - outbreak_pTime),10,10]
now.strftime("%d/%m/%Y %H:%M:%S")
#tickdates.strftime("%d/%m/%Y %H:%M:%S")

# Through 3/28 - last 7 days
ga_cases  = [552,620,800,1097,1387,1643,2198]
ga_hosp =  [186,240,361,438,509,607,660]
ga_death = [25,25,38,47,56,65,79]
#ga_date = [(now-6),now]


# ## Mathematical Model

# State level model function (S,E,Ia,Is,R)

# In[23]:


def f(Y,t,_): #Y is the 81 len arr from prob, t is pycall.jlwrap.diffeqbase.nullparameters(); is this intended?
	## Algebraic relations ##
	Ia=sum(agepars_Ia_ids)
	Is=sum(agepars_Is_ids)
	R = sum(agepars_R_ids)
	S = sum(agepars_S_ids)
	E = sum(agepars_E_ids)
	
	# Define output variables for plotting
	u = Ia,Is,R,S,E
	
	## Trigger Loop ##
	Itot = 1-sum([Y[0],Y[9],Y[18],Y[27],Y[36],Y[45],Y[54],Y[63],Y[72]])
	value = Itot - pars_Itrigger
	
	if value < 0:
		trigger = 0
	else:
		trigger = 1
		
	## Differential variables ##
	return [

# Ages 9-19
			#Y[0] = S population = agepars_S_ids
			-pars_beta_a*Y[0]*Ia - pars_beta_s*Y[0]*Is,
		
			#Y[1] = E population
			pars_beta_a*Y[0]*Ia + pars_beta_s*Y[0]*Is - pars_gamma_e*Y[1],
		
			#Y[2] = Ia population
			np.transpose(pars_p)*pars_gamma_e*Y[1]-pars_gamma_a*Y[2],
		
			#Y[3] = Is population
			np.transpose(np.ones(len(pars_p))-pars_p)*pars_gamma_e*Y[3] - pars_gamma_s*Y[3],
		
			#Y[4] = Ihsub population
			np.transpose(agepars_hosp_frac)*(1-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[3] - pars_gamma_h*Y[4],
		
			#Y[5] = Ihcri population
			np.transpose(agepars_hosp_frac)*(1-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[3] - pars_gamma_h*Y[5],
		
			#Y[6] = R population
			(pars_gamma_a*Y[2]) + (pars_gamma_s*Y[3]*(1-np.transpose(agepars_hosp_frac))) + (pars_gamma_h*Y[4]) + (pars_gamma_h*Y[5])*(1-np.transpose(agepars_crit_die)),
		
			#Y[7] = D population
			pars_gamma_h*Y[5]*np.transpose(agepars_crit_die),
		
			#Y[8] = Hcum population
			(np.transpose(agepars_hosp_frac)*(1-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[3]) + (np.transpose(agepars_hosp_frac)*np.transpose(agepars_hosp_crit)*pars_gamma_s*Y[3]),

# Ages 19-29
			#Y[9] = S population = agepars_S_ids
			-pars_beta_a*Y[9]*Ia - pars_beta_s*Y[9]*Is,
		
			#Y[10] = E population
			pars_beta_a*Y[9]*Ia + pars_beta_s*Y[9]*Is - pars_gamma_e*Y[10],
		
			#Y[11] = Ia population
			np.transpose(pars_p)*pars_gamma_e*Y[10]-pars_gamma_a*Y[11],
		
			#Y[12] = Is population
			np.transpose(np.ones(len(pars_p))-pars_p)*pars_gamma_e*Y[12] - pars_gamma_s*Y[12],
		
			#Y[13] = Ihsub population
			np.transpose(agepars_hosp_frac)*(10-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[12] - pars_gamma_h*Y[13],
		
			#Y[14] = Ihcri population
			np.transpose(agepars_hosp_frac)*(10-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[12] - pars_gamma_h*Y[14],
		
			#Y[15] = R population
			(pars_gamma_a*Y[11]) + (pars_gamma_s*Y[12]*(10-np.transpose(agepars_hosp_frac))) + (pars_gamma_h*Y[13]) + (pars_gamma_h*Y[14])*(10-np.transpose(agepars_crit_die)),
		
			#Y[16] = D population
			pars_gamma_h*Y[14]*np.transpose(agepars_crit_die),
		
			#Y[17] = Hcum population
			(np.transpose(agepars_hosp_frac)*(10-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[12]) + (np.transpose(agepars_hosp_frac)*np.transpose(agepars_hosp_crit)*pars_gamma_s*Y[12]),

# Ages 29-39		 
			#Y[18] = S population = agepars_S_ids
			-pars_beta_a*Y[18]*Ia - pars_beta_s*Y[18]*Is,
		
			#Y[19] = E population
			pars_beta_a*Y[18]*Ia + pars_beta_s*Y[18]*Is - pars_gamma_e*Y[19],
		
			#Y[20] = Ia population
			np.transpose(pars_p)*pars_gamma_e*Y[19]-pars_gamma_a*Y[20],
		
			#Y[21] = Is population
			np.transpose(np.ones(len(pars_p))-pars_p)*pars_gamma_e*Y[21] - pars_gamma_s*Y[21],
		
			#Y[22] = Ihsub population
			np.transpose(agepars_hosp_frac)*(19-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[21] - pars_gamma_h*Y[22],
		
			#Y[23] = Ihcri population
			np.transpose(agepars_hosp_frac)*(19-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[21] - pars_gamma_h*Y[23],
		
			#Y[24] = R population
			(pars_gamma_a*Y[20]) + (pars_gamma_s*Y[21]*(19-np.transpose(agepars_hosp_frac))) + (pars_gamma_h*Y[22]) + (pars_gamma_h*Y[23])*(19-np.transpose(agepars_crit_die)),
		
			#Y[25] = D population
			pars_gamma_h*Y[23]*np.transpose(agepars_crit_die),
		
			#Y[26] = Hcum population
			(np.transpose(agepars_hosp_frac)*(19-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[21]) + (np.transpose(agepars_hosp_frac)*np.transpose(agepars_hosp_crit)*pars_gamma_s*Y[21]),
		
	  
# Ages 39-49		  
			#Y[27] = S population = agepars_S_ids
			-pars_beta_a*Y[27]*Ia - pars_beta_s*Y[27]*Is,
		
			#Y[28] = E population
			pars_beta_a*Y[27]*Ia + pars_beta_s*Y[27]*Is - pars_gamma_e*Y[28],
		
			#Y[29] = Ia population
			np.transpose(pars_p)*pars_gamma_e*Y[28]-pars_gamma_a*Y[29],
		
			#Y[30] = Is population
			np.transpose(np.ones(len(pars_p))-pars_p)*pars_gamma_e*Y[30] - pars_gamma_s*Y[30],
		
			#Y[31] = Ihsub population
			np.transpose(agepars_hosp_frac)*(28-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[30] - pars_gamma_h*Y[31],
		
			#Y[32] = Ihcri population
			np.transpose(agepars_hosp_frac)*(28-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[30] - pars_gamma_h*Y[32],
		
			#Y[33] = R population
			(pars_gamma_a*Y[29]) + (pars_gamma_s*Y[30]*(28-np.transpose(agepars_hosp_frac))) + (pars_gamma_h*Y[31]) + (pars_gamma_h*Y[32])*(28-np.transpose(agepars_crit_die)),
		
			#Y[34] = D population
			pars_gamma_h*Y[32]*np.transpose(agepars_crit_die),
		
			#Y[35] = Hcum population
			(np.transpose(agepars_hosp_frac)*(28-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[30]) + (np.transpose(agepars_hosp_frac)*np.transpose(agepars_hosp_crit)*pars_gamma_s*Y[30]),
	
# Ages 49-59		   
			#Y[36] = S population = agepars_S_ids
			-pars_beta_a*Y[36]*Ia - pars_beta_s*Y[36]*Is,
		
			#Y[37] = E population
			pars_beta_a*Y[36]*Ia + pars_beta_s*Y[36]*Is - pars_gamma_e*Y[37],
		
			#Y[38] = Ia population
			np.transpose(pars_p)*pars_gamma_e*Y[37]-pars_gamma_a*Y[38],
		
			#Y[39] = Is population
			np.transpose(np.ones(len(pars_p))-pars_p)*pars_gamma_e*Y[39] - pars_gamma_s*Y[39],
		
			#Y[40] = Ihsub population
			np.transpose(agepars_hosp_frac)*(37-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[39] - pars_gamma_h*Y[40],
		
			#Y[41] = Ihcri population
			np.transpose(agepars_hosp_frac)*(37-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[39] - pars_gamma_h*Y[41],
		
			#Y[42] = R population
			(pars_gamma_a*Y[38]) + (pars_gamma_s*Y[39]*(37-np.transpose(agepars_hosp_frac))) + (pars_gamma_h*Y[40]) + (pars_gamma_h*Y[41])*(37-np.transpose(agepars_crit_die)),
		
			#Y[43] = D population
			pars_gamma_h*Y[41]*np.transpose(agepars_crit_die),
		
			#Y[44] = Hcum population
			(np.transpose(agepars_hosp_frac)*(37-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[39]) + (np.transpose(agepars_hosp_frac)*np.transpose(agepars_hosp_crit)*pars_gamma_s*Y[39]),

# Ages 59-69		 
			#Y[45] = S population = agepars_S_ids
			-pars_beta_a*Y[45]*Ia - pars_beta_s*Y[45]*Is,
		
			#Y[46] = E population
			pars_beta_a*Y[45]*Ia + pars_beta_s*Y[45]*Is - pars_gamma_e*Y[46],
		
			#Y[47] = Ia population
			np.transpose(pars_p)*pars_gamma_e*Y[46]-pars_gamma_a*Y[47],
		
			#Y[48] = Is population
			np.transpose(np.ones(len(pars_p))-pars_p)*pars_gamma_e*Y[48] - pars_gamma_s*Y[48],
		
			#Y[49] = Ihsub population
			np.transpose(agepars_hosp_frac)*(46-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[48] - pars_gamma_h*Y[49],
		
			#Y[50] = Ihcri population
			np.transpose(agepars_hosp_frac)*(46-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[48] - pars_gamma_h*Y[50],
		
			#Y[51] = R population
			(pars_gamma_a*Y[47]) + (pars_gamma_s*Y[48]*(46-np.transpose(agepars_hosp_frac))) + (pars_gamma_h*Y[49]) + (pars_gamma_h*Y[50])*(46-np.transpose(agepars_crit_die)),
		
			#Y[52] = D population
			pars_gamma_h*Y[50]*np.transpose(agepars_crit_die),
		
			#Y[53] = Hcum population
			(np.transpose(agepars_hosp_frac)*(46-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[48]) + (np.transpose(agepars_hosp_frac)*np.transpose(agepars_hosp_crit)*pars_gamma_s*Y[48]),

# Ages 69-79		 
			#Y[54] = S population = agepars_S_ids
			-pars_beta_a*Y[54]*Ia - pars_beta_s*Y[54]*Is,
		
			#Y[55] = E population
			pars_beta_a*Y[54]*Ia + pars_beta_s*Y[54]*Is - pars_gamma_e*Y[55],
		
			#Y[56] = Ia population
			np.transpose(pars_p)*pars_gamma_e*Y[55]-pars_gamma_a*Y[56],
		
			#Y[57] = Is population
			np.transpose(np.ones(len(pars_p))-pars_p)*pars_gamma_e*Y[57] - pars_gamma_s*Y[57],
		
			#Y[58] = Ihsub population
			np.transpose(agepars_hosp_frac)*(55-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[57] - pars_gamma_h*Y[58],
		
			#Y[59] = Ihcri population
			np.transpose(agepars_hosp_frac)*(55-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[57] - pars_gamma_h*Y[59],
		
			#Y[60] = R population
			(pars_gamma_a*Y[56]) + (pars_gamma_s*Y[57]*(55-np.transpose(agepars_hosp_frac))) + (pars_gamma_h*Y[58]) + (pars_gamma_h*Y[59])*(55-np.transpose(agepars_crit_die)),
		
			#Y[61] = D population
			pars_gamma_h*Y[59]*np.transpose(agepars_crit_die),
		
			#Y[62] = Hcum population
			(np.transpose(agepars_hosp_frac)*(55-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[57]) + (np.transpose(agepars_hosp_frac)*np.transpose(agepars_hosp_crit)*pars_gamma_s*Y[57]),

# Ages 79-89		 
			#Y[63] = S population = agepars_S_ids
			-pars_beta_a*Y[63]*Ia - pars_beta_s*Y[63]*Is,
		
			#Y[64] = E population
			pars_beta_a*Y[63]*Ia + pars_beta_s*Y[63]*Is - pars_gamma_e*Y[64],
		
			#Y[65] = Ia population
			np.transpose(pars_p)*pars_gamma_e*Y[64]-pars_gamma_a*Y[65],
		
			#Y[66] = Is population
			np.transpose(np.ones(len(pars_p))-pars_p)*pars_gamma_e*Y[66] - pars_gamma_s*Y[66],
		
			#Y[67] = Ihsub population
			np.transpose(agepars_hosp_frac)*(64-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[66] - pars_gamma_h*Y[67],
		
			#Y[68] = Ihcri population
			np.transpose(agepars_hosp_frac)*(64-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[66] - pars_gamma_h*Y[68],
		
			#Y[69] = R population
			(pars_gamma_a*Y[65]) + (pars_gamma_s*Y[66]*(64-np.transpose(agepars_hosp_frac))) + (pars_gamma_h*Y[67]) + (pars_gamma_h*Y[68])*(64-np.transpose(agepars_crit_die)),
		
			#Y[70] = D population
			pars_gamma_h*Y[68]*np.transpose(agepars_crit_die),
		
			#Y[71] = Hcum population
			(np.transpose(agepars_hosp_frac)*(64-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[66]) + (np.transpose(agepars_hosp_frac)*np.transpose(agepars_hosp_crit)*pars_gamma_s*Y[66]),
	  
		
# Ages 89-99		
			#Y[72] = S population = agepars_S_ids
			-pars_beta_a*Y[72]*Ia - pars_beta_s*Y[72]*Is,
		
			#Y[73] = E population
			pars_beta_a*Y[72]*Ia + pars_beta_s*Y[72]*Is - pars_gamma_e*Y[73],
		
			#Y[74] = Ia population
			np.transpose(pars_p)*pars_gamma_e*Y[73]-pars_gamma_a*Y[74],
		
			#Y[75] = Is population
			np.transpose(np.ones(len(pars_p))-pars_p)*pars_gamma_e*Y[75] - pars_gamma_s*Y[75],
		
			#Y[76] = Ihsub population
			np.transpose(agepars_hosp_frac)*(73-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[75] - pars_gamma_h*Y[76],
		
			#Y[77] = Ihcri population
			np.transpose(agepars_hosp_frac)*(73-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[75] - pars_gamma_h*Y[77],
		
			#Y[78] = R population
			(pars_gamma_a*Y[74]) + (pars_gamma_s*Y[75]*(73-np.transpose(agepars_hosp_frac))) + (pars_gamma_h*Y[76]) + (pars_gamma_h*Y[77])*(73-np.transpose(agepars_crit_die)),
		
			#Y[79] = D population
			pars_gamma_h*Y[77]*np.transpose(agepars_crit_die),
		
			#Y[80] = Hcum population
			(np.transpose(agepars_hosp_frac)*(73-np.transpose(agepars_hosp_crit))*pars_gamma_s*Y[75]) + (np.transpose(agepars_hosp_frac)*np.transpose(agepars_hosp_crit)*pars_gamma_s*Y[75]),
			
			trigger
			]


# Run solver and initialise solver parameters
u0 = np.concatenate([outbreak_y0_9_19,outbreak_y0_19_29,outbreak_y0_29_39,outbreak_y0_39_49,outbreak_y0_49_59,outbreak_y0_59_69,outbreak_y0_69_79,outbreak_y0_79_89,outbreak_y0_89_99])
tspan = (0., outbreak_pTime)
prob = de.ODEProblem(f, u0, tspan)
sol = de.solve(prob)

plt.plot(sol.t,sol.u)
plt.show()


# Statistics from simulation

# In[27]:


stats_Hcum = sum(sol.u[8],sol.u[16],sol.u[24],sol.u[32],sol.u[40],sol.u[48],sol.u[56],sol.u[64],sol.u[72],sol.u[80])
stats_Dcum = sum(sol.u[7],sol.u[15],sol.u[23],sol.u[31],sol.u[39],sol.u[47],sol.u[55],sol.u[63],sol.u[71],sol.u[79])


# ## Rough code below - please ignore

# In[ ]:


# Simulating ODE model - COMPLETE - ODE INITIAL CONDS and SOLVER SETTINGS


# In[3]:


ts = np.linspace(0, outbreak_pTime, 0.1)
Ys = odeint(dY_dt, outbreak_y0, ts) #solver settings not as defined as per matlab

#Simulation statistics
stats_Dcum = sum(Ys[:,7])
stats_Hcum = sum(Ys[:,8])

#% Sims - Get to Crossing
#opts=odeset('reltol',1e-8,'maxstep',0.1,'events',@intervene_trigger);
#[tpre,ypre,te,ye,ie]=ode45(@covid_model_ga,[0:1:outbreak.pTime], outbreak.y0,opts,pars,agepars);


# In[2]:


Ys[1]


# In[ ]:


fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
				   xticklabels=[], ylim=(0, 10))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
				   ylim=(0, 10))


# Plotting

# In[ ]:


fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4],
				   xticklabels=[], ylim=(0, 10))
ax2 = fig.add_axes([0.1, 0.1, 0.8, 0.4],
				   ylim=(0, 10))

x = np.linspace(0, 10)
ax1.plot(Ps[:,6])
ax2.plot(Ps[:,7]);
plt.xlabel("Time")
plt.ylabel("Absolute Cell Number")
plt.legend();


# In[ ]:




