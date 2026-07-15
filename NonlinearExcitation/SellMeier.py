# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 14:14:47 2026

@author: giber
"""

import numpy as np
import matplotlib.pyplot as plt



'''
To compute the SellMeyer equations for the wavelength
dependence of the refractive index

https://calcmountain.com/calculators/optics/sellmeier-equation



'''

c_speed = 3e8 ### m/s
tau0 = 30e-15 ## s  initial pulse width
z    =   1e-1   ### m  propagation length



def SellMeier(B,C,x):
  # x is wavelength in um
  # B and C 3D vectors
  ###  this is the pure phase Ref. Ind.
  S = np.zeros((3))
  RI = np.zeros((len(x)))

  for i in range(len(x)):
    for k in range(3):
      S[k] = B[k]*x[i]**2/(x[i]**2-C[k])
    RI[i] = np.sqrt( 1 + np.sum(S))

  return RI

################first derivative RI
def Der1_SElleMeier(B,C,x):
  # x is wavelength in um
  # B and C 3D vectors
  ## first derivative with respect to lambda
  RI_sm     = SellMeier(B,C,x)

  S         = np.zeros((3))
  RI_der1   = np.zeros((len(x)))

  for i in range(len(x)):
    for k in range(3):
      S[k] = - 2 * B[k] * x[i] * C[k] / (x[i]**2-C[k])**2
    RI_der1[i] = np.sum(S) / 2 / RI_sm[i]

  return RI_der1

################first derivative RI
def Der2_SElleMeier(B,C,x):
  # x is wavelength in um
  # B and C 3D vectors
  RI_sm     = SellMeier(B,C,x)
  RI_der1_sm= Der1_SElleMeier(B,C,x)
  S         = np.zeros((3))
  RI_der2   = np.zeros((len(x)))

  for i in range(len(x)):
    for k in range(3):
      S[k] =  B[k] * C[k] * (x[i]**2 + C[k]) / (x[i]**2-C[k])**3
    RI_der2[i] = np.sum(S) / RI_sm[i]

  return RI_der1_sm**2 + RI_der2
############################################ 

def Compute_vgroup(x,y):
   dRI_dx    = np.gradient(y, x)
   RI_group  = y - x*dRI_dx
   v_group   = 1/RI_group
   return v_group
 ############################################ 
def Compute_vgroup_direct(x,B,C):
   y         = SellMeier(B,C,x)
   dRI_dx    = Der1_SElleMeier(B,C,x)
   RI_group  = y - x*dRI_dx
   v_group   = 1/RI_group
   return v_group
 ############################################ 
############################################ 
def Compute_beta2_direct(x,B,C):
   RIy       = SellMeier(B,C,x)  ### Ref. Ind.
   dRI_dx    = Der1_SElleMeier(B,C,x)  ### dn/dlambda
   dRI2_d2x  = Der2_SElleMeier(B,C,x)
   
   beta2     = x**3 * dRI2_d2x / 2/ np.pi / c_speed**2
###### the wavelength is in um, therefore I must multiply by 
######  1e-18  !!!!!
   return beta2
 ############################################ 
def TauSpread(tau0,b2,z):
    LD = tau0**2 / b2
    return tau0 * np.sqrt(1+(z/LD)**2)
   
##################################################
##################################################
##################################################
##################################################
##################################################
##################################################

wavmin  = 0.350 ### in um !!!!!!!!!!!!!!!!!!!!
wavmax  = 0.650
dwav    = 0.005
nwav    = int((wavmax-wavmin)/dwav)

wav = np.linspace(wavmin,wavmax,nwav)

beta2_factor = 1e24/1e-3 ###  ps**2/km

print ('number of elements in wave os ', nwav)

#B_BK7   = [1.03961212,0.231792344,1.01046945]
#C_BK7   = [6.00069867e-3,2.00179144e-2,111.886764]
B_BK7   = [1.03961212,0.231792344,1.01046945]
C_BK7   = [6.00069867e-3,2.00179144e-2,103.56]

B_SF2   = [1.34533359,0.209073176,0.937357162]
C_SF2   = [0.00997743871,0.0470450767,1.03560653e2]

B_SF11  = [1.73848, 0.31117, 1.17491]
C_SF11   = [0.013607, 0.06159,121.92271]

RI_BK7  = SellMeier(B_BK7,C_BK7,wav)
RI_SF2  = SellMeier(B_SF2,C_SF2,wav)
RI_SF11 = SellMeier(B_SF11,C_SF11,wav)

v_phase_BK7  = 1/RI_BK7
v_phase_SF2  = 1/RI_SF2
v_phase_SF11 = 1/RI_SF11

RI_der1_BK7 = Der1_SElleMeier(B_BK7,C_BK7,wav)
RI_der1_SF2 = Der1_SElleMeier(B_SF2,C_SF2,wav)
RI_der1_SF11= Der1_SElleMeier(B_SF11,C_SF11,wav)

RI_der2_BK7 = Der2_SElleMeier(B_BK7,C_BK7,wav)
RI_der2_SF2 = Der2_SElleMeier(B_SF2,C_SF2,wav)
RI_der2_SF11= Der2_SElleMeier(B_SF11,C_SF11,wav)

fonts = 20
v_group_BK7   = Compute_vgroup(wav,RI_BK7)
v_group_SF2   = Compute_vgroup(wav,RI_SF2)
v_group_SF11  = Compute_vgroup(wav,RI_SF11)
v_group_BK7_d = Compute_vgroup_direct(wav,B_BK7,C_BK7)
v_group_SF2_d = Compute_vgroup_direct(wav,B_SF2,C_SF2)
v_group_SF11_d= Compute_vgroup_direct(wav,B_SF11,C_SF11)

### beta2 is proportional to wave**3, wav are in um.
beta2_BK7     = 1e-6* Compute_beta2_direct(wav,B_BK7,C_BK7)
beta2_SF2     = 1e-6* Compute_beta2_direct(wav,B_SF2,C_SF2)
beta2_SF11    = 1e-6* Compute_beta2_direct(wav,B_SF11,C_SF11)

Tau_BK7 = TauSpread(tau0,beta2_BK7,z)
Tau_SF2 = TauSpread(tau0,beta2_SF2,z)
Tau_SF11 = TauSpread(tau0,beta2_SF11,z)


plt.plot(wav,v_phase_BK7,label='BK7 v_phase')
plt.plot(wav,v_group_BK7,label='BK7 v_group')
plt.plot(wav,v_group_BK7_d,'o',alpha=0.2,label='BK7 v_group_dir')
plt.plot(wav,v_phase_SF2,label='SF2 v_phase')
plt.plot(wav,v_group_SF2,label='SF2 v_group')
plt.plot(wav,v_group_SF2_d,'o',alpha=0.2,label='SF2 v_group_der')
plt.plot(wav,v_phase_SF11,label='SF11 v_phase')
plt.plot(wav,v_group_SF11,label='SF11 v_group')
plt.plot(wav,v_group_SF11_d,'o',alpha=0.2,label='SF11 v_group_der')

plt.xlabel('Wavelength (um)',fontsize=fonts)
plt.ylabel('v/c',fontsize=fonts)
plt.tick_params(axis='both', labelsize=14)



plt.legend()
plt.show()

Fig, ax = plt.subplots()
ax.plot(wav,RI_BK7,label='RI of BK7')
ax.plot(wav,RI_SF2,label='RI of SF2')
ax.plot(wav,RI_SF11,label='RI of SF11')
ax.set_xlabel('Wavelength (um)',fontsize=fonts)
ax.set_ylabel('RI',fontsize=fonts)
ax.tick_params(axis='both', labelsize=14)


# Create right y-axis
ax2 = ax.twinx()
ax2.plot(wav, beta2_factor *  beta2_BK7 , color='red', linestyle='--', label='beta2 BK7')
ax2.plot(wav, beta2_factor *  beta2_SF2 , color='green', linestyle='--', label='beta2 SF2')
ax2.plot(wav, beta2_factor *  beta2_SF11 , color='blue', linestyle='--', label='beta2 SF11')
ax2.set_ylabel('beta2 [fs**2/mm]', fontsize=fonts, color='red',x = 1.75)
ax2.tick_params(axis='y', labelcolor='red', labelsize=14)

# Combine legends from both axes
lines_1, labels_1 = ax.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.tight_layout()
plt.show()



#######
wave0 = 0.60 ##  um
idx = np.argmin(np.abs(wav - wave0))

print ('RI_BK7 at ',wav[idx],' um is: ', RI_BK7[idx])
print ('RI_SF2 at ',wav[idx],' um is: ', RI_SF2[idx])
print ('RI_SF11 at ',wav[idx],' um is: ', RI_SF11[idx])

print ('RIg_BK7 at ',wav[idx],' um is: ', 1./v_group_BK7_d[idx])
print ('RIg_SF2 at ',wav[idx],' um is: ', 1./v_group_SF2_d[idx])
print ('RIg_SF11 at ',wav[idx],' um is: ', 1./v_group_SF11_d[idx])

print ('beta2_BK7  at ',wav[idx],' um is: ', beta2_factor *beta2_BK7[idx])
print ('beta2_SF2  at ',wav[idx],' um is: ', beta2_factor *beta2_SF2[idx])
print ('beta2_SF11  at ',wav[idx],' um is: ', beta2_factor *beta2_SF11[idx])

print ('----- NOW evaluate spread of the pusle width')
print ('----- ASSUME pulswe width = ',tau0*1e15,' [fs]')
print ('tau_BK7  at ',wav[idx],' um  for ',z*1e3,' mm, is: ', 1e15 * Tau_BK7[idx])
print ('tau_SF2  at ',wav[idx],' um  for ',z*1e3,' mm, is ', 1e15 * Tau_SF2[idx])
print ('tau_SF11  at ',wav[idx],' um  for ',z*1e3,' mm, is: ', 1e15 * Tau_SF11[idx])

print ( beta2_BK7[idx]*beta2_factor)
print (beta2_factor)

