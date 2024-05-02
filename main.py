import numpy as np
import healpy as hp

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

from scipy import stats

import params
from utils import Sky_SEDs
from utils import Sky_Cls
from utils import Instrument
from utils import Analysis

import time

referee_suggestion = True

start = time.time()

plt.rcParams.update({
    "font.size":10,
    "text.usetex":True,
    "font.family":"serif", 
    "font.serif":"cm"
    })  
    
col_FGs = '#D95D39'    
col_cmb = '#0E1428'
col_dus = '#F18805'
col_syn = '#F5B22B'
col_sol = '#78C2A9'
col_noise = '#BABABA'

col_fit = '#FFB948'
col_likelihood = 'gray'
col_r = 'gray'
col_sigma = 'gray'

r_sim_temp = params.r_sim
A_sim = params.A_sim

r_in = 0.00461
#r_in = 0.

num = 0.002*(1+2*r_in/0.032)

idx = np.where((r_sim_temp>=(r_in-num)) & (r_sim_temp<=(r_in+num)))
r_sim = r_sim_temp[idx]

print(len(r_sim))

analysis = Analysis(r_in = r_in, NSIDE = params.NSIDE, params_path = params.params_path, telescopes = params.telescopes, chan_dicts = params.chan_dicts, 
                    plotting = True, ideal = True, gain_calibration = True, fews = True)

ell = np.arange(2,analysis.lmax+1)
C2D = ell*(ell+1)/(2*np.pi) 

B_cov_all, B_cov_FGs_noise, N_cov_model = analysis.compute_covs()
B_w = analysis.compute_weights(B_cov_all)
        
for i in np.arange(2,len(B_w)):
    if not np.isclose(np.sum(B_w[i]),1):
        print('for ell='+str(i)+' the sum of the weights is '+str(np.sum(B_w[i])))
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FIGURE 8 (HILC's weights)

cm = matplotlib.colormaps['viridis']

if referee_suggestion:
    fig = plt.figure(figsize=(2.8,2.6))

    colors = [cm(.0), cm(.09), cm(.18), cm(.27), cm(.36), cm(.45), cm(.53), cm(.64), cm(.73), cm(.82), cm(.90), cm(.99)]
    yticks = [40, 50, 60, 68, 68, 78, 78, 89, 89, 100, 119, 140]
    for c, k in zip(colors, np.arange(len(yticks))):
        xs = ell
        ys = B_w[2:analysis.lmax+1,k]
        zs = yticks[k]

        #if np.any(ys < 0):
            #print('LFT: '+str(zs)+' GHz has negative values')
        
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        plt.plot(xs, ys, color=c, alpha=0.8)#, label=str(zs)+' GHz')

    plt.ylim([-0.17,0.17])
    plt.xlabel(r'$\ell$', labelpad=5)
    plt.title(r'LFT weights', fontsize = 11)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    linemin = Line2D([0], [0], label='40 GHz', color=colors[0])
    linemax = Line2D([0], [0], label='140 GHz', color=colors[-1])
    handles.extend([linemin, linemax])
    plt.legend(handles=handles, loc='lower right')

    plt.tight_layout()
    plt.savefig('output/weights_IDEAL_LFT.pdf')
    plt.clf()
    
    ###

    fig = plt.figure(figsize=(2.8,2.6))

    colors = [cm(.0), cm(.25), cm(.5), cm(.75), cm(.99)]
    yticks = [100, 119, 140, 166, 195]
    for c, k in zip(colors, np.arange(len(yticks))):
        xs = ell
        ys = B_w[2:analysis.lmax+1,12+k]
        zs = yticks[k]

        #if np.any(ys < 0):
            #print('MFT: '+str(zs)+' GHz has negative values')
        
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        plt.plot(xs, ys, color=c, alpha=0.8)#, label=str(zs)+' GHz')

    plt.ylim([-0.199,0.7])
    plt.xlabel(r'$\ell$', labelpad=5)
    plt.title(r'MFT weights', fontsize = 11)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    linemin = Line2D([0], [0], label='100 GHz', color=colors[0])
    linemax = Line2D([0], [0], label='195 GHz', color=colors[-1])
    handles.extend([linemin, linemax])
    plt.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('output/weights_IDEAL_MFT.pdf')
    plt.clf()
    
    ###
    
    fig = plt.figure(figsize=(2.8,2.6))

    colors = [cm(.0), cm(.25), cm(.5), cm(.75), cm(.99)]
    yticks = [195, 235, 280, 337, 402]
    for c, k in zip(colors, np.arange(len(yticks))):
        xs = ell
        ys = B_w[2:analysis.lmax+1,17+k]
        zs = yticks[k]

        #if np.any(ys < 0):
            #print('HFT: '+str(zs)+' GHz has negative values')
        
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        plt.plot(xs, ys, color=c, alpha=0.8)#, label=str(zs)+' GHz')

    plt.ylim([-0.09,0.29])
    plt.xlabel(r'$\ell$', labelpad=5)
    plt.title(r'HFT weights', fontsize = 11)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    linemin = Line2D([0], [0], label='195 GHz', color=colors[0])
    linemax = Line2D([0], [0], label='402 GHz', color=colors[-1])
    handles.extend([linemin, linemax])
    plt.legend(handles=handles, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('output/weights_IDEAL_HFT.pdf')
    plt.clf()   

else:
    fig = plt.figure(figsize=(2.8,2.6))
    ax1 = plt.subplot2grid((1, 1), (0, 0), projection='3d')

    colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1), cm(.3), cm(.5), cm(.7), cm(.9), cm(.7),
              cm(.5), cm(.3)]
    yticks = [40, 50, 60, 68, 68, 78, 78, 89, 89, 100, 119, 140]
    for c, k in zip(colors, np.arange(len(yticks))):
        xs = ell
        ys = B_w[2:analysis.lmax+1,k]
        zs = yticks[k]
        cs = [c] * len(xs)

        #if np.any(ys < 0):
            #print('LFT: '+str(zs)+' GHz has negative values')
        
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        ax1.bar(xs, ys, zs, zdir='x', color=cs, alpha=0.8)

    ax1.set_xticks(yticks, labels=['40', '', '', '', '', '', '', '', '', '', '', '140'])
    ax1.tick_params(axis='x', which='major', pad=-3)
    ax1.set_xlabel('Frequency [GHz]', labelpad=-3)
    ax1.set_zlim([-0.05,0.15])
    ax1.tick_params(axis='y', which='major', pad=-3)
    ax1.set_ylabel(r'$\ell$', labelpad=-3)
    ax1.set_title(r'LFT weights', fontsize = 11, pad=-14)
    plt.savefig('output/weights_IDEAL_LFT.pdf')
    plt.clf()

    ###
    
    fig = plt.figure(figsize=(2.8,2.6))
    ax2 = plt.subplot2grid((1, 1), (0, 0), projection='3d')
    
    colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1)]
    yticks = [100, 119, 140, 166, 195]
    for c, k in zip(colors, np.arange(len(yticks))):
        xs = ell
        ys = B_w[2:analysis.lmax+1, 12+k]
        zs = yticks[k]
        cs = [c] * len(xs)

        #if np.any(ys < 0):
            #print('MFT: '+str(zs)+' GHz has negative values')
        
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        ax2.bar(xs, ys, zs, zdir='x', color=cs, alpha=0.8)

    ax2.set_xticks(yticks, labels=['100', '', '', '', '195'])
    ax2.tick_params(axis='x', which='major', pad=-3)
    ax2.set_xlabel('Frequency [GHz]', labelpad=-3)
    ax2.set_zlim([-0.05,0.35])
    ax2.tick_params(axis='y', which='major', pad=-3)
    ax2.set_ylabel(r'$\ell$', labelpad=-3)
    ax2.set_title(r'MFT weights', fontsize = 11, pad=-14)
    plt.savefig('output/weights_IDEAL_MFT.pdf')
    plt.clf()
    
    ###

    fig = plt.figure(figsize=(2.8,2.6))
    ax3 = plt.subplot2grid((1, 1), (0, 0), projection='3d')
    
    colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1)]
    yticks = [195, 235, 280, 337, 402]
    for c, k in zip(colors, np.arange(len(yticks))):
        xs = ell
        ys = B_w[2:analysis.lmax+1, 17+k]
        zs = yticks[k]
        cs = [c] * len(xs)
    
        #if np.any(ys < 0):
            #print('HFT: '+str(zs)+' GHz has negative values')
            
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        ax3.bar(xs, ys, zs, zdir='x', color=cs, alpha=0.8)
    
    ax3.set_xticks(yticks, labels=['195', '', '', '', '402'])
    ax3.tick_params(axis='x', which='major', pad=-3)
    ax3.set_xlabel('Frequency [GHz]', labelpad=-3)
    ax3.set_zlim([-0.05,0.25])
    ax3.tick_params(axis='y', which='major', pad=-3)
    ax3.set_ylabel(r'$\ell$', labelpad=-3)
    ax3.set_title(r'HFT weights', fontsize = 11, pad=-14)
    plt.savefig('output/weights_IDEAL_HFT.pdf')
    plt.clf()
   
BBls_HILC_all = analysis.compute_HILC_solution(B_w, B_cov_all)
BBls_HILC_FGs_noise = analysis.compute_HILC_solution(B_w, B_cov_FGs_noise)
BBls_HILC_noiseonly = analysis.compute_HILC_solution(B_w, N_cov_model) 

Cls_cmb = analysis.Cls_cmb
Cls_cmb_tensor = analysis.Cls_cmb_tensor
Cls_cmb_scalar = analysis.Cls_cmb_scalar
Cls_dus = analysis.Cls_dus
Cls_syn = analysis.Cls_syn

fig = plt.figure(figsize=(5,3.5))

plt.loglog(ell,Cls_dus[2,2:]*C2D, label='input dust', color = col_dus)#, linestyle=':')
plt.loglog(ell,Cls_syn[2,2:]*C2D, label='input synchrotron', color = col_syn)#, linestyle=':')
plt.loglog(ell,Cls_cmb[2,2:]*C2D, label='input CMB', color = col_cmb)#, linestyle=':')
plt.loglog(ell,BBls_HILC_noiseonly[2:]*C2D, label='noise only', color = col_noise, linestyle=':')
plt.loglog(ell,((BBls_HILC_FGs_noise-BBls_HILC_noiseonly))[2:]*C2D, label='FGs residual', color = col_FGs, linestyle=':')
plt.loglog(ell,((BBls_HILC_all-BBls_HILC_noiseonly))[2:]*C2D, label='HILC solution', color = col_sol, linestyle='--')
handles, labels = plt.gca().get_legend_handles_labels()
order = [5,2,0,1,4,3]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'lower right')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell^{BB}$ [$\mu$K$^2$]')
plt.tight_layout()
ylim_IDEAL = plt.ylim()
plt.savefig('output/HILC_IDEAL.pdf')
plt.clf()

lmax_beam = 2+np.argmin(np.abs(BBls_HILC_noiseonly[2:] - Cls_cmb[2,2:]))
print('the noise residual intersects the theoretical CMB spectrum at ell='+str(lmax_beam))

lmax_likelihood = 200
ell_likelihood = np.arange(2,lmax_likelihood+1)
C2D_new = ell_likelihood*(ell_likelihood+1)/(2*np.pi)      
        
r_fit, sigma_r, likelihood_r, A_fit, sigma_A, likelihood_A = analysis.estimate_r(BBls_HILC_all, ell_likelihood, 1, Cls_cmb_tensor[2], Cls_cmb_scalar[2] + BBls_HILC_noiseonly, r_sim, A_sim)           
print('r = ', r_fit)
print('sigma = ', sigma_r)

print('A = ', A_fit)
print('sigma = ', sigma_A)

likelihood_r_IDEAL = likelihood_r
r_fit_IDEAL = r_fit
sigma_r_IDEAL = sigma_r

likelihood_A_IDEAL = likelihood_A
A_fit_IDEAL = A_fit
sigma_A_IDEAL = sigma_A

#################

print('#################################')

#################

analysis = Analysis(r_in = r_in, NSIDE = params.NSIDE, params_path = params.params_path, telescopes = params.telescopes, chan_dicts = params.chan_dicts, 
                    plotting = True, ideal = False, gain_calibration = True, fews = False)

B_cov_all, B_cov_FGs_noise, N_cov_model, B_cov_cmb_rho, B_cov_cmb_eta, B_cov_dus_rho, B_cov_dus_eta, B_cov_syn_rho, B_cov_syn_eta = analysis.compute_covs()
B_w = analysis.compute_weights(B_cov_all)
        
#for i in np.arange(2,len(B_w)):
    #if not np.isclose(np.sum(B_w[i]),1):
        #print('for ell='+str(i)+' the sum of the weights is '+str(np.sum(B_w[i])))
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FIGURE 8 (HILC's weights)
        
plt.close('all')  

if referee_suggestion:
    fig = plt.figure(figsize=(2.8,2.6))

    colors = [cm(.0), cm(.09), cm(.18), cm(.27), cm(.36), cm(.45), cm(.53), cm(.64), cm(.73), cm(.82), cm(.90), cm(.99)]
    yticks = [40, 50, 60, 68, 68, 78, 78, 89, 89, 100, 119, 140]
    for c, k in zip(colors, np.arange(len(yticks))):
        xs = ell
        ys = B_w[2:analysis.lmax+1,k]
        zs = yticks[k]

        #if np.any(ys < 0):
            #print('LFT: '+str(zs)+' GHz has negative values')
        
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        plt.plot(xs, ys, color=c, alpha=0.8)#, label=str(zs)+' GHz')

    plt.ylim([-0.17,0.17])
    plt.xlabel(r'$\ell$', labelpad=5)
    plt.title(r'LFT weights', fontsize = 11)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    linemin = Line2D([0], [0], label='40 GHz', color=colors[0])
    linemax = Line2D([0], [0], label='140 GHz', color=colors[-1])
    handles.extend([linemin, linemax])
    plt.legend(handles=handles, loc='lower right')

    
    plt.tight_layout()
    plt.savefig('output/weights_LFT.pdf')
    plt.clf()
    
    ###

    fig = plt.figure(figsize=(2.8,2.6))

    colors = [cm(.0), cm(.25), cm(.5), cm(.75), cm(.99)]
    yticks = [100, 119, 140, 166, 195]
    for c, k in zip(colors, np.arange(len(yticks))):
        xs = ell
        ys = B_w[2:analysis.lmax+1,12+k]
        zs = yticks[k]

        #if np.any(ys < 0):
            #print('MFT: '+str(zs)+' GHz has negative values')
        
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        plt.plot(xs, ys, color=c, alpha=0.8)#, label=str(zs)+' GHz')

    plt.ylim([-0.199,0.7])
    plt.xlabel(r'$\ell$', labelpad=5)
    plt.title(r'MFT weights', fontsize = 11)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    linemin = Line2D([0], [0], label='100 GHz', color=colors[0])
    linemax = Line2D([0], [0], label='195 GHz', color=colors[-1])
    handles.extend([linemin, linemax])
    plt.legend(handles=handles, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('output/weights_MFT.pdf')
    plt.clf()
    
    ###
    
    fig = plt.figure(figsize=(2.8,2.6))

    colors = [cm(.0), cm(.25), cm(.5), cm(.75), cm(.99)]
    yticks = [195, 235, 280, 337, 402]
    for c, k in zip(colors, np.arange(len(yticks))):
        xs = ell
        ys = B_w[2:analysis.lmax+1,17+k]
        zs = yticks[k]

        #if np.any(ys < 0):
            #print('HFT: '+str(zs)+' GHz has negative values')
        
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        plt.plot(xs, ys, color=c, alpha=0.8)#, label=str(zs)+' GHz')

    plt.ylim([-0.09,0.29])
    plt.xlabel(r'$\ell$', labelpad=5)
    plt.title(r'HFT weights', fontsize = 11)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    linemin = Line2D([0], [0], label='195 GHz', color=colors[0])
    linemax = Line2D([0], [0], label='402 GHz', color=colors[-1])
    handles.extend([linemin, linemax])
    plt.legend(handles=handles, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('output/weights_HFT.pdf')
    plt.clf()   

else:
    fig = plt.figure(figsize=(2.8,2.6))
    ax1 = plt.subplot2grid((1, 1), (0, 0), projection='3d')

    colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1), cm(.3), cm(.5), cm(.7), cm(.9), cm(.7),
              cm(.5), cm(.3)]
    yticks = [40, 50, 60, 68, 68, 78, 78, 89, 89, 100, 119, 140]
    for c, k in zip(colors, np.arange(len(yticks))):
        xs = ell
        ys = B_w[2:analysis.lmax+1,k]
        zs = yticks[k]
        cs = [c] * len(xs)

        #if np.any(ys < 0):
            #print('LFT: '+str(zs)+' GHz has negative values')
        
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        ax1.bar(xs, ys, zs, zdir='x', color=cs, alpha=0.8)

    ax1.set_xticks(yticks, labels=['40', '', '', '', '', '', '', '', '', '', '', '140'])
    ax1.tick_params(axis='x', which='major', pad=-3)
    ax1.set_xlabel('Frequency [GHz]', labelpad=-3)
    ax1.set_zlim([-0.05,0.15])
    ax1.tick_params(axis='y', which='major', pad=-3)
    ax1.set_ylabel(r'$\ell$', labelpad=-3)
    ax1.set_title(r'LFT weights', fontsize = 11, pad=-14)
    plt.savefig('output/weights_LFT.pdf')
    plt.clf()

    ###
    
    fig = plt.figure(figsize=(2.8,2.6))
    ax2 = plt.subplot2grid((1, 1), (0, 0), projection='3d')
    
    colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1)]
    yticks = [100, 119, 140, 166, 195]
    for c, k in zip(colors, np.arange(len(yticks))):
        xs = ell
        ys = B_w[2:analysis.lmax+1, 12+k]
        zs = yticks[k]
        cs = [c] * len(xs)

        #if np.any(ys < 0):
            #print('MFT: '+str(zs)+' GHz has negative values')
        
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        ax2.bar(xs, ys, zs, zdir='x', color=cs, alpha=0.8)

    ax2.set_xticks(yticks, labels=['100', '', '', '', '195'])
    ax2.tick_params(axis='x', which='major', pad=-3)
    ax2.set_xlabel('Frequency [GHz]', labelpad=-3)
    ax2.set_zlim([-0.05,0.35])
    ax2.tick_params(axis='y', which='major', pad=-3)
    ax2.set_ylabel(r'$\ell$', labelpad=-3)
    ax2.set_title(r'MFT weights', fontsize = 11, pad=-14)
    plt.savefig('output/weights_MFT.pdf')
    plt.clf()
    
    ###

    fig = plt.figure(figsize=(2.8,2.6))
    ax3 = plt.subplot2grid((1, 1), (0, 0), projection='3d')
    
    colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1)]
    yticks = [195, 235, 280, 337, 402]
    for c, k in zip(colors, np.arange(len(yticks))):
        xs = ell
        ys = B_w[2:analysis.lmax+1, 17+k]
        zs = yticks[k]
        cs = [c] * len(xs)
    
        #if np.any(ys < 0):
            #print('HFT: '+str(zs)+' GHz has negative values')
            
        # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
        ax3.bar(xs, ys, zs, zdir='x', color=cs, alpha=0.8)
    
    ax3.set_xticks(yticks, labels=['195', '', '', '', '402'])
    ax3.tick_params(axis='x', which='major', pad=-3)
    ax3.set_xlabel('Frequency [GHz]', labelpad=-3)
    ax3.set_zlim([-0.05,0.25])
    ax3.tick_params(axis='y', which='major', pad=-3)
    ax3.set_ylabel(r'$\ell$', labelpad=-3)
    ax3.set_title(r'HFT weights', fontsize = 11, pad=-14)
    plt.savefig('output/weights_HFT.pdf')
    plt.clf()

'''
#cm = matplotlib.colormaps['viridis']

fig = plt.figure(figsize=(2.8,2.6))
ax1 = plt.subplot2grid((1, 1), (0, 0), projection='3d')

colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1), cm(.3), cm(.5), cm(.7), cm(.9), cm(.7),
          cm(.5), cm(.3)]
yticks = [40, 50, 60, 68, 68, 78, 78, 89, 89, 100, 119, 140]
for c, k in zip(colors, np.arange(len(yticks))):
    xs = ell
    ys = B_w[2:analysis.lmax+1,k]
    zs = yticks[k]
    cs = [c] * len(xs)
    
    if np.any(ys < 0):
        print('LFT: '+str(zs)+' GHz has negative values')

    # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
    ax1.bar(xs, ys, zs, zdir='x', color=cs, alpha=0.8)

ax1.set_xticks(yticks, labels=['40', '', '', '', '', '', '', '', '', '', '', '140'])
ax1.tick_params(axis='x', which='major', pad=-3)
ax1.set_xlabel('Frequency [GHz]', labelpad=-3)
ax1.set_zlim([-0.10,0.15])
ax1.tick_params(axis='y', which='major', pad=-3)
ax1.set_ylabel(r'$\ell$', labelpad=-3)
ax1.set_title(r'LFT weights', fontsize = 11, pad=-14)
plt.savefig('output/weights_LFT.pdf')
plt.clf()

###

fig = plt.figure(figsize=(2.8,2.6))
ax2 = plt.subplot2grid((1, 1), (0, 0), projection='3d')

colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1)]
yticks = [100, 119, 140, 166, 195]
for c, k in zip(colors, np.arange(len(yticks))):
    xs = ell
    ys = B_w[2:analysis.lmax+1, 12+k]
    zs = yticks[k]
    cs = [c] * len(xs)

    if np.any(ys < 0):
        print('MFT: '+str(zs)+' GHz has negative values')
        
    # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
    ax2.bar(xs, ys, zs, zdir='x', color=cs, alpha=0.8)

ax2.set_xticks(yticks, labels=['100', '', '', '', '195'])
ax2.tick_params(axis='x', which='major', pad=-3)
ax2.set_xlabel('Frequency [GHz]', labelpad=-3)
ax2.set_zlim([-0.05,0.35])
ax2.tick_params(axis='y', which='major', pad=-3)
ax2.set_ylabel(r'$\ell$', labelpad=-3)
ax2.set_title(r'MFT weights', fontsize = 11, pad=-14)
plt.savefig('output/weights_MFT.pdf')
plt.clf()

###

fig = plt.figure(figsize=(2.8,2.6))
ax3 = plt.subplot2grid((1, 1), (0, 0), projection='3d')

colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1)]
yticks = [195, 235, 280, 337, 402]
for c, k in zip(colors, np.arange(len(yticks))):
    xs = ell
    ys = B_w[2:analysis.lmax+1, 17+k]
    zs = yticks[k]
    cs = [c] * len(xs)

    if np.any(ys < 0):
        print('HFT: '+str(zs)+' GHz has negative values')
       
    # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
    ax3.bar(xs, ys, zs, zdir='x', color=cs, alpha=0.8)

ax3.set_xticks(yticks, labels=['195', '', '', '', '402'])
ax3.tick_params(axis='x', which='major', pad=-3)
ax3.set_xlabel('Frequency [GHz]', labelpad=-3)
ax3.set_zlim([-0.05,0.25])
ax3.tick_params(axis='y', which='major', pad=-3)
ax3.set_ylabel(r'$\ell$', labelpad=-3)
ax3.set_title(r'HFT weights', fontsize = 11, pad=-14)
plt.savefig('output/weights_HFT.pdf')
plt.clf()
'''
   
BBls_HILC_all = analysis.compute_HILC_solution(B_w, B_cov_all)
BBls_HILC_FGs_noise = analysis.compute_HILC_solution(B_w, B_cov_FGs_noise)
BBls_HILC_noiseonly = analysis.compute_HILC_solution(B_w, N_cov_model)

fig = plt.figure(figsize=(5,3.5))

plt.loglog(ell,Cls_dus[2,2:]*C2D, label='input dust', color = col_dus)
plt.loglog(ell,Cls_syn[2,2:]*C2D, label='input synchrotron', color = col_syn)
plt.loglog(ell,Cls_cmb[2,2:]*C2D, label='input CMB', color = col_cmb)
plt.loglog(ell,BBls_HILC_noiseonly[2:]*C2D, label='noise only', color = col_noise, linestyle=':')
plt.loglog(ell,((BBls_HILC_FGs_noise-BBls_HILC_noiseonly))[2:]*C2D, label='FGs residual', color = col_FGs, linestyle=':')
plt.loglog(ell,((BBls_HILC_all-BBls_HILC_noiseonly))[2:]*C2D, label='HILC solution', color = col_sol, linestyle='--')
handles, labels = plt.gca().get_legend_handles_labels()
order = [5,2,0,1,4,3]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'lower right')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell^{BB}$ [$\mu$K$^2$]')
plt.tight_layout()
plt.ylim(ylim_IDEAL)
plt.savefig('output/HILC.pdf')
plt.clf()

####### FIXME: implement bar((rho_cmb/g_cmb)**2) as output of some function!!!

fig = plt.figure(figsize=(4,3))

#overlapping_ells = ell[np.argwhere(np.isclose(0.943033768824104*Cls_cmb[2,2:], (BBls_HILC_all-BBls_HILC_noiseonly)[2:]))]
#print(np.min(overlapping_ells))
#print(np.max(overlapping_ells))

'''
plt.loglog(ell,Cls_cmb[2,2:]*C2D, label='input CMB', color = col_cmb)
plt.loglog(ell,0.8789368174180908*Cls_cmb[2,2:]*C2D, label='input CMB, rescaled', color = 'red')
plt.loglog(ell,((BBls_HILC_all-BBls_HILC_noiseonly))[2:]*C2D, label='HILC solution', color = col_sol, linestyle='--')
handles, labels = plt.gca().get_legend_handles_labels()
plt.ylim(top=0.1)
plt.xlim(right=np.max(overlapping_ells))
order = [0,1,2]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])#, loc = 'lower right')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell^{BB}$ [$\mu$K$^2$]')
plt.tight_layout()
plt.savefig('output/HILC_closeup.pdf')
plt.clf()
'''

BBls_HILC_cmb_rhoonly = analysis.compute_HILC_solution(B_w, B_cov_cmb_rho)
BBls_HILC_cmb_etaonly = analysis.compute_HILC_solution(B_w, B_cov_cmb_eta)
BBls_HILC_dus_rhoonly = analysis.compute_HILC_solution(B_w, B_cov_dus_rho)
BBls_HILC_dus_etaonly = analysis.compute_HILC_solution(B_w, B_cov_dus_eta)
BBls_HILC_syn_rhoonly = analysis.compute_HILC_solution(B_w, B_cov_syn_rho)
BBls_HILC_syn_etaonly = analysis.compute_HILC_solution(B_w, B_cov_syn_eta)

BBls_HILC_cmb = BBls_HILC_cmb_rhoonly + BBls_HILC_cmb_etaonly
BBls_HILC_dus = BBls_HILC_dus_rhoonly + BBls_HILC_dus_etaonly
BBls_HILC_syn = BBls_HILC_syn_rhoonly + BBls_HILC_syn_rhoonly

lmax_again = 100
ell_again = np.arange(2,lmax_again+1)
C2D_again = ell_again*(ell_again+1)/(2*np.pi)

fig = plt.figure(figsize=(6,3))
ax1 = plt.subplot2grid((1, 3), (0, 0))
ax2 = plt.subplot2grid((1, 3), (0, 1), sharey=ax1)
ax3 = plt.subplot2grid((1, 3), (0, 2), sharey=ax1)

ax1.loglog(ell_again,(BBls_HILC_all-BBls_HILC_noiseonly)[2:lmax_again+1]*C2D_again,color=col_sol)
ax1.loglog(ell_again,BBls_HILC_cmb_rhoonly[2:lmax_again+1]*C2D_again, color=col_cmb, linestyle='--')
ax1.loglog(ell_again,BBls_HILC_cmb_etaonly[2:lmax_again+1]*C2D_again, color=col_cmb, linestyle=':')
ax1.set_title('CMB')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$D_\ell^{BB}$ [$\mu$K$^2$]')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=col_sol, lw=1.5),
                Line2D([0], [0], color=col_cmb, lw=1.5),
                Line2D([0], [0], color=col_dus, lw=1.5),
                Line2D([0], [0], color=col_syn, lw=1.5),
                Line2D([0], [0], color='silver', linestyle='--', lw=1.5),
                Line2D([0], [0], color='silver', linestyle=':',  lw=1.5)]
ax1.legend(custom_lines, ['HILC', 'CMB', 'dust', 'synch', r'$\rho$ only', r'$\eta$ only'], loc='lower right')

ax2.loglog(ell_again,(BBls_HILC_all-BBls_HILC_noiseonly)[2:lmax_again+1]*C2D_again, color=col_sol)
ax2.loglog(ell_again,BBls_HILC_dus_rhoonly[2:lmax_again+1]*C2D_again, color=col_dus, linestyle='--')
ax2.loglog(ell_again,BBls_HILC_dus_etaonly[2:lmax_again+1]*C2D_again, color=col_dus, linestyle=':')
ax2.set_title('dust')
ax2.set_xlabel(r'$\ell$')

ax3.loglog(ell_again,(BBls_HILC_all-BBls_HILC_noiseonly)[2:lmax_again+1]*C2D_again,color=col_sol)
ax3.loglog(ell_again,BBls_HILC_syn_rhoonly[2:lmax_again+1]*C2D_again, color=col_syn, linestyle='--')
ax3.loglog(ell_again,BBls_HILC_syn_etaonly[2:lmax_again+1]*C2D_again, color=col_syn, linestyle=':')
ax3.set_title('synchrotron')
ax3.set_xlabel(r'$\ell$')

plt.setp(ax2.get_yticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)
plt.tight_layout()
plt.savefig('output/HILC_components.pdf')
plt.clf()


'''
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# FIGURE 2 (HILC output spectra)

fig = plt.figure(figsize=(6,4))
ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))

# plotting output Cls
ax1.loglog(ell,Cls_commander78_dus[2,2:lmax+1]*C2D,color='teal',label='input dust', alpha=0.6)
ax1.loglog(ell,Cls_commander78_syn[2,2:lmax+1]*C2D,color='darkturquoise',label='input synchrotron', alpha=0.6)
ax1.loglog(ell,Cls_CMB[2,2:lmax+1]*C2D,color='black',linestyle='-',label='input CMB',alpha=0.6)
ax1.loglog(ell,BBls_HILC_noiseless[2:lmax+1]*C2D,color='red',  label='HILC output',alpha=0.6)
ax1.legend(loc='upper left')
ax1.set_xlabel(r'$\ell$')
ax1.set_ylabel(r'$D_\ell^{BB}$ [$\mu$K$^2$]')

lmax_again = 16
ell_again = np.arange(2,lmax_again+1)
C2D_again = ell_again*(ell_again+1)/(2*np.pi)

import matplotlib.ticker as mticker

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3,2))
ax2.yaxis.set_major_formatter(formatter)

# plotting output Cls
ax2.plot(ell_again,Cls_CMB[2,2:lmax_again+1]*C2D_again,color='black',linestyle='-',label='input CMB',alpha=0.6)
ax2.plot(ell_again,BBls_HILC_noiseless[2:lmax_again+1]*C2D_again,color='red',  label='HILC output',alpha=0.6)
ax2.plot(ell_again,BBls_HILC_rhoonly[2:lmax_again+1]*C2D_again,color='dimgrey',linestyle='--', label=r'$\rho$ component',alpha=0.6)
ax2.plot(ell_again,BBls_HILC_etaonly[2:lmax_again+1]*C2D_again,color='dimgrey',linestyle=':', label=r'$\eta$ component',alpha=0.6)
ax2.plot(ell_again,BBls_HILC_FGsonly[2:lmax_again+1]*C2D_again,color='cadetblue',label='FG residual', alpha=0.6)
ax2.legend(loc='upper left')
ax2.set_xlabel(r'$\ell$')

fig.tight_layout()
plt.savefig('output/HILC_output_'+rstr+'.pdf')
plt.clf()
'''

#lmax_beam = 2+np.argmin(np.abs(BBls_HILC_noiseonly[2:] - Cls_cmb[2,2:]))
#print('the noise residual intersects the theoretical CMB spectrum at ell='+str(lmax_beam))

#lmax_likelihood = lmax_beam
#ell_likelihood = np.arange(2,lmax_likelihood+1)
#C2D_new = ell_likelihood*(ell_likelihood+1)/(2*np.pi)      
        
r_fit, sigma_r, likelihood_r, A_fit, sigma_A, likelihood_A = analysis.estimate_r(BBls_HILC_all, ell_likelihood, 1, Cls_cmb_tensor[2], Cls_cmb_scalar[2] + BBls_HILC_noiseonly, r_sim, A_sim)           
print('r = ', r_fit)
print('sigma = ', sigma_r)

print('A = ', A_fit)
print('sigma = ', sigma_A)

#fig = plt.figure(figsize=(3,3))
#
# plotting output Cls
#plt.loglog(ell_likelihood,np.fabs(Cls_cmb[2,2:lmax_likelihood+1]*C2D_new),color=col_cmb,label='input CMB')
#plt.loglog(ell_likelihood,np.fabs(r_fit*Cls_cmb_tensor[2,2:lmax_likelihood+1]*C2D_new+A_fit*Cls_cmb_scalar[2,2:lmax_likelihood+1]*C2D_new),color=col_fit,label=r'$D_\ell^{BB}(\hat{r},\hat{A})$')
#plt.loglog(ell_likelihood,np.fabs((BBls_HILC_all-BBls_HILC_noiseonly)[2:lmax_likelihood+1]*C2D_new),color=col_sol,  label='HILC output', linestyle='--')
#handles, labels = plt.gca().get_legend_handles_labels()
#order = [2,1,0]
#plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'upper left')
#plt.xlim(right=100)
#plt.ylim(top=5e-3)
#plt.xlabel(r'$\ell$')
#plt.ylabel(r'$D_\ell^{BB}$ [$\mu$K$^2$]')
#plt.tight_layout()
#plt.savefig('output/HILC_r.pdf')
#plt.clf()

fig = plt.figure(figsize=(5,3))

#plt.axvline(x = r_fit_IDEAL, color = 'gray', alpha = 0.5)
#plt.axvline(x = (r_fit_IDEAL-sigma_r_IDEAL), color = 'gray', linestyle='--', alpha = 0.5)
#plt.axvline(x = (r_fit_IDEAL+sigma_r_IDEAL), color = 'gray', linestyle='--', alpha = 0.5)
#plt.semilogx(r_sim, likelihood_r_IDEAL/np.max(likelihood_r_IDEAL), color='teal', alpha = 0.5)
#plt.axvline(x = r_fit, color = 'gray')
#plt.axvline(x = (r_fit-sigma_r), color = 'gray', linestyle='--')
#plt.axvline(x = (r_fit+sigma_r), color = 'gray', linestyle='--')
plt.fill_between(x = r_sim,
                 y1 = likelihood_r/np.max(likelihood_r), 
                 where = (r_sim > r_fit - sigma_r) & (r_sim < r_fit + sigma_r),
                 color= "teal",
                 alpha= 0.2)
plt.plot(r_sim, likelihood_r/np.max(likelihood_r), color='teal')
plt. ylim(bottom=0)
plt.axvline(x = r_in, color = 'firebrick', alpha = 0.9)
plt.xlabel(r'$r$')
plt.xlim([0.002,0.006])
plt.ylabel(r'$L(r)$')
plt.tight_layout()
plt.savefig('output/likelihood_r.pdf')
plt.clf()

print('#################################')

#################

analysis = Analysis(r_in = r_in, NSIDE = params.NSIDE, params_path = params.params_path, telescopes = params.telescopes, chan_dicts = params.chan_dicts, 
                    plotting = True, ideal = False, gain_calibration = False, fews = True)

B_cov_all, B_cov_FGs_noise, N_cov_model = analysis.compute_covs()
B_w = analysis.compute_weights(B_cov_all)
        
#for i in np.arange(2,len(B_w)):
    #if not np.isclose(np.sum(B_w[i]),1):
        #print('for ell='+str(i)+' the sum of the weights is '+str(np.sum(B_w[i])))
          
BBls_HILC_all = analysis.compute_HILC_solution(B_w, B_cov_all)
BBls_HILC_FGs_noise = analysis.compute_HILC_solution(B_w, B_cov_FGs_noise)
BBls_HILC_noiseonly = analysis.compute_HILC_solution(B_w, N_cov_model)

lmax_again = 100
ell_again = np.arange(2,lmax_again+1)
C2D_again = ell_again*(ell_again+1)/(2*np.pi)
       
r_fit_NOGAIN, sigma_r_NOGAIN, likelihood_r_NOGAIN, A_fit_NOGAIN, sigma_A_NOGAIN, likelihood_A = analysis.estimate_r(BBls_HILC_all, ell_likelihood, 1, Cls_cmb_tensor[2], Cls_cmb_scalar[2] + BBls_HILC_noiseonly, r_sim, A_sim)    


fig = plt.figure(figsize=(5,3))
plt.plot(r_sim, likelihood_r_NOGAIN/np.max(likelihood_r_NOGAIN), color='lightseagreen', linestyle=':', label = 'w/o calibration')
plt.fill_between(x = r_sim,
                 y1 = likelihood_r/np.max(likelihood_r), 
                 where = (r_sim > r_fit - sigma_r) & (r_sim < r_fit + sigma_r),
                 color= "teal",
                 alpha= 0.2)
plt.plot(r_sim, likelihood_r/np.max(likelihood_r), color='teal', label = 'w/ calibration')
plt.ylim(bottom=0)
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'upper left')
plt.axvline(x = r_in, color = 'firebrick', alpha = 0.9)
plt.xlabel(r'$r$')
plt.xlim([0.002,0.006])
plt.ylabel(r'$L(r)$')
plt.tight_layout()
plt.savefig('output/likelihood_r_TOGETHER.pdf')
plt.clf()

fig = plt.figure(figsize=(5,3))
plt.fill_between(x = r_sim,
                 y1 = likelihood_r_NOGAIN/np.max(likelihood_r_NOGAIN), 
                 where = (r_sim > r_fit_NOGAIN - sigma_r_NOGAIN) & (r_sim < r_fit_NOGAIN + sigma_r_NOGAIN),
                 color= "teal",
                 alpha= 0.2)
plt.plot(r_sim, likelihood_r_NOGAIN/np.max(likelihood_r_NOGAIN), color='teal')
plt.ylim(bottom=0)
plt.axvline(x = r_in, color = 'firebrick', alpha = 0.9)
plt.xlabel(r'$r$')
plt.xlim([0.002,0.006])
plt.ylabel(r'$L(r)$')
plt.tight_layout()
plt.savefig('output/likelihood_r_NOGAIN.pdf')
plt.clf()
       
print('r = ', r_fit_NOGAIN)
print('sigma = ', sigma_r_NOGAIN)

print('A = ', A_fit_NOGAIN)
print('sigma = ', sigma_A_NOGAIN)

end = time.time()
print('it took ', end-start)
