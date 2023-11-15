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
start = time.time()

# plotting setup
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

# initializing r_in and the likelihood domains
r_in = 0.00461
r_sim = np.linspace(0.001, 0.008, num = 1000)
A_sim = np.linspace(0.7, 1, 1000)

print('###############################################')
print('running for ideal HWP with gain calibration')

# initializing Analysis class for ideal HWP with gain calibration
analysis = Analysis(r_in = r_in, NSIDE = params.NSIDE, params_path = params.params_path, telescopes = params.telescopes, chan_dicts = params.chan_dicts, plotting = True, ideal = True, gain_calibration = True, fews = True)

# defining multipole array and C2D conversion array
ell = np.arange(2,analysis.lmax+1)
C2D = ell*(ell+1)/(2*np.pi) 

# computing the covariance matrices and the weights
B_cov_all, B_cov_FGs_noise, N_cov_model = analysis.compute_covs()
B_w = analysis.compute_weights(B_cov_all)
        
# uncomment to test that the weights sum to 1
#for i in np.arange(2,len(B_w)):
#    if not np.isclose(np.sum(B_w[i]),1):
#        print('for ell='+str(i)+' the sum of the weights is '+str(np.sum(B_w[i])))
        
# plotting HILC weights (figure 2)
cm = matplotlib.colormaps['viridis']

fig = plt.figure(figsize=(2.8,2.6))
ax1 = plt.subplot2grid((1, 1), (0, 0), projection='3d')
colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1), cm(.3), cm(.5), cm(.7), cm(.9), cm(.7), cm(.5), cm(.3)]
yticks = [40, 50, 60, 68, 68, 78, 78, 89, 89, 100, 119, 140]
for c, k in zip(colors, np.arange(len(yticks))):
    xs = ell
    ys = B_w[2:analysis.lmax+1,k]
    zs = yticks[k]
    cs = [c] * len(xs)
    # uncomment to print whether there are negative weights (testing purposes)
    #if np.any(ys < 0):
    #    print('LFT: '+str(zs)+' GHz has negative values')
    ax1.bar(xs, ys, zs, zdir='x', color=cs, alpha=0.8)
ax1.set_xticks(yticks, labels=['40', '', '', '', '', '', '', '', '', '', '', '140'])
ax1.tick_params(axis='x', which='major', pad=-3)
ax1.set_xlabel('Frequency [GHz]', labelpad=-3)
ax1.set_zlim([-0.10,0.15])
ax1.tick_params(axis='y', which='major', pad=-3)
ax1.set_ylabel(r'$\ell$', labelpad=-3)
ax1.set_title(r'LFT weights', fontsize = 11, pad=-14)
plt.savefig('output/weights_IDEAL_LFT.pdf')
plt.clf()

fig = plt.figure(figsize=(2.8,2.6))
ax2 = plt.subplot2grid((1, 1), (0, 0), projection='3d')
colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1)]
yticks = [100, 119, 140, 166, 195]
for c, k in zip(colors, np.arange(len(yticks))):
    xs = ell
    ys = B_w[2:analysis.lmax+1, 12+k]
    zs = yticks[k]
    cs = [c] * len(xs)
    # uncomment to print whether there are negative weights (testing purposes)
    #if np.any(ys < 0):
    #    print('MFT: '+str(zs)+' GHz has negative values')
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

fig = plt.figure(figsize=(2.8,2.6))
ax3 = plt.subplot2grid((1, 1), (0, 0), projection='3d')
colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1)]
yticks = [195, 235, 280, 337, 402]
for c, k in zip(colors, np.arange(len(yticks))):
    xs = ell
    ys = B_w[2:analysis.lmax+1, 17+k]
    zs = yticks[k]
    cs = [c] * len(xs)
    # uncomment to print whether there are negative weights (testing purposes)
    #if np.any(ys < 0):
    #    print('HFT: '+str(zs)+' GHz has negative values')
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

# compute HILC solution for all components, FG+noise and noise-only
BBls_HILC_all = analysis.compute_HILC_solution(B_w, B_cov_all)
BBls_HILC_FGs_noise = analysis.compute_HILC_solution(B_w, B_cov_FGs_noise)
BBls_HILC_noiseonly = analysis.compute_HILC_solution(B_w, N_cov_model) 

Cls_cmb = analysis.Cls_cmb
Cls_cmb_tensor = analysis.Cls_cmb_tensor
Cls_cmb_scalar = analysis.Cls_cmb_scalar
Cls_dus = analysis.Cls_dus
Cls_syn = analysis.Cls_syn

# plotting HILC solution (figure 1)
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

# uncomment to estimate beam resolution
#lmax_beam = 2+np.argmin(np.abs(BBls_HILC_noiseonly[2:] - Cls_cmb[2,2:]))
#print('the noise residual intersects the theoretical CMB spectrum at ell='+str(lmax_beam))

# set up for likelihood inference
lmax_likelihood = 200
ell_likelihood = np.arange(2,lmax_likelihood+1)
C2D_new = ell_likelihood*(ell_likelihood+1)/(2*np.pi)      
        
# perform likelihood inference
r_fit, sigma_r, likelihood_r, A_fit, sigma_A, likelihood_A = analysis.estimate_r(BBls_HILC_all, ell_likelihood, 1, Cls_cmb_tensor[2], Cls_cmb_scalar[2] + BBls_HILC_noiseonly, r_sim, A_sim)           
print('r = ', r_fit)
print('sigma = ', sigma_r)
print('A = ', A_fit)
print('sigma = ', sigma_A)

print('###############################################')
print('running for non-ideal HWP with gain calibration')

# initializing Analysis class for non-ideal HWP with gain calibration
analysis = Analysis(r_in = r_in, NSIDE = params.NSIDE, params_path = params.params_path, telescopes = params.telescopes, chan_dicts = params.chan_dicts, plotting = True, ideal = False, gain_calibration = True, fews = False)

# computing the covariance matrices and the weights
B_cov_all, B_cov_FGs_noise, N_cov_model, B_cov_cmb_rho, B_cov_cmb_eta, B_cov_dus_rho, B_cov_dus_eta, B_cov_syn_rho, B_cov_syn_eta = analysis.compute_covs()
B_w = analysis.compute_weights(B_cov_all)
 
# uncomment to test that the weights sum to 1        
#for i in np.arange(2,len(B_w)):
#    if not np.isclose(np.sum(B_w[i]),1):
#        print('for ell='+str(i)+' the sum of the weights is '+str(np.sum(B_w[i])))
        
# plotting HILC weights (figure 5)
plt.close('all')  

fig = plt.figure(figsize=(2.8,2.6))
ax1 = plt.subplot2grid((1, 1), (0, 0), projection='3d')
colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1), cm(.3), cm(.5), cm(.7), cm(.9), cm(.7), cm(.5), cm(.3)]
yticks = [40, 50, 60, 68, 68, 78, 78, 89, 89, 100, 119, 140]
for c, k in zip(colors, np.arange(len(yticks))):
    xs = ell
    ys = B_w[2:analysis.lmax+1,k]
    zs = yticks[k]
    cs = [c] * len(xs)
    # uncomment to print whether there are negative weights (testing purposes)
    #if np.any(ys < 0):
    #    print('LFT: '+str(zs)+' GHz has negative values')
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

fig = plt.figure(figsize=(2.8,2.6))
ax2 = plt.subplot2grid((1, 1), (0, 0), projection='3d')
colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1)]
yticks = [100, 119, 140, 166, 195]
for c, k in zip(colors, np.arange(len(yticks))):
    xs = ell
    ys = B_w[2:analysis.lmax+1, 12+k]
    zs = yticks[k]
    cs = [c] * len(xs)
    # uncomment to print whether there are negative weights (testing purposes)
    #if np.any(ys < 0):
    #    print('MFT: '+str(zs)+' GHz has negative values')
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

fig = plt.figure(figsize=(2.8,2.6))
ax3 = plt.subplot2grid((1, 1), (0, 0), projection='3d')
colors = [cm(.9), cm(.7), cm(.5), cm(.3), cm(.1)]
yticks = [195, 235, 280, 337, 402]
for c, k in zip(colors, np.arange(len(yticks))):
    xs = ell
    ys = B_w[2:analysis.lmax+1, 17+k]
    zs = yticks[k]
    cs = [c] * len(xs)
    # uncomment to print whether there are negative weights (testing purposes)
    #if np.any(ys < 0):
    #    print('HFT: '+str(zs)+' GHz has negative values')
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
   
# compute HILC solution for all components, FG+noise and noise-only
BBls_HILC_all = analysis.compute_HILC_solution(B_w, B_cov_all)
BBls_HILC_FGs_noise = analysis.compute_HILC_solution(B_w, B_cov_FGs_noise)
BBls_HILC_noiseonly = analysis.compute_HILC_solution(B_w, N_cov_model)

# plotting HILC solution (figure 1)
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

# uncomment to print the multipole range where the HILC solution is close to bar((rho_cmb/g_cmb)**2)*Cls_cmb FIXME: bar((rho_cmb/g_cmb)**2) should be the output of some function!
#overlapping_ells = ell[np.argwhere(np.isclose(0.943033768824104*Cls_cmb[2,2:], (BBls_HILC_all-BBls_HILC_noiseonly)[2:]))]
#print(np.min(overlapping_ells))
#print(np.max(overlapping_ells))

# compute HILC solution for each component, separately
BBls_HILC_cmb_rhoonly = analysis.compute_HILC_solution(B_w, B_cov_cmb_rho)
BBls_HILC_cmb_etaonly = analysis.compute_HILC_solution(B_w, B_cov_cmb_eta)
BBls_HILC_dus_rhoonly = analysis.compute_HILC_solution(B_w, B_cov_dus_rho)
BBls_HILC_dus_etaonly = analysis.compute_HILC_solution(B_w, B_cov_dus_eta)
BBls_HILC_syn_rhoonly = analysis.compute_HILC_solution(B_w, B_cov_syn_rho)
BBls_HILC_syn_etaonly = analysis.compute_HILC_solution(B_w, B_cov_syn_eta)
#
BBls_HILC_cmb = BBls_HILC_cmb_rhoonly + BBls_HILC_cmb_etaonly
BBls_HILC_dus = BBls_HILC_dus_rhoonly + BBls_HILC_dus_etaonly
BBls_HILC_syn = BBls_HILC_syn_rhoonly + BBls_HILC_syn_rhoonly

# set up for likelihood inference
lmax_again = 100
ell_again = np.arange(2,lmax_again+1)
C2D_again = ell_again*(ell_again+1)/(2*np.pi)

# plotting all components of the HILC solution (figure 6)
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

# perform likelihood inference       
r_fit, sigma_r, likelihood_r, A_fit, sigma_A, likelihood_A = analysis.estimate_r(BBls_HILC_all, ell_likelihood, 1, Cls_cmb_tensor[2], Cls_cmb_scalar[2] + BBls_HILC_noiseonly, r_sim, A_sim)           
print('r = ', r_fit)
print('sigma = ', sigma_r)
print('A = ', A_fit)
print('sigma = ', sigma_A)

print('###############################################')
print('running for non-ideal HWP without gain calibration')

# initializing Analysis class for non-ideal HWP without gain calibration
analysis = Analysis(r_in = r_in, NSIDE = params.NSIDE, params_path = params.params_path, telescopes = params.telescopes, chan_dicts = params.chan_dicts, plotting = True, ideal = False, gain_calibration = False, fews = True)

# computing the covariance matrices and the weights
B_cov_all, B_cov_FGs_noise, N_cov_model = analysis.compute_covs()
B_w = analysis.compute_weights(B_cov_all)

# uncomment to test that the weights sum to 1        
#for i in np.arange(2,len(B_w)):
#    if not np.isclose(np.sum(B_w[i]),1):
#        print('for ell='+str(i)+' the sum of the weights is '+str(np.sum(B_w[i])))

# compute HILC solution for all components, FG+noise and noise-only          
BBls_HILC_all = analysis.compute_HILC_solution(B_w, B_cov_all)
BBls_HILC_FGs_noise = analysis.compute_HILC_solution(B_w, B_cov_FGs_noise)
BBls_HILC_noiseonly = analysis.compute_HILC_solution(B_w, N_cov_model)

# set up for likelihood inference
lmax_again = 100
ell_again = np.arange(2,lmax_again+1)
C2D_again = ell_again*(ell_again+1)/(2*np.pi)
    
# perform likelihood inference              
r_fit_NOGAIN, sigma_r_NOGAIN, likelihood_r_NOGAIN, A_fit, sigma_A, likelihood_A = analysis.estimate_r(BBls_HILC_all, ell_likelihood, 1, Cls_cmb_tensor[2], Cls_cmb_scalar[2] + BBls_HILC_noiseonly, r_sim, A_sim)    

print('r = ', r_fit_NOGAIN)
print('sigma = ', sigma_r_NOGAIN)
print('A = ', A_fit)
print('sigma = ', sigma_A)

# plotting likelihoods (figure 7)
fig = plt.figure(figsize=(5,3))
plt.plot(r_sim, likelihood_r_NOGAIN/np.max(likelihood_r_NOGAIN), color='lightseagreen', linestyle=':', label = 'w/o calibration')
plt.fill_between(x = r_sim,
                 y1 = likelihood_r/np.max(likelihood_r), 
                 where = (r_sim > r_fit - sigma_r) & (r_sim < r_fit + sigma_r),
                 color= "teal",
                 alpha= 0.2)
plt.plot(r_sim, likelihood_r/np.max(likelihood_r), color='teal', label = 'w/ calibration')
plt. ylim(bottom=0)
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
       
end = time.time()
print('it took ', end-start)
