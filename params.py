import numpy as np

r_sim = np.linspace(0., 0.036, num = 9000)
A_sim = np.linspace(0.7, 1.3, num = 2000)

#SEDs params (from 1807.06208)
Tdus = 19.6		#K
beta_dus = 1.55
nu_star_dus = 353e9
beta_syn = -3.1
nu_star_syn = 30e9

###

NSIDE = 512

# parameter file from https://github.com/cmbant/CAMB/blob/master/inifiles/planck_2018.ini
# (only change: I've set get_tensor_cls = T)
params_path = 'input/planck_2018.ini' 

q_dus_E = 323.
a_dus_E = -.40
q_dus_B = 199.
a_dus_B = -.50

q_syn_E = 2.3
a_syn_E = -.84
q_syn_B = 0.8
a_syn_B = -.76

###

telescopes = ['LFT','MFT','HFT']

# list of all channels
chan_dicts = np.array([{'telescope':'LFT', 'nu':40. , 'delta':12. , 'fwhm':70.5 , 'sensitivity':37.42, 'sigma_alpha':49.8},
                       {'telescope':'LFT', 'nu':50. , 'delta':15. , 'fwhm':58.5 , 'sensitivity':33.46, 'sigma_alpha':39.8},
                       {'telescope':'LFT', 'nu':60. , 'delta':14. , 'fwhm':51.1 , 'sensitivity':21.31, 'sigma_alpha':16.1},
                       {'telescope':'LFT', 'nu':68. , 'delta':16. , 'fwhm':41.6 , 'sensitivity':19.91, 'sigma_alpha':1.09},
                       {'telescope':'LFT', 'nu':68. , 'delta':16. , 'fwhm':47.1 , 'sensitivity':31.77, 'sigma_alpha':35.9},
                       {'telescope':'LFT', 'nu':78. , 'delta':18. , 'fwhm':36.9 , 'sensitivity':15.55, 'sigma_alpha':8.6 },
                       {'telescope':'LFT', 'nu':78. , 'delta':18. , 'fwhm':43.8 , 'sensitivity':19.13, 'sigma_alpha':13.0},
                       {'telescope':'LFT', 'nu':89. , 'delta':20. , 'fwhm':33.0 , 'sensitivity':12.28, 'sigma_alpha':5.4 },
                       {'telescope':'LFT', 'nu':89. , 'delta':20. , 'fwhm':41.5 , 'sensitivity':28.77, 'sigma_alpha':29.4},
                       {'telescope':'LFT', 'nu':100., 'delta':23. , 'fwhm':30.2 , 'sensitivity':10.34, 'sigma_alpha':3.8 },
                       {'telescope':'MFT', 'nu':100., 'delta':23. , 'fwhm':37.8 , 'sensitivity':8.48 , 'sigma_alpha':2.6 },
                       {'telescope':'LFT', 'nu':119., 'delta':36. , 'fwhm':26.3 , 'sensitivity':7.69 , 'sigma_alpha':2.1 },
                       {'telescope':'MFT', 'nu':119., 'delta':36. , 'fwhm':33.6 , 'sensitivity':5.70 , 'sigma_alpha':1.2 },
                       {'telescope':'LFT', 'nu':140., 'delta':42. , 'fwhm':23.7 , 'sensitivity':7.25 , 'sigma_alpha':1.8 },
                       {'telescope':'MFT', 'nu':140., 'delta':42. , 'fwhm':30.8 , 'sensitivity':6.38 , 'sigma_alpha':1.5 },
                       {'telescope':'MFT', 'nu':166., 'delta':50. , 'fwhm':28.9 , 'sensitivity':5.57 , 'sigma_alpha':1.1 },
                       {'telescope':'MFT', 'nu':195., 'delta':59. , 'fwhm':28.0 , 'sensitivity':7.05 , 'sigma_alpha':1.8 },
                       {'telescope':'HFT', 'nu':195., 'delta':59. , 'fwhm':28.6 , 'sensitivity':10.50, 'sigma_alpha':3.9 },
                       {'telescope':'HFT', 'nu':235., 'delta':71. , 'fwhm':24.7 , 'sensitivity':10.79, 'sigma_alpha':4.1 },
                       {'telescope':'HFT', 'nu':280., 'delta':84. , 'fwhm':22.5 , 'sensitivity':13.80, 'sigma_alpha':6.8 },
                       {'telescope':'HFT', 'nu':337., 'delta':101., 'fwhm':20.9 , 'sensitivity':21.95, 'sigma_alpha':17.1},
                       {'telescope':'HFT', 'nu':402., 'delta':92. , 'fwhm':17.9 , 'sensitivity':47.45, 'sigma_alpha':80.0},
                       ]) 
