import numpy as np
import healpy as hp

import camb

import matplotlib
from matplotlib import pyplot as plt

class Sky_SEDs(object):
    '''
    A class to compute CMB, dust and synchrotron SEDs.
    '''                    
    def __init__(self, Tdus = 19.6, beta_dus = 1.55, nu_star_dus = 353e9, beta_syn = -3.1, nu_star_syn = 30e9):
        '''
        Initialize Sky_SEDs class.

        Keyword arguments
        -----------------
        Tdus : float
            Dust temperature, in K (default : 19.6)
        beta_dus : float
            Dust spectral index, dimensionless (default : 1.55)
        nu_star_dus : float
            Dust reference frequency, in Hz (default: 353e9)
        beta_syn : float
            Synchrotron spectral index, dimensionless (default : -3.1)
        nu_star_syn : float
            Synchrotron reference frequency, in Hz (default: 30e9)
            
        Notes
        -----------------
        All the default values are taken from 1807.06208
        '''
        self.Tdus = Tdus
        self.beta_dus = beta_dus
        self.nu_star_dus = nu_star_dus
        self.beta_syn = beta_syn
        self.nu_star_syn = nu_star_syn
        
        self.h = 6.62607015e-34	#J/Hz
        self.c = 299792458		#m/s 
        self.kB = 1.380649e-23	#J/K
        self.Tcmb = 2.725	 	#K 

    def _x_func(self,nu,T):
        return self.h*nu/(self.kB*T) 				#dimensionless, as it should be

    def _planck(self,nu,T):
        x = self._x_func(nu,T)
        factor = 2*self.h*nu**3/self.c**2
        spectrum = factor/(np.exp(x)-1)			#dimensions: W/(m^2*Hz)/K = 10^26 Jy/K = 10^20 MJy/K
        spectrum *= 1e20					#dimensions: MJy/K
        return spectrum
    
    def _CMB_resp(self,nu):
        x = self._x_func(nu,self.Tcmb)
        factor = 2*nu**2*self.kB/self.c**2
        spectrum = factor*x**2*np.exp(x)/(np.exp(x)-1)**2	#dimensions: W/(m^2*Hz)/K = 10^26 Jy/K = 10^20 MJy/K
        spectrum *= 1e20					#dimensions: MJy/K
        return spectrum
        
    def CMB(self,nu):
        '''
        Returns CMB SED.
  
        Arguments
        -----------------
        nu : float
            Frequency, or array of frequencies, in Hz

        Notes
        -----------------
                
        '''
        resp = self._CMB_resp(nu)/self._CMB_resp(nu)
        return resp

    def dus(self,nu):
        '''
        Returns dust SED.
  
        Arguments
        -----------------
        nu : float
            Frequency, or array of frequencies, in Hz

        Notes
        -----------------
               
        '''
        factor_1 = (nu/self.nu_star_dus)**self.beta_dus
        factor_2 = self._planck(nu,self.Tdus)/self._planck(self.nu_star_dus,self.Tdus)  
        factor_3 = self._CMB_resp(self.nu_star_dus)/self._CMB_resp(nu)
        return factor_1*factor_2*factor_3
        
    def syn(self,nu):
        '''
        Returns synchrotron SED.
  
        Arguments
        -----------------
        nu : float
            Frequency, or array of frequencies, in Hz

        Notes
        -----------------
                
        '''
        factor_1 = (nu/self.nu_star_syn)**self.beta_syn
        factor_2 = self._CMB_resp(self.nu_star_syn)/self._CMB_resp(nu)
        return factor_1*factor_2
      
class Sky_Cls(object):
    '''
    A class to define 
    '''
    def __init__(self, lmax, params_path, params_dus_E = [323., -.40], params_dus_B = [199., -.50], params_syn_E = [2.3, -.84], params_syn_B = [0.8, -.76], theory_to_realization = False):
        '''
        Initialize Sky_Cls class.

        Arguments
        -----------------
        lmax : int
            Highest multipole considered
        params_path : str
            String specifying the path to the parameter file for running CAMB
        
        Keyword arguments
        -----------------
        params_dus_E : list of floats
            Dust E-modes q and a power law parameters (default [323., -.40])
        params_dus_B : list of floats
            Dust B-modes q and a power law parameters (default [199., -.50])
        params_syn_E : list of floats
            Synchrotron E-modes q and a power law parameters (default [2.3, -.84])
        params_syn_B : list of floats
            Synchrotron B-modes q and a power law parameters (default [0.8, -.76])
        theory_to_realization: bool
            If True, the returned CMB Cls are the ones of a particular realization
            If False, the returned CMB Cls are the theoretical ones
            
        Notes
        -----------------
        All the default values are taken from 1807.06208
        '''
        self.params_path = params_path
        self.lmax = lmax
        self.q_dus_E, self.a_dus_E = params_dus_E
        self.q_dus_B, self.a_dus_B = params_dus_B
        self.q_syn_E, self.a_syn_E = params_syn_E 
        self.q_syn_B, self.a_syn_B = params_syn_B
        self.theory_to_realization = theory_to_realization
    
        self.ell = np.arange(2,self.lmax+1)
        self.C2D = self.ell*(self.ell+1)/(2*np.pi)
        self.D2C = np.append([1,1],1/self.C2D)
        
        self.pars = camb.read_ini(self.params_path)
        self.results = camb.get_results(self.pars)
        self.powers = self.results.get_cmb_power_spectra(self.pars, CMB_unit='muK')

    def _Dls_power(self,q,alpha,lmax):
        return np.append(0,q*(np.arange(1,lmax+1)/80)**alpha)
        
    def CMB_scalar(self):
        '''
        Returns the angular power spectra induced by scalar perturbations from 0 to lmax.

        Notes
        -----------------
                
        '''
        CAMB = self.powers['lensed_scalar']
        Dls = np.array([CAMB[:,0], CAMB[:,1], CAMB[:,2], CAMB[:,3]])
        Cls = Dls[:,:self.lmax+1]*self.D2C
        if self.theory_to_realization:
            Cls = hp.alm2cl(hp.synalm(Cls, lmax=self.lmax, new=True), lmax=self.lmax)
        return Cls
        
    def CMB_tensor(self):
        '''
        Returns the angular power spectra induced by tensor perturbations from 0 to lmax.

        Notes
        -----------------
                
        '''
        CAMB = self.powers['tensor']
        Dls = np.array([CAMB[:,0], CAMB[:,1], CAMB[:,2], CAMB[:,3]])
        Cls = Dls[:,:self.lmax+1]*self.D2C
        if self.theory_to_realization:
            Cls = hp.alm2cl(hp.synalm(Cls, lmax=self.lmax, new=True), lmax=self.lmax)
        return Cls
        
    def CMB(self,r):
        '''
        Returns the total angular power spectra induced by scalar and tensor perturbations from 0 to lmax.
  
        Arguments
        -----------------
        r : float
            Value of the tensor-to-scalar ratio

        Notes
        -----------------
                
        '''
        Cls = self.CMB_scalar() + r*self.CMB_tensor()
        if self.theory_to_realization:
            Cls = hp.alm2cl(hp.synalm(Cls, lmax=self.lmax, new=True), lmax=self.lmax)
        return Cls

    def dus(self):
        '''
        Returns the dust angular power spectra from 0 to lmax.

        Notes
        -----------------
        Only the EE and BB spectra are non-zero.        
        '''
        Cls = np.empty((6,self.lmax+1))
        Cls[1] = self._Dls_power(self.q_dus_E,self.a_dus_E,self.lmax)*self.D2C
        Cls[2] = self._Dls_power(self.q_dus_B,self.a_dus_B,self.lmax)*self.D2C
        return Cls

    def syn(self):
        '''
        Returns the synchrotron angular power spectra from 0 to lmax.

        Notes
        -----------------
        Only the EE and BB spectra are non-zero.        
        '''
        Cls = np.empty((6,self.lmax+1))       
        Cls[1] = self._Dls_power(self.q_syn_E,self.a_syn_E,self.lmax)*self.D2C
        Cls[2] = self._Dls_power(self.q_syn_B,self.a_syn_B,self.lmax)*self.D2C       
        return Cls
        
class Instrument(object):
    '''
    A class to define and work with the instrument specifics.
    '''
    def __init__(self, telescopes, lmax):
        '''
        Initialize Instrument class.

        Arguments
        -----------------
        telescopes : str
            It can be 'LFT','MFT', or 'HFT'
        lmax : int
            Highest multipole considered
        '''
        self.telescopes = telescopes
        self.lmax = lmax
    
    def muellers_telescopes(self):
        '''
        Returns the Mueller matrix elements for the given telescope.
        
        Notes
        -----------------
              
        '''
        ms_all = []
        fs_all = []
        
        for telescope in self.telescopes:
            f = np.load('input/' + telescope + '_mueller.npz')
            fs_all.append(f['freq']*1e9)                             # frequency [Hz]
            ms_all.append(np.transpose(f['m'].real, axes=[2,0,1]))   # mueller matrix
            
        return fs_all, ms_all
         
    def muellers_channel(self,chan_dict):
        '''
        Returns the Mueller matrix elements of a specific frequency channel.
  
        Arguments
        -----------------
        chan_dict : dict
            Must contain 'telescope', 'nu', and 'delta'.
            For example {'telescope':'LFT', 'nu':40. , 'delta':12. , 'fwhm':70.5 , 'sensitivity':37.42, 'sigma_alpha':49.8}

        Notes
        -----------------
              
        '''
        telescope = chan_dict['telescope']
        bandcenter = chan_dict['nu']*1e9         #Hz
        bandwidth = chan_dict['delta']*1e9       #Hz
        
        f = np.load('input/' + telescope + '_mueller.npz')
        fs = f['freq']*1e9                             # frequency [Hz]
        ms = np.transpose(f['m'].real, axes=[2,0,1])   # mueller matrix
    
        bandidx = np.where((fs >= bandcenter - bandwidth/2) & (fs <= bandcenter + bandwidth/2))
            
        return fs[bandidx], ms[bandidx,:,:][0]
    
    def Nls(self,chan_dict):
        '''
        Returns noise angular power spectra from 0 to lmax.
  
        Arguments
        -----------------
        chan_dict : dict
            Must contain 'sensitivity'.
            For example {'telescope':'LFT', 'nu':40. , 'delta':12. , 'fwhm':70.5 , 'sensitivity':37.42, 'sigma_alpha':49.8}

        Notes
        -----------------
                     
        '''
        sensitivity = chan_dict['sensitivity']
        return np.ones(self.lmax+1)*(np.pi*sensitivity/10800)**2   
        
    def Bls(self,chan_dict):
        '''
        Returns beam coefficients from 0 to lmax.
  
        Arguments
        -----------------
        chan_dict : dict
            Must contain 'fwhm'.
            For example {'telescope':'LFT', 'nu':40. , 'delta':12. , 'fwhm':70.5 , 'sensitivity':37.42, 'sigma_alpha':49.8}

        Notes
        -----------------
              
        '''
        fwhm = chan_dict['fwhm']/60*(np.pi/180)  #radians
        return hp.gauss_beam(fwhm, lmax=self.lmax, pol=True)[:,2]
        
class Analysis(object):
    '''
    A class to define all the functions used in the analysis.
    '''   
    def __init__(self, r_in, NSIDE, params_path, telescopes, chan_dicts, Tdus = 19.6, beta_dus = 1.55, nu_star_dus = 353e9, beta_syn = -3.1, nu_star_syn = 30e9,
                       params_dus_E = [323., -.40], params_dus_B = [199., -.50], params_syn_E = [2.3, -.84], params_syn_B = [0.8, -.76], theory_to_realization = False,
                       plotting = False, ideal = True, gain_calibration = True, fews = True, pm = False):
        '''
        Initialize Analysis class.

        Arguments
        -----------------
        r_in : float
            input value of the tensor-to-scalar ratio
        NSIDE : int
            Healpy pixel resolution
        params_path : str
            String specifying the path to the parameter file for running CAMB
        telescopes : list of strings
            For example ['LFT','MFT','HFT']
        chan_dicts : list of dictionaries
            For an example, see params.py file
        
        Keyword arguments
        -----------------
        Tdus : float
            Dust temperature, in K (default : 19.6)
        beta_dus : float
            Dust spectral index, dimensionless (default : 1.55)
        nu_star_dus : float
            Dust reference frequency, in Hz (default: 353e9)
        beta_syn : float
            Synchrotron spectral index, dimensionless (default : -3.1)
        nu_star_syn : float
            Synchrotron reference frequency, in Hz (default: 30e9)
        params_dus_E : list of floats
            Dust E-modes q and a power law parameters (default [323., -.40])
        params_dus_B : list of floats
            Dust B-modes q and a power law parameters (default [199., -.50])
        params_syn_E : list of floats
            Synchrotron E-modes q and a power law parameters (default [2.3, -.84])
        params_syn_B : list of floats
            Synchrotron B-modes q and a power law parameters (default [0.8, -.76])
        theory_to_realization: bool
            If True, the returned CMB Cls are the ones of a particular realization
            If False, the returned CMB Cls are the theoretical ones
        plotting : bool
            If True, plots the Mueller matrix elements for LFT, MFT and HFT, and saves the figure
        ideal : bool
            If True, assumes the HWP to be ideal
        gain_calibration : bool
            If True, performs (ideal) gain calibration
        fews : bool
            If True, returns only the B_cov_all, B_cov_FGs_noise, N_cov_model covariances
            If False, returns B_cov_all, B_cov_FGs_noise, N_cov_model, B_cov_cmb_rho, B_cov_cmb_eta, B_cov_dus_rho, B_cov_dus_eta, B_cov_syn_rho, B_cov_syn_eta
        pm : bool
            If True, returns the bounds of the asymmetric 68% CL interval for the MLE tensor-to-scalar ratio
            
        Notes
        -----------------
        All the default values are taken from 1807.06208
        '''
        self.Tdus = Tdus
        self.beta_dus = beta_dus
        self.nu_star_dus = nu_star_dus
        self.beta_syn = beta_syn
        self.nu_star_syn = nu_star_syn
        
        self.r_in = r_in
        self.NSIDE = NSIDE
        self.params_path = params_path
        self.q_dus_E, self.a_dus_E = params_dus_E
        self.q_dus_B, self.a_dus_B = params_dus_B
        self.q_syn_E, self.a_syn_E = params_syn_E 
        self.q_syn_B, self.a_syn_B = params_syn_B
        
        self.theory_to_realization = theory_to_realization
        self.plotting = plotting
        self.ideal = ideal
        self.gain_calibration = gain_calibration
        self.fews = fews
        self.pm = pm
    
        self.lmax = 2*self.NSIDE + 1
        self.ell = np.arange(2,self.lmax+1)
        self.C2D = self.ell*(self.ell+1)/(2*np.pi)
        self.D2C = np.append([1,1],1/self.C2D)
        
        self.pars = camb.read_ini(self.params_path)
        self.results = camb.get_results(self.pars)
        self.powers = self.results.get_cmb_power_spectra(self.pars, CMB_unit='muK')
        
        self.telescopes = telescopes
        self.chan_dicts = chan_dicts
        self.nchan = len(self.chan_dicts)
          
        self.input_SEDs = Sky_SEDs(self.Tdus, self.beta_dus, self.nu_star_dus, self.beta_syn, self.nu_star_syn)
        self.input_Cls = Sky_Cls(self.lmax, self.params_path, [self.q_dus_E, self.a_dus_E], [self.q_dus_B, self.a_dus_B], [self.q_syn_E, self.a_syn_E], [self.q_syn_B, self.a_syn_B])
        self.instrument = Instrument(self.telescopes, self.lmax)
        
        # computing input angular power spectra
        self.Cls_cmb_scalar = self.input_Cls.CMB_scalar()
        self.Cls_cmb_tensor = self.input_Cls.CMB_tensor()
        self.Cls_cmb = self.input_Cls.CMB(self.r_in)
        self.Cls_dus = self.input_Cls.dus()
        self.Cls_syn = self.input_Cls.syn() 
    
        # filling frequencies and Mueller matrices arrays
        freqs_all = np.array([]) 
        
        freqs_temp, muellers_temp = self.instrument.muellers_telescopes()    
        freqs_all = np.concatenate((freqs_all,freqs_temp[0]),0) 
        freqs_all = np.concatenate((freqs_all,freqs_temp[1]),0) 
        freqs_all = np.concatenate((freqs_all,freqs_temp[2]),0)
        self.freqs = np.unique(freqs_all)
            
        f_LFT, f_MFT, f_HFT = freqs_temp                #stores f_LFT, f_MFT, f_HFT
        M_LFT, M_MFT, M_HFT = muellers_temp             #stores M_LFT, M_MFT, M_HFT
        
        if self.plotting:
        
            plt.rcParams.update({
            "font.size":10,
            "text.usetex": True
            })  
            
            from matplotlib.ticker import FormatStrFormatter
            
            ###
            
            default = np.diag([1,1,-1,-1])
                            
            fig, axs = plt.subplots(3, 3)
            cmap = matplotlib.cm.get_cmap('magma')
            col_LFT = cmap(0.44)
            col_MFT = cmap(0.65)
            col_HFT = cmap(0.85)

            for i in np.arange(3):
                for j in np.arange(3):
                    axs[i,j].axhline(y=default[i,j], color='gray', linestyle='--')
                    axs[i,j].plot(f_LFT*1e-9, M_LFT[:,i,j], color=col_LFT, label='LFT')#, alpha=0.9)
                    axs[i,j].plot(f_MFT*1e-9, M_MFT[:,i,j], color=col_MFT, label='MFT')#, alpha=0.9)
                    axs[i,j].plot(f_HFT*1e-9, M_HFT[:,i,j], color=col_HFT, label='HFT')#, alpha=0.9)
                    axs[i,j].tick_params(direction='in')
                    axs[i,j].set_xticks([100,400])
                    if i != 2:
                        axs[i,j].set_xticklabels(['', ''])
                    axs[i,j].locator_params(axis='y', nbins=3)
                    axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            axs[2,1].set_xlabel('Frequency [GHz]', labelpad=5)
            axs[1,0].set_ylabel('HWP Mueller matrix elements', labelpad=5)
            axs[2,2].legend (loc='upper right')

            fig.set_size_inches(6, 4)
            plt.tight_layout(pad=0.3)
            fig.subplots_adjust(wspace=0.35)
            plt.savefig('output/muellers.pdf')
            plt.clf()  
        return     

    def _effective_SEDs(self, SED, muellers, ideal=True):
        mIIs = np.copy(muellers[:,0,0])
        mQQs = np.copy(muellers[:,1,1])
        mUUs = np.copy(muellers[:,2,2])
        mQUs = np.copy(muellers[:,1,2])
        mUQs = np.copy(muellers[:,2,1])
        
        if ideal:
            mIIs = np.ones_like(mIIs)
            mQQs = np.ones_like(mQQs)
            mUUs = -np.ones_like(mUUs)
            mQUs = np.zeros_like(mQUs)
            mUQs = np.zeros_like(mUQs)
    
        a   = np.mean((SED[:-1]+SED[1:])/2)
        g   = np.mean((SED[:-1]*mIIs[:-1]+SED[1:]*mIIs[1:])/2)
        rho = np.mean((SED[:-1]*(mQQs-mUUs)[:-1]+SED[1:]*(mQQs-mUUs)[1:])/4)
        eta = np.mean((SED[:-1]*(mQUs+mUQs)[:-1]+SED[1:]*(mQUs+mUQs)[1:])/4)
        return a, g, rho, eta
        
    def compute_covs(self):        
        Nls, Bls = np.empty((2,self.nchan,self.lmax+1))  
        a_cmb, g_cmb, rho_cmb, eta_cmb = np.empty((4,self.nchan))
        a_dus, g_dus, rho_dus, eta_dus = np.empty((4,self.nchan))
        a_syn, g_syn, rho_syn, eta_syn = np.empty((4,self.nchan)) 
           
        fwhms = np.empty(self.nchan)
        
        for i in np.arange(self.nchan):       
            Nls[i,:] = self.instrument.Nls(self.chan_dicts[i])
            Bls[i,:] = self.instrument.Bls(self.chan_dicts[i])
            
            fwhms[i] = self.chan_dicts[i]['fwhm']
            
            freqs_temp, muellers_temp = self.instrument.muellers_channel(self.chan_dicts[i])
            
            a_cmb[i], g_cmb[i], rho_cmb[i], eta_cmb[i] = self._effective_SEDs(self.input_SEDs.CMB(freqs_temp), muellers_temp, self.ideal)
            a_dus[i], g_dus[i], rho_dus[i], eta_dus[i] = self._effective_SEDs(self.input_SEDs.dus(freqs_temp), muellers_temp, self.ideal)
            a_syn[i], g_syn[i], rho_syn[i], eta_syn[i] = self._effective_SEDs(self.input_SEDs.syn(freqs_temp), muellers_temp, self.ideal)
       
        #if self.gain_calibration:
            #print('(rho_cmb/g_cmb)**2 averages to ', np.mean((rho_cmb/g_cmb)**2))
            #print('(eta_cmb/g_cmb)**2 averages to ', np.mean((eta_cmb/g_cmb)**2))        
        #else:
            #print('(rho_cmb)**2 averages to ', np.mean((rho_cmb)**2))
            #print('(eta_cmb)**2 averages to ', np.mean((eta_cmb)**2))
       
        if self.fews:
            # filling covariance matrices    
            N_cov_model, N_cov, B_cov_CMB, B_cov_FGs = np.zeros((4,self.lmax+1,self.nchan,self.nchan))
          
            for i in np.arange(self.nchan):
                if self.gain_calibration:
                    prefactor_noise = 1/Bls[i]**2/g_cmb[i]**2
                else:
                    prefactor_noise = 1/Bls[i]**2
            
                N_cov[:,i,i] = prefactor_noise*Nls[i]
                N_cov_model[:,i,i] = prefactor_noise*Nls[i]
            
                for j in np.arange(self.nchan):
                    if self.gain_calibration:	
                        prefactor_sky = 1/(g_cmb[i]*g_cmb[j])
                    else:
                        prefactor_sky = 1
                        
                    B_cov_CMB[:,i,j] = prefactor_sky*(rho_cmb[i]*rho_cmb[j]*self.Cls_cmb[2] + eta_cmb[i]*eta_cmb[j]*self.Cls_cmb[1])
                    B_cov_FGs[:,i,j] = prefactor_sky*(rho_dus[i]*rho_dus[j]*self.Cls_dus[2] + eta_dus[i]*eta_dus[j]*self.Cls_dus[1]
                                               + rho_syn[i]*rho_syn[j]*self.Cls_syn[2] + eta_syn[i]*eta_syn[j]*self.Cls_syn[1])
                
            B_cov_all = B_cov_CMB + B_cov_FGs + N_cov
            B_cov_FGs_noise = B_cov_FGs + N_cov
            return B_cov_all, B_cov_FGs_noise, N_cov_model
        else:
            # filling covariance matrices    
            N_cov_model, N_cov, B_cov_CMB, B_cov_FGs, B_cov_cmb_rho, B_cov_cmb_eta, B_cov_dus_rho, B_cov_dus_eta, B_cov_syn_rho, B_cov_syn_eta = np.zeros((10,self.lmax+1,self.nchan,self.nchan))
          
            for i in np.arange(self.nchan):
                if self.gain_calibration:
                    prefactor_noise = 1/Bls[i]**2/g_cmb[i]**2
                else:
                    prefactor_noise = 1/Bls[i]**2
            
                N_cov[:,i,i] = prefactor_noise*Nls[i]
                N_cov_model[:,i,i] = prefactor_noise*Nls[i]
            
                for j in np.arange(self.nchan):
                    if self.gain_calibration:	
                        prefactor_sky = 1/(g_cmb[i]*g_cmb[j])
                    else:
                        prefactor_sky = 1
                
                    B_cov_CMB[:,i,j] = prefactor_sky*(rho_cmb[i]*rho_cmb[j]*self.Cls_cmb[2] + eta_cmb[i]*eta_cmb[j]*self.Cls_cmb[1])
                    B_cov_FGs[:,i,j] = prefactor_sky*(rho_dus[i]*rho_dus[j]*self.Cls_dus[2] + eta_dus[i]*eta_dus[j]*self.Cls_dus[1]
                                               + rho_syn[i]*rho_syn[j]*self.Cls_syn[2] + eta_syn[i]*eta_syn[j]*self.Cls_syn[1])
                    B_cov_cmb_rho[:,i,j] = prefactor_sky*(rho_cmb[i]*rho_cmb[j]*self.Cls_cmb[2])
                    B_cov_cmb_eta[:,i,j] = prefactor_sky*(eta_cmb[i]*eta_cmb[j]*self.Cls_cmb[1])
                    B_cov_dus_rho[:,i,j] = prefactor_sky*(rho_dus[i]*rho_dus[j]*self.Cls_dus[2])
                    B_cov_dus_eta[:,i,j] = prefactor_sky*(eta_dus[i]*eta_dus[j]*self.Cls_dus[1])
                    B_cov_syn_rho[:,i,j] = prefactor_sky*(rho_syn[i]*rho_syn[j]*self.Cls_syn[2])
                    B_cov_syn_eta[:,i,j] = prefactor_sky*(eta_syn[i]*eta_syn[j]*self.Cls_syn[1])
                
            B_cov_all = B_cov_CMB + B_cov_FGs + N_cov
            B_cov_FGs_noise = B_cov_FGs + N_cov            
            return B_cov_all, B_cov_FGs_noise, N_cov_model, B_cov_cmb_rho, B_cov_cmb_eta, B_cov_dus_rho, B_cov_dus_eta, B_cov_syn_rho, B_cov_syn_eta
        
    def compute_weights(self, B_cov):
        e = np.ones(self.nchan)
        B_w_temp = np.zeros((self.lmax+1,self.nchan))
        for l in np.arange(2,self.lmax+1):
            B_cov_inv = np.linalg.inv(B_cov[l])
            for power in -np.arange(9):
                atol = 10.**power
                if not np.allclose(np.dot(B_cov_inv,B_cov[l]), np.diag(np.ones(self.nchan)), atol=atol):
                    #print('np.dot(cov,cov_inv) differs from identity (atol=1e' + str(power) + ') for ell = ' + str(l))
                    break
            B_w_temp[l,:] = np.dot(B_cov_inv,e)/(e.transpose().dot(B_cov_inv.dot(e)))           
        return B_w_temp
        
    def compute_HILC_solution(self, B_w, B_cov):
        B_w_temp = np.copy(B_w)
        return np.einsum('li,lj,lij->l', B_w_temp, B_w_temp, B_cov)  
        
    def _likelihood_func(ell, BB_obs, BB_prim, BB_model_minus_prim, r, A):
        fsky = 0.78
        
        BB_model = r*BB_prim + A*BB_model_minus_prim
        logPl = -fsky*(2*ell+1)/2*(BB_obs/BB_model + np.log(BB_model) - (2*ell-1)/(2*ell+1)*np.log(BB_obs))
        logL = np.sum(logPl)
        return logL  
        
    def estimate_r(self, BB_obs, ell, fsky, BB_prim, BB_model_minus_prim, r_sim, A_sim):       
        BB_obs = np.copy(BB_obs[ell])
        BB_prim = np.copy(BB_prim[ell])
        BB_model_minus_prim = np.copy(BB_model_minus_prim[ell])
        
        logL = np.zeros((len(r_sim),len(A_sim)))
        
        xs, ys = np.meshgrid(np.arange(len(r_sim)), np.arange(len(A_sim)), indexing='ij')
        
        r_ids = xs.flatten()
        A_ids = ys.flatten()
        
        # compute 2D likelihood
        for i in range(len(r_ids)):
                logL[r_ids[i],A_ids[i]] = Analysis._likelihood_func(ell, BB_obs, BB_prim, BB_model_minus_prim, r_sim[r_ids[i]], A_sim[A_ids[i]])        
        logL_norm = (logL - np.max(logL))
        likelihood = np.exp(logL_norm)
          
        ### profiling the likelihood        
        likelihood_r_p = np.max(likelihood, axis=1)
        
        r = r_sim[np.argmax(likelihood_r_p)]
        
        dr = r_sim[1:]-r_sim[:-1]
        
        int0 = np.sum((likelihood_r_p[1:]+likelihood_r_p[:-1])*dr/2) 
        likelihood_r_p /= int0
        
        int1 = np.sum((likelihood_r_p[1:]*r_sim[1:]**2+likelihood_r_p[:-1]*r_sim[:-1]**2)*dr/2)
        int2 = (np.sum((likelihood_r_p[1:]*r_sim[1:]+likelihood_r_p[:-1]*r_sim[:-1])*dr/2))**2
        
        sigma_r = (int1-int2)**0.5
        
        def argwhere_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
            
        r_LEFT = r_sim[:np.argmax(likelihood_r_p)+1]
        r_RIGH = r_sim[np.argmax(likelihood_r_p):]
        
        likelihood_r_p_LEFT = likelihood_r_p[:np.argmax(likelihood_r_p)+1]     
        likelihood_r_p_RIGH = likelihood_r_p[np.argmax(likelihood_r_p):]
        
        L_array = np.linspace(np.max(likelihood_r_p),0,1000)
        int_lr = np.empty_like(L_array)
        
        for i in np.arange(len(L_array)):
            idx_l = argwhere_nearest(likelihood_r_p_LEFT, L_array[i])
            idx_r = np.argmax(likelihood_r_p) + argwhere_nearest(likelihood_r_p_RIGH, L_array[i])
            dr_lr = r_sim[idx_l+1:idx_r+1]-r_sim[idx_l:idx_r]
            int_lr[i] = np.sum((likelihood_r_p[idx_l+1:idx_r+1]+likelihood_r_p[idx_l:idx_r])*dr_lr/2)
           
        L_sigma = L_array[argwhere_nearest(int_lr/int_lr[-1], 0.68)]
        idx_l = argwhere_nearest(likelihood_r_p_LEFT, L_sigma)
        idx_r = np.argmax(likelihood_r_p) + argwhere_nearest(likelihood_r_p_RIGH, L_sigma)
        
        plus = r_sim[idx_r]-r
        minus = r-r_sim[idx_l]
        print('the MLE for r is ', r)
        print('plus  ', plus)
        print('minus ', minus)
               
        ###    
        likelihood_A_p = np.max(likelihood, axis=0)

        A = A_sim[np.argmax(likelihood_A_p)]
        
        dA = A_sim[1:]-A_sim[:-1]
        
        int0 = np.sum((likelihood_A_p[1:]+likelihood_A_p[:-1])*dA/2) 
        likelihood_A_p /= int0
        
        int1 = np.sum((likelihood_A_p[1:]*A_sim[1:]**2+likelihood_A_p[:-1]*A_sim[:-1]**2)*dA/2)
        int2 = (np.sum((likelihood_A_p[1:]*A_sim[1:]+likelihood_A_p[:-1]*A_sim[:-1])*dA/2))**2
        
        sigma_A = (int1-int2)**0.5
        
        A_LEFT = A_sim[:np.argmax(likelihood_A_p)+1]
        A_RIGH = A_sim[np.argmax(likelihood_A_p):]
        
        likelihood_A_p_LEFT = likelihood_A_p[:np.argmax(likelihood_A_p)+1]     
        likelihood_A_p_RIGH = likelihood_A_p[np.argmax(likelihood_A_p):]
        
        L_array = np.linspace(np.max(likelihood_A_p),0,1000)
        int_lr = np.empty_like(L_array)
        
        for i in np.arange(len(L_array)):
            idx_l = argwhere_nearest(likelihood_A_p_LEFT, L_array[i])
            idx_r = np.argmax(likelihood_A_p) + argwhere_nearest(likelihood_A_p_RIGH, L_array[i])
            dA_lr = A_sim[idx_l+1:idx_r+1]-A_sim[idx_l:idx_r]
            int_lr[i] = np.sum((likelihood_A_p[idx_l+1:idx_r+1]+likelihood_A_p[idx_l:idx_r])*dA_lr/2)
           
        L_sigma = L_array[argwhere_nearest(int_lr/int_lr[-1], 0.68)]
        idx_l = argwhere_nearest(likelihood_A_p_LEFT, L_sigma)
        idx_r = np.argmax(likelihood_A_p) + argwhere_nearest(likelihood_A_p_RIGH, L_sigma)
        
        plus = A_sim[idx_r]-A
        minus = A-A_sim[idx_l]
        print('the MLE for A is ', A)
        print('plus  ', plus)
        print('minus ', minus)
        
        ### marginalizing the likelihood
        likelihood_r_m = np.sum((likelihood[:,1:]+likelihood[:,:-1])*dA/2, axis=1)
        likelihood_A_m = np.sum((likelihood[1:,:]+likelihood[:-1,:]).transpose()*dr/2, axis=1)
        
        ### comparing profiled and marginalized likelihoods
        test0 = np.sum(np.abs((likelihood_r_m/np.max(likelihood_r_m)-likelihood_r_p/np.max(likelihood_r_p))))
        print('integrated deviation for r: ', test0/np.sum(likelihood_r_p/np.max(likelihood_r_p)))
        
        test0 = np.sum(np.abs((likelihood_A_m/np.max(likelihood_A_m)-likelihood_A_p/np.max(likelihood_A_p))))
        print('integrated deviation for A: ', test0/np.sum(likelihood_A_p/np.max(likelihood_A_p)))
        
        if self.pm == True:
            return r, plus, minus, likelihood_r_p, A, sigma_A, likelihood_A_p   
        else:
            return r, sigma_r, likelihood_r_p, A, sigma_A, likelihood_A_p             
