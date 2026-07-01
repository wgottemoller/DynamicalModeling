# ======================
# functions for fitting basic sersic, mass profiles, etc.
# ======================

import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # assume fiducial cosmology

# for sersic profile
from photutils.segmentation import detect_sources
import statmorph

# for q_intr prior
from scipy.interpolate import interp1d
import scipy as sp

import mgefit as mge
import jampy as jam

# plotting
from . import plotting_util


# constants
from astropy.constants import G
import astropy.units as u
c = 299792458 * u.m/u.s # m/s
G = G.to(u.pc * u.M_sun**-1 * (u.m / u.s)**2) # convert to units for EPL

# Create a mask (from Gottemoller+26 methodology)
def mask_center_2d(center_x, center_y, r, x_grid, y_grid):
    x_shift = x_grid - center_x
    y_shift = y_grid - center_y
    R = np.sqrt(x_shift * x_shift + y_shift * y_shift)
    mask = np.empty_like(R, dtype="int")
    mask[R > r] = 1
    mask[R <= r] = 0
    return mask


# find the inclination angle from axis ratio
def find_inclination(q, q_intr = 0.7): 
    """params
    - q: observed axis ratio
    - q_intr: 0.7, see Fig. 13 in https://arxiv.org/pdf/2301.02656
    """

    # preterms
    num = np.abs(q**2 - q_intr**2)
    denom = 1 - q_intr**2

    # obtain inclination
    i = np.arccos(np.sqrt(num / denom))

    # convert to degrees
    i = np.degrees(i)

    return i


# RMS velocity uncertainty approximation
def rms_vel_uncertainty(sigma, v_rot, sigma_err, v_rot_err):
    """
    params
    - sigma: velocity dispersion
    - v_rot: rotation velocity
    - sigma_err: uncertainty in velocity dispersion
    - v_rot_err: uncertainty in rotation velocity

    returns rms velocity uncertainty from standard uncertainty propagation
    """

    # calculate v_rms
    v_rms = np.sqrt(v_rot**2 + sigma**2)

    # calculate uncertainty
    sigma_rat = (sigma / v_rms) * sigma_err
    v_rot_rat = (v_rot / v_rms) * v_rot_err

    v_rms_err = np.sqrt(sigma_rat**2 + v_rot_rat**2)

    return v_rms_err


# epl with mass density as critical density times it. 
# See eqn 2 in Tessore & Metcalf (2016)
def epl_phys(r, q, z_defl, z_src, theta_E, gamma = 2, H0 = 70, Omega_m = 0.3):
    """
    params
    - x, y: coords, in arcseconds
    - q: axis ratio (should be measured from light)
    - z_defl: ...
    - z_src: ...
    - gamma: assumed isothermal
    - theta_E: Einstein radius in arcseconds
    - H0: assumed 70
    - Omega_m: assumed 0.3

    returns
    - mass surface density (kappa * Sigma_crit) in M_sun / pc^2
    """

    

    # measure critical density
    cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m) # assume fiducial cosmology
    
    d_l = cosmo.angular_diameter_distance(z_defl).to('pc') # in pc
    d_s = cosmo.angular_diameter_distance(z_src).to('pc')
    d_ls = cosmo.angular_diameter_distance_z1z2(z_defl, z_src).to('pc')

    ratio = c**2 / (4* np.pi * G)
    ratio_2 = d_s / (d_l * d_ls)
    sigma_crit = ratio * ratio_2

    if sigma_crit.unit != u.M_sun / u.pc**2:
        raise ValueError(f"units of sigma crit are {sigma_crit.unit}, expected M_sun / pc^2")
    else:
        sigma_crit = sigma_crit.value  # convert to M_sun / pc^2

    # apply the EPL to get convergence

    t = gamma - 1

    b = theta_E # assumption for isothermal case, need to check
    #elif gamma != 2: # worry about this later

    # convert x, y to r
    #r = np.sqrt(x**2 * q**2 + y**2)
    
    dist_ratio = (b / r)**(t)
    kappas = ((2 - t) / 2) * dist_ratio

    # convert to mass surface density
    Sigma = np.array(kappas) * sigma_crit # in M_sun / pc^2

    return r, Sigma

def pc_to_arcsec(pc, d_l_pc):
    """convert pc to arcseconds"""
    return (pc/d_l_pc) * 206265

def arcsec_to_pc(arcsec, d_l_pc):
    """convert arcseconds to pc"""
    return (arcsec * d_l_pc) / 206265



# assign correct vrms to each bin/spaxel individually



class DynamicalModel():
    def __init__(self, filter, image, z_defl, z_src, priors, vrms, vrms_err, center, xcoords, 
                 ycoords, bin_mapping, covariance_matrix, range = True):
        """
        params:
        - filter: HST filter, either F200LP or F140W
        - image: img cutout
        - z_defl: deflector redshift
        - z_src: source redshift
        - priors: dictionary of priors for the model, including:
            - theta_E: Einstein radius in arcsec (value)
            - gamma: power law slope (value, edge)
            - beta: anisotropy parameter (min, max)
            - mbh: black hole mass in M_sun (min, max)
            - oblate or prolate: boolean for oblate or prolate assumption
        - vrms: RMS velocity dispersion
        - vrms_err: uncertainty in RMS velocity dispersion
        - center: (x, y) center of the galaxy in pixels
        - xcoords: x coordinates of the spaxels
        - ycoords: y coordinates of the spaxels
        - covariance_matrix: covariance matrix for the velocity dispersions (from spatially resolved kinematics map)
        """


        # params found in HST documentation, not exact. will need stellar PSFs
        if filter == 'F140W':
            self.pixel_scale = 0.1
            self.psf_fwhm = 0.14
            self.photflam = 1.4713e-20 # erg cm^-2 A^-1 e^-1
            self.photbw = 1132.39 # A
            self.exp_time = 597.694383
            self.zp = 25.654 # zeropoint
            self.solar = 4.60 # absolute magnitude of sun in F140W (found online, may need to check later if necessary)
        elif filter == 'F200LP':
            self.pixel_scale = 0.05
            self.psf_fwhm = 0.075
        else:
            raise ValueError("check the filter used")
        
        # priors
        self.priors = priors
        if range:
            try:
                self.q_intr = priors['q_intr']
            except:
                self.q_intr = None
            self.mbh = 10**np.random.uniform(priors['mbh'][0], priors['mbh'][1]) 
            self.theta_E = np.random.normal(priors['theta_E'][0], priors['theta_E'][1]) 
            self.beta = np.random.uniform(priors['beta'][0], priors['beta'][1])
            self.gamma = 2 + np.random.uniform(priors['gamma'][0], priors['gamma'][1]) 
            self.oblate = priors['oblate or prolate']
        else:
            self.mbh = 10**(4)
            self.theta_E = 3.2
            self.beta = 0
            self.gamma = 2
            self.oblate = 'oblate'
            self.q_intr = 0.9
        # need to edit for if we give fixed values for params
        
        # initialize vrms and errors
        self.vrms = vrms
        self.vrms_err = vrms_err

        # cutout
        self.cutout = image

        # redshifts and distances
        self.z_defl = z_defl
        self.z_src = z_src
        self.d_l_pc = cosmo.angular_diameter_distance(z_defl).to('pc').value # in pc
        self.d_l_mpc = cosmo.angular_diameter_distance(z_defl).to('Mpc').value # in Mpc

        # center
        self.x_center = center[0]
        self.y_center = center[1]

        # coords
        self.xcoords = xcoords
        self.ycoords = ycoords
        self.bin_mapping = bin_mapping

        # kcwi details
        self.kcwi_pixel_scale = 0.294 # arcsec/pix
        self.kcwi_psf_fwhm = 1 # arcsec, measure from standard star later on
        self.kcwi_psf_sigma = self.kcwi_psf_fwhm / 2.355 # psf sigma go back later

        # covariance
        self.cov = covariance_matrix
        self.inv_cov = sp.linalg.inv(self.cov)

    
    def assign_vrms_to_spaxel(self):
        diff_x = self.xcoords # center is already assigned in binning algorithm
        diff_y = self.ycoords

        # create 2D array of x and y coordinates
        diff_2d = np.stack((diff_x, diff_y), axis=-1) # shape (n_y, n_x, 2)

        # keep only coordinates that belong to a valid bin
        self.valid_mask = self.bin_mapping != -1

        # coordinates for valid pixels
        diff_2d_valid = diff_2d[self.valid_mask]

        # map each valid coordinate to its bin value
        self.bin_ids = self.bin_mapping[self.valid_mask].astype(int)

        # assign kinematics to each valid coordinate
        self.assigned_rms_velocity = self.vrms[self.bin_ids]
        self.assigned_rms_velocity_uncertainty = self.vrms_err[self.bin_ids]

        # extract associated coordinates
        self.assigned_x_coords = diff_2d_valid[:, 0]
        self.assigned_y_coords = diff_2d_valid[:, 1]

        return self.assigned_rms_velocity, self.assigned_rms_velocity_uncertainty, self.assigned_x_coords, self.assigned_y_coords
        


    def luminous_component(self, model_lum = None):
        """
        use mgefit to find the galaxy params (will 
        be replaced with pyautolens sersic in later version)
        """

        if model_lum != None:
            self.model_lum = model_lum
            return self.model_lum
        
        # identify gal
        f= mge.find_galaxy(self.cutout, fraction = 0.2, plot = True)
        plt.show()


        theta = f.theta
        self.position_theta = f.theta
        self.pa = f.pa
        eps = f.eps
        self.q = 1- eps
        x_center = f.xpeak
        y_center = f.ypeak
        self.mge_x_center = x_center
        self.mge_y_center = y_center

        # obtain sectors
        sectors = mge.sectors_photometry(self.cutout, eps, theta, 
                                        x_center, y_center, plot = True, ) 
        plt.show()
        # couldn't find source code, so just used chat example

        # fit multigaussians
        r = sectors.radius
        print("Radii:", r)
        thetas = sectors.angle
        counts = sectors.counts

        # fit the MGE model to the sectors
        m = mge.fit_sectors(r, thetas, counts, eps, theta, plot = True, ngauss = 20)
        plt.show()

        # solution
        model_lum = m.sol
        model_lum[2] = np.full_like(model_lum[0], self.q) # set the axis ratio to the MGE axis ratio
        #model_lum[2] = np.full_like(model_lum[0], self.q) # set the axis ratio to the MGE axis ratio

        model_lum[1] *= self.pixel_scale # DOESNT WORK IF INCLUDE, not sure why????
        #model_lum[1] = model_lum[1] # deconvolve the PSF

        model_lum[0] /= (2* np.pi * model_lum[1]**2 * model_lum[2]) # Gaussian normalization, see mgefit docs + Anowar's code. Luminosity per arcsecond squared ideally
        
        """
        # surface luminosity density in L_sun / pc^2
        model_lum[0] = (model_lum[0] / self.pixel_scale**2) * 10**(-0.4 * (self.zp - self.solar - 21.572)) # convert to L_sun / pc^2, 21.572 is standard conversion factor from wiki
        print('yoooo', model_lum[0])"""
        #model_lum[2] = self.q # set the axis ratio to the MGE axis ratio

        self.r = r
        self.thetas = thetas
        self.model_lum = tuple(np.asarray(arr, dtype=float) for arr in model_lum)

        return self.model_lum, self.r, self.thetas
    

    # obtain the potential using EPL for the dynamical model

    def potential_component(self, plot = False, mass_follows_light = False, ml = 5):
        """
        potential component model with EPL"""

        # testing with constant ML ratio
        if mass_follows_light:
            self.model_pot = np.full_like(self.model_lum, 0.0)
            self.model_pot[0] = ml * self.model_lum[0] # scale the surface luminosity density by a factor of ml to get the surface mass density
            self.model_pot[1] = self.model_lum[1] # keep the same sigma
            self.model_pot[2] = self.model_lum[2] # keep the
            return self.model_pot

        # r originally in pixels
        r_arcsec = np.logspace(-2, 2, 1000)
        #r_arcsec = self.r * self.pixel_scale

        # apply the conversion to arcseconds for the coordinates
        """x_arcsec = r_arcsec * np.cos(self.thetas)
        y_arcsec = r_arcsec * np.sin(self.thetas)"""

        # convert to pc cuz simpler
        """x_pc = arcsec_to_pc(x_arcsec, self.d_l_pc)
        y_pc = arcsec_to_pc(y_arcsec, self.d_l_pc)"""

        # convert theta_E to pc
        #self.theta_E_pc = arcsec_to_pc(self.theta_E, self.d_l_pc)
        
        # surface densities from EPL
        radii, surface_densities = epl_phys(r_arcsec, self.q, self.z_defl, self.z_src, self.theta_E, gamma = self.gamma)

        # apply the mge fit on the potential component
        m = mge.fit_1d(radii, surface_densities, plot = True, ngauss = 7)
        plt.show()

        surf = m.sol[0]
        sigma = m.sol[1]

        # make necessary alterations
        surf /= np.sqrt(2* np.pi * self.q * (sigma)**2) # Gaussian normalization for 1D
        sigma = pc_to_arcsec(sigma, self.d_l_pc) # convert sigma to arcseconds

        model_pot = (surf, sigma, np.full_like(surf, self.q))
        self.model_pot = tuple(np.asarray(arr, dtype=float) for arr in model_pot)

        # plotting
        if plot == True:
            plotting_util.unified_plotting_style(xmin = 0.9 * radii.min(), xmax = 1.1* radii.max(), ymin = 0.9 * surface_densities.min(), ymax = 1.1 * surface_densities.max(),
                                 figsize = (16, 16), x_major_locator = 1e3, x_minor_locator = 4e2, y_major_locator = 5e4, xlabel = r'$R_{\rm{ell}}$ ($\log_{10}$pc)', ylabel = r'$\kappa(R_{\rm{ell}}) \Sigma_{\rm{cr}}$ ($\log_{10}M_\odot$/pc$^2$)')
            plt.xscale('log')
            plt.yscale('log')
            plt.scatter(radii, surface_densities, color = 'blue', s = 250)
            plt.show()

        return self.model_pot
    

    # intrinsic q prior, from TDCOSMO XII paper Shajib Github repo
    def get_intrinsic_q_prior(self):
        """
        Get the intrinsic q prior. The values are taken from Chang et al.
        2013 (figure 7, https://ui.adsabs.harvard.edu/abs/2013ApJ...773
        ..149C/abstract)
        :param intrinsic_q: intrinsic q value
        """
        if self.q_intr is None:
            if self.oblate == 'oblate':
                scrapped_points = np.array([
                    0.16240221274286662, 0.0,
                    0.2029264579533636, 0.1911580724417048,
                    0.2407752136623306, 0.39252068793994876,
                    0.2777045053735887, 0.5686809459927957,
                    0.30559742550080704, 0.6692868878404652,
                    0.34066141115113935, 0.7656308879610525,
                    0.3775002637806548, 0.8409515698717298,
                    0.41704475227228144, 0.9330562380356633,
                    0.45573006948736117, 1.0671849328489817,
                    0.48546945420014154, 1.2265988875992946,
                    0.5152389852735029, 1.4196260343969964,
                    0.5468549809324269, 1.671461194097342,
                    0.5712697647076556, 1.8939451034773822,
                    0.5938493887825392, 2.070225946972551,
                    0.6155246220400041, 2.2381110290460176,
                    0.63809294122967, 2.4017869255234157,
                    0.6606085042883197, 2.5066397359178803,
                    0.6803958217144235, 2.5694986660235433,
                    0.7019467012344933, 2.598729330901526,
                    0.7252272281928764, 2.556516889498514,
                    0.7413103115626366, 2.4891548467811213,
                    0.7555846132975597, 2.405001281220324,
                    0.7698438418521923, 2.304041119635831,
                    0.7805231900878766, 2.2115144024237665,
                    0.7894012932788689, 2.110599460380145,
                    0.7991687141069892, 2.00127368373453,
                    0.815131212034427, 1.7994588728275769,
                    0.8364145426043439, 1.530372458284973,
                    0.8567633359962618, 1.2192770902732764,
                    0.8833298162579323, 0.8409025820357834,
                    0.9063239527907992, 0.4793648161825659,
                    0.9275922101804258, 0.19347180561626676,
                    0.9435697812881539, 0.0,
                ])
            else:
                scrapped_points = np.array([
                    0.16868396062885307, 0.010773555612499486,
                    0.2117706464887026, 0.052428289344770285,
                    0.2477164131860181, 0.13195815685151402,
                    0.28546342492802557, 0.21987624918981652,
                    0.3214204965105588, 0.31201106371433385,
                    0.3537826145938532, 0.39577272658758345,
                    0.3951321164252446, 0.5004672685889999,
                    0.4311155660732857, 0.6220136261549825,
                    0.46082103613041325, 0.7436127398519812,
                    0.4869503941636647, 0.877846946927332,
                    0.5067942360158569, 1.0037306121218514,
                    0.5284543960930316, 1.154809098171623,
                    0.5456152118535489, 1.2891186711484255,
                    0.5636879550216299, 1.4402273035587774,
                    0.5826726255972747, 1.6081349954026796,
                    0.6016648327630647, 1.7844459852584293,
                    0.6215689673364183, 1.977556034547729,
                    0.6423699561370454, 2.170658547246883,
                    0.6676627526641847, 2.372126675007159,
                    0.6902273035587777, 2.5316009224786327,
                    0.7181503700465761, 2.6658200563736933,
                    0.7576647121776222, 2.724311532490239,
                    0.795204467690638, 2.5811389295027345,
                    0.8210398987082287, 2.3876445141160323,
                    0.8432653030462894, 2.168970351054368,
                    0.8645410970260615, 1.891480638499916,
                    0.8822520838671751, 1.6392309663415872,
                    0.8990737530711602, 1.3953921287852515,
                    0.915865275914565, 1.1179400991815256,
                    0.9317750177109868, 0.8573022021916397,
                    0.9485590039642465, 0.5714468745760661,
                    0.9662398444447794, 0.28558401037034775,
                    0.9884049560616796, 0.0,
                ])

            x = scrapped_points[::2]
            y = scrapped_points[1::2]

            self.interp_q_int = interp1d(x, y, bounds_error=False,
                                                      fill_value=0.)
            
            p = y / np.sum(y)
            self.q_intr = np.random.choice(x, p=p)
        else:
            pass

        return np.log(self.interp_q_int(self.q_intr)), self.q_intr
    
    def find_inclination(self):
        """find inclination angle from axis ratio"""

        q_intr = self.q_intr # see Fig. 13 in https://arxiv.org/pdf/2301.02656

        # preterms
        num = np.abs(self.q**2 - q_intr**2)
        denom = 1 - q_intr**2

        # obtain inclination
        i = np.arccos(np.sqrt(num / denom))

        # convert to degrees
        i = np.degrees(i)

        self.inclination_angle = i
        return self.inclination_angle
    
    # rotate the coordinates to align with PA
    def rotate_coords(self):
        """rotate coords"""

        # convert PA to radians
        self.position_theta = np.radians(self.position_theta)

        # apply rotation
        x_prime = self.xcoords * np.cos(self.position_theta) + self.ycoords * np.sin(self.position_theta) # NEED TO CHECK THIS LATER ON
        y_prime = -self.xcoords * np.sin(self.position_theta) + self.ycoords * np.cos(self.position_theta)

        # assign the rotated coordinates
        self.xcoords = x_prime
        self.ycoords = y_prime

        return self
    

    # run dynamical model
    def do_dynamical_model(self, get_ll = False, get_plot = True):
        """calculate jampy model"""


        try:
            model = jam.axi.proj(
                self.model_lum, self.model_pot,
                distance=self.d_l_mpc, inc=self.inclination_angle, mbh=self.mbh,
                beta=self.beta, xbin=self.assigned_x_coords, ybin=self.assigned_y_coords,
                data=self.assigned_rms_velocity, errors=self.assigned_rms_velocity_uncertainty, 
                sigmapsf = self.kcwi_psf_sigma, pixsize = self.kcwi_pixel_scale, plot = get_plot)
        except:
            # if show up a nan, adjust params to get correct result
            print("NaN values found in model, adjusting parameters...")
            self.mbh *= 1.1  # Increase black hole mass by 10%
            self.beta *= 0.9  # Decrease anisotropy parameter by 10%
            model = jam.axi.proj(
                self.model_lum, self.model_pot,
                distance=self.d_l_mpc, inc=self.inclination_angle, mbh=self.mbh,
                beta=self.beta, xbin=self.assigned_x_coords, ybin=self.assigned_y_coords,
                data=self.assigned_rms_velocity, errors=self.assigned_rms_velocity_uncertainty, 
                sigmapsf = self.kcwi_psf_sigma, pixsize = self.kcwi_pixel_scale, plot = get_plot)
        
        self.model = model

        print("model_pot surf:", self.model_pot[0])
        print('model lum surf:', self.model_lum[0])
        print('model_lum sigma arcsec:', self.model_lum[1])
        print("model_pot sigma arcsec:", self.model_pot[1])
        print("model_pot q:", self.model_pot[2])
        print("model min/max:", np.nanmin(self.model.model), np.nanmax(self.model.model))
        print("nan count:", np.isnan(self.model.model).sum())
        print("zero/negative count:", np.sum(self.model.model <= 0))


        # get log likelihood if configured, NEED TO ALTER FOR SIGMA CLIPPING
        self.model_pred_fin = [[] for _ in np.arange(len(self.vrms))]
        if get_ll:
            for mpd, flxs, bin_no in zip(self.model.model, self.model.flux, self.bin_ids):
                self.model_pred_fin[int(bin_no)] += [[mpd, flxs]]
            model_pred_fin = np.array([np.sqrt(np.average(np.array(mpf).T[0]**2, weights=np.array(mpf).T[1])) for mpf in self.model_pred_fin])

            delta_pred = model_pred_fin - self.vrms
            self.jampy_ll = -0.5 * delta_pred.T.dot(self.inv_cov.dot(delta_pred)) # NEED TO ADJUST FOR SIGMA CLIPPING
            return self.jampy_ll, self.model



        return self.model
    

    # plot velocity map

    def plot_velocity_map(self):
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))

        plotting_util.plot_map(axes[0], fig, self.model.xbin, self.model.ybin, self.model.model, 
                        title=r'$V_{\rm{rms, model}}$ (km/s)', cmap='RdBu_r')
        cbar = fig.colorbar(axes[0].collections[0], ax=axes[0], orientation='vertical', 
                            fraction=0.046, pad=0.04, label = r'$V_{\rm{rms, model}}$ (km/s)')

        plotting_util.plot_map(axes[1], fig, self.model.xbin, self.model.ybin, self.assigned_rms_velocity, 
                        title=r'$V_{\rm{rms, data}}$ (km/s)', cmap='RdBu_r')
        cbar = fig.colorbar(axes[1].collections[0], ax=axes[1], orientation='vertical', 
                            fraction=0.046, pad=0.04, label = r'$V_{\rm{rms, data}}$ (km/s)')

        self.sigmas = self.match_model_to_bins()

        plotting_util.plot_map(axes[2], fig, self.model.xbin, self.model.ybin, self.sigmas, 
                        title=f'Residuals (Mean $\sigma = {np.abs(self.sigmas).mean():.2f}$)', cmap='RdBu_r',)
        cbar = fig.colorbar(axes[2].collections[0], ax=axes[2], orientation='vertical', 
                            fraction=0.046, pad=0.04, label = r'$\Delta V_{\rm{rms}} / \sigma_{\rm{data}}$')
    
        plt.show()


    def match_model_to_bins(self, return_vrms = False):
        """calculate sigma residuals between model and data"""

        snr_map = np.load('functions/snr.npy') # snr map for weighted spaxels

        snr_valid = snr_map[self.valid_mask]

        # map the bin ids to the model spaxels

        sigma_uncertainty = np.full_like(self.assigned_rms_velocity, np.nan)  # Initialize with NaN values
        
        sigmas = []
        weighted_vrms = []
        for bin in np.unique(self.bin_ids):
            bin_mask = self.bin_ids == bin
            model_bin_values = self.model.model[bin_mask]
            snr_values = snr_valid[bin_mask]
            data_values = self.assigned_rms_velocity[bin_mask]
            data_err = self.assigned_rms_velocity_uncertainty[bin_mask]

            # calculate weighted mean of model values for this bin
            weighted_mean_model = np.average(model_bin_values, weights=snr_values)
            residual = weighted_mean_model - np.mean(data_values)
            sigma = residual / np.mean(data_err)
            sigmas.append(sigma)
            sigma_uncertainty[bin_mask] = sigma
            weighted_vrms.append(weighted_mean_model)
        
        if return_vrms:
            return np.array(weighted_vrms)
        sigmas = np.array(sigmas)        


        print(f"Mean residual: {np.mean(sigmas):.2f}")
        print(f"Standard deviation residual: {np.std(sigmas):.2f}")

        return sigma_uncertainty
    
    def inclination_prior(self):
        """calculate inclination prior based on axis ratio and intrinsic axis ratio"""

        # calculate q log likelihood
        q_ll = sp.stats.norm.pdf(self.q_intr, 0.7, 0.1) # need to use TDCOSMO prior in the future, 
        # potentially (not doing super high precision cosmology though, so maybe fine)

        # calculate inclination angle
        i = find_inclination(self.q, self.q_intr)

        # calculate the log_ll
        log_ll = np.log(q_ll) + np.log(self.q) - np.log(self.q_intr) - np.log(np.sin(i*np.pi/180))

        return log_ll
    
    def sigma_clip_vrms(self, sigma_threshold = 3):
        """remove outliers from the sigma clipping
        - sigma threshold: maximum sigma * np.abs(bin_vrms - model) / bin_vrms_err threshold allowed
        """

        # calculate model prediction per bin
        sigmas = self.match_model_to_bins()
        
        # remove bins
        remove_bins = sigmas > sigma_threshold
        print('num bins clipped:', len(remove_bins))

        # clip
        vrms_clipped = self.vrms[~remove_bins]
        vrms_err_clipped = self.vrms_err[~remove_bins]

        return # PERFORM TESTING FIRST AND THEN IMPLEMENT THE CLIPPING ALGORITHM


    def log_likelihood(self):
        """calculate log likelihood of all components. Will need to check other stuff later"""

        # log likelihood value
        ll_val = 0

        # inclination prior
        inclination_ll = self.inclination_prior()
        print('inclination ll', inclination_ll)
        ll_val += inclination_ll

        # mbh prior
        mbh_ll = sp.stats.uniform.logpdf(np.log10(self.mbh), self.priors['mbh'][0], self.priors['mbh'][1] - self.priors['mbh'][0])
        print('mbh ll', mbh_ll)
        ll_val += mbh_ll

        # beta prior
        beta_ll = sp.stats.uniform.logpdf(self.beta, self.priors['beta'][0], self.priors['beta'][1] - self.priors['beta'][0])
        print('beta ll', beta_ll)
        ll_val += beta_ll

        # gamma prior
        gamma_ll = sp.stats.uniform.logpdf(self.gamma, 2 + self.priors['gamma'][0], self.priors['gamma'][1] - self.priors['gamma'][0])
        print('gamma ll', gamma_ll)
        ll_val += gamma_ll

        # theta_E prior
        theta_E_ll = sp.stats.norm.logpdf(self.theta_E, self.priors['theta_E'][0], self.priors['theta_E'][1])
        print('theta_E ll', theta_E_ll)
        ll_val += theta_E_ll

        # jampy prior
        ll_jam = self.do_dynamical_model(get_ll = True)[0]
        print('jampy ll', ll_jam)
        ll_val += ll_jam

        # ======================
        # WILL NEED TO INCLUDE LL FROM MASS AND LIGHT MODELS PROBABLY, SEE WILLIAM'S CODE
        # ======================

        

        return ll_val
    

    def ll_after_pso(self, new_params, get_plot = True, velocity_map = False):
        """ calculate log likelihood of best fit params after PSO optimization
        
        - new_params: list of new params fitted for, including:
            - mbh
            - inclination
            - beta
            - gamma
            - theta_E
            - q_l"""

        self.inclination_angle = new_params[1]
        self.theta_E = new_params[4]
        self.gamma = new_params[3]
        self.beta = new_params[2]
        self.mbh = new_params[0]

        # run full model through again
        self.rotate_coords()
        self.assign_vrms_to_spaxel()
        self.luminous_component()
        self.potential_component()
        self.do_dynamical_model(get_plot = get_plot)
        if velocity_map:
            self.plot_velocity_map()
        self.log_likelihood()

        return self.log_likelihood()


# set up if need be later
"""class ModelOptimization():
    def __init__(self, )"""




