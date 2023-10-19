from __future__ import print_function, division
import pandas as pd
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy import constants as const
from astropy import units as u
from scipy.interpolate import splrep,splev
from scipy import interpolate 
from scipy import signal
import scipy.integrate as it
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors
import matplotlib as mpl
from scipy.optimize import curve_fit


##functions###
def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma], maxfev=5000)
    return popt

def cc2(vec1,wl1,vec2,wl2): 
    N=len(vec1)
    nc=750
    #lagvec = np.linspace(-20,20,nc) #units of km/s
    lagvec = np.linspace(-8,8,nc) #units of km/s
    corr=np.zeros((nc))
    for iv in range(nc):
        model = interpolate.splrep(wl1,vec1,s=0)
        wShift = wl1 * (1.0 - lagvec[iv]/2.998E5)
        fShift = splev(wShift,model,der=0)
        corr[iv]=np.sum((fShift-np.mean(fShift))*(vec2-np.mean(vec2)),axis=0)/N
        corr[iv]=corr[iv]/((np.std(fShift)*np.std(vec2)))
    #corr-=np.mean(corr)
    return lagvec,corr

def interp_x_y(x_input, y_input, max_v):
    f8 = interp1d(x_input, y_input / max(y_input), kind='cubic')
    x_new = np.linspace(-max_v, max_v, num=2500)
    y_new = f8(x_new)
    return x_new, y_new
# define Gaussian function
def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) * (x - x0) / (2 * sigma * sigma))

#####plot stuff###
# Figure aesthetics
fig, axes = plt.subplots(1, 3, figsize=(24.27, 30))
plt.subplots_adjust(wspace=0.04, hspace=0)

font = {'size' : 20, 'family' : 'sans-serif'}
plt.rc('font', **font)


base = '/Users/hbeltz/Documents/Spectra/wasp76b/dragwork/'

phases=np.linspace(0,348.75,32,endpoint=True)
#phases=[0.0,45.0,90.0,135.0,180.0,225.0,270.0]
#phases=[0.0]
#buda, batlow, roma, 
cm_name = 'roma'
cm_file = np.loadtxt('/Users/hbeltz/Documents/Research Codes/ScientificColourMaps5/roma/roma.txt')
my_colors = mcolors.LinearSegmentedColormap.from_list(cm_name, cm_file[::-1])
colors = np.linspace(0, 256, len(phases) + 1)
#print(colors)

alphas = [1, 0.8, 0.6, 0.4, 0.2]
dragfree_list  = []
drag3_list=[]
sigmas=[]
i=0

for phase in phases:
        phase = str(phase)
        color_val = int(colors[i]) #- 64
        extra=-0.5*i
        # Do the drag free one!
        file1 = base + '0G/Spec_0_{}_phase_{}_inc_0.00.00.0000.00.dat'.format('asp-76b-0g', phase)
        file2 = base + '0G/Spec_1_{}_phase_{}_inc_0.00.00.0000.00.dat'.format('ASP-76b-0g', phase)

        tw, tf = np.loadtxt(file1, unpack=True)
        dw, df = np.loadtxt(file2, unpack=True)
        tw=tw[10:-10]
        tf=tf[10:-10]
        dw=dw[10:-10]
        df=df[10:-10]

        # Get the cross correlation
        #rv, cc = pyasl.crosscorrRV(dw, df, tw, tf, -5., 5., 0.02, mode='doppler', skipedge=100)
        rv,cc=cc2(tf,tw,df,dw)
        # normalize cc function 
        cc = (cc - min(cc)) / (max(cc) - min(cc))

        # Find the index of maximum cross-correlation function
        maxind = np.argmax(cc)

        # fit Guassian to peak region of cc function
        rv_range = rv[maxind - 20: maxind + 20]
        cc_range = cc[maxind - 20: maxind + 20]

        #popt, pcov = curve_fit(gaussian, rv_range, cc_range, p0=[1, 0, 1])

        # calculate best fit guassian
        #rv_fit = np.linspace(rv[maxind - 20], rv[maxind + 20], 1000)
        #cc_fit = gaussian(rv_fit, *popt)

        # index of maximum of Gaussian
        #max_gauss = np.argmax(cc_fit)
        max_gauss2 = np.argmax(cc) 
        #print(rv[max_gauss2])

        # plot CCFs
        axes[0].plot(rv, cc + i+extra, lw=3, color=my_colors(color_val), alpha=alphas[0])
        axes[0].scatter(rv[max_gauss2], cc[max_gauss2]+i+extra, color=my_colors(color_val), s=50)
        #axes[0].plot(rv_fit[max_gauss], cc_fit[max_gauss] + i, '.', ms=15, color=my_colors(color_val))    
        #dragfree_list.append(rv_fit[max_gauss])

                # Do the magnetic one!
        file1 = base + '3G/Spec_0_{}_phase_{}_inc_0.00.00.0000.00.dat'.format('asp-76b-3G', phase)
        file2 = base + '3G/Spec_1_{}_phase_{}_inc_0.00.00.0000.00.dat'.format('ASP-76b-3G', phase)
        tw, tf = np.loadtxt(file1, unpack=True)
        tw=tw[10:-10]
        tf=tf[10:-10]
        dw, df = np.loadtxt(file2, unpack=True)
        dw=dw[10:-10]
        df=df[10:-10]

        # Get the cross correlation
        print('on then off')
        print(file2)
        print(file1)
        rv,cc=cc2(tf,tw,df,dw)
        #rv, cc = pyasl.crosscorrRV(dw, df, tw, tf, -5., 5., 0.02, mode='doppler',skipedge=100)

        # normalize cc function 
        cc = (cc - min(cc)) / (max(cc) - min(cc))

        # Find the index of maximum cross-correlation function
        maxind = np.argmax(cc)

        # fit Guassian to peak region of cc function
        rv_range = rv[maxind - 20: maxind + 20]
        cc_range = cc[maxind - 20: maxind + 20]

        #popt, pcov = curve_fit(gaussian, rv_range, cc_range, p0=[1, 0, 1])

        # calculate best fit guassian
        #rv_fit = np.linspace(rv[maxind - 20], rv[maxind + 20], 1000)
        #cc_fit = gaussian(rv_fit, *popt)

        # index of maximum of Gaussian
        #max_gauss = np.argmax(cc_fit)
        max_gauss2 = np.argmax(cc)
        print(rv[max_gauss2])
        #print (max_gauss2)
        #print (cc[max_gauss2])  

        # plot CCFs
        axes[1].plot(rv, cc + i+extra, lw=3, color=my_colors(color_val), alpha=alphas[0])
        axes[1].scatter(rv[max_gauss2], cc[max_gauss2]+i+extra, color=my_colors(color_val),s=50)


                        # Do the uniform!
        file1 = base + 'tdragshort/Spec_0_{}_phase_{}_inc_0.00.00.0000.00.dat'.format('ASP-76b-short', phase)
        file2 = base + 'tdragshort/Spec_1_{}_phase_{}_inc_0.00.00.0000.00.dat'.format('ASP-76b-short', phase)
        tw, tf = np.loadtxt(file1, unpack=True)
        tw=tw[10:-10]
        tf=tf[10:-10]
        dw, df = np.loadtxt(file2, unpack=True)
        dw=dw[10:-10]
        df=df[10:-10]

        # Get the cross correlation
        rv,cc=cc2(tf,tw,df,dw)
        #rv, cc = pyasl.crosscorrRV(dw, df, tw, tf, -5., 5., 0.02, mode='doppler',skipedge=100)

        # normalize cc function 
        cc = (cc - min(cc)) / (max(cc) - min(cc))

        # Find the index of maximum cross-correlation function
        maxind = np.argmax(cc)

        # fit Guassian to peak region of cc function
        rv_range = rv[maxind - 20: maxind + 20]
        cc_range = cc[maxind - 20: maxind + 20]

        #popt, pcov = curve_fit(gaussian, rv_range, cc_range, p0=[1, 0, 1])

        # calculate best fit guassian
        #rv_fit = np.linspace(rv[maxind - 20], rv[maxind + 20], 1000)
        #cc_fit = gaussian(rv_fit, *popt)

        # index of maximum of Gaussian
        #max_gauss = np.argmax(cc_fit)
        max_gauss2 = np.argmax(cc)
        #print (max_gauss2)
        #print (cc[max_gauss2])  

        # plot CCFs
        axes[2].plot(rv, cc + i+extra, lw=3, color=my_colors(color_val), alpha=alphas[0])
        axes[2].scatter(rv[max_gauss2], cc[max_gauss2]+i+extra, color=my_colors(color_val),s=50)
        #axes[1].plot(rv_fit[max_gauss], cc_fit[max_gauss] + i, '.', ms=15, color=my_colors(color_val))        
        #drag3_list.append(rv_fit[max_gauss])
        
        #val1, val2, sigma = popt
        #sigmas.append(sigma)

        i=i+1
#dragfree_rounded = [ round(elem, 4) for elem in dragfree_list ]
#cloudy_rounded = [ round(elem, 4) for elem in cloudy_list ]
#cloudy_ax = axes[1].twinx()
#x_ticks_labels = ['0/12', '2/12', '4/12', '6/12', '8/12', '10/12', '12/12']

axes[0].tick_params(labelsize=20)
axes[1].tick_params(labelsize=20)
axes[2].tick_params(labelsize=20)

axes[2].yaxis.set_ticks([])
axes[1].yaxis.set_ticks([])
#cloudy_ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 12.5])
#cloudy_ax.set_yticklabels(x_ticks_labels)

axes[0].xaxis.set_ticks([-6, -4, -2, 0, 2, 4, 6])
axes[1].xaxis.set_ticks([-6, -4, -2, 0, 2, 4, 6])
axes[2].xaxis.set_ticks([-6, -4, -2, 0, 2, 4, 6])

axes[0].set_xlim(-5, 5)
axes[1].set_xlim(-5, 5)
axes[2].set_xlim(-5, 5)

axes[0].text(-.25,-.5,"0 G", weight='bold')
axes[1].text(-.25,-.5,"3 G",weight='bold')
axes[2].text(-.75,-.5,"Uniform", weight='bold')

#axes[0].set_title('Drag Free')
#axes[1].set_title('3 G')
#axes[2].set_title('Uniform Drag')

#axes[0].set_xlabel('Doppler shift (km s$^{-1}$)',fontsize=22, weight='bold')
axes[1].set_xlabel('Doppler shift (km s$^{-1}$)',fontsize=28, weight='bold')
axes[0].set_ylabel('Cross correlation Function (+ offset)',fontsize=28, weight='bold')

#cloudy_ax.set_ylabel('Orbital Phase',fontsize=22, weight='bold')

axes[0].axvline(linewidth=1, color='k',linestyle='dashed')
axes[1].axvline(linewidth=1, color='k',linestyle='dashed')
axes[2].axvline(linewidth=1, color='k',linestyle='dashed')
    
sm = plt.cm.ScalarMappable(cmap=my_colors, norm=plt.Normalize(vmin=0, vmax=1))
sm._A = []
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), location='top', aspect=50, pad=0.02)
cbar.set_label('Orbital phase', fontsize=28, weight='bold')
plt.savefig('/Users/hbeltz/Documents/Spectra/wasp76b/dragwork/cc-3final.png', bbox_inches='tight', dpi=250)
