#!/usr/bin/env python
# coding: utf-8

import os
from os import path,mkdir

import numpy as np
from numpy.polynomial import chebyshev

import glob

import pickle

import scipy
from scipy.interpolate import interp1d, splev, splrep, UnivariateSpline
import scipy.signal as signal
from scipy.signal import medfilt
from scipy.io import readsav

import pandas as pd

# pretty plots
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns 
sns.set_context("talk")
#plt.style.use('dark_background')

from astropy.io import fits
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter
import astropy.units as u
from astropy.time import Time

import juliet

from barycorrpy import utc_tdb

from transitspectroscopy import spectroscopy

import batman

import lmfit

import dynesty

get_ipython().run_line_magic('matplotlib', 'inline')


# function that opens each orbit fits file and gets the science exposures
# optional kwargs to also get the dq extensions and jit vectors, for later use in cleaning/detrending the data
def get_data(files, dq = True, jit = True, keep_first_orbit = True):
    
    # initializing some lists to hold the data
    data_hold = []
    error_hold = []
    headers_hold = []
    jitter_hold = []
    dqs_hold = []
    
    # keeping track of how many exposures in each orbit
    visits_idx = []
    
    
    for fits_file in files:
        hdulist = fits.open(fits_file)
        sciextlist = []
        errextlist = []
        dqextlist = []

        # finding which fits extensions correspond to the science, error, and dq frames, if applicable
        for i in range(len(hdulist.info(0))):
            if hdulist.info(0)[i][1] == "SCI":
                sciextlist.append(i)
            
            if hdulist.info(0)[i][1] == "ERR":
                errextlist.append(i)
                
            if dq == True:
                if hdulist.info(0)[i][1] == "DQ":
                    dqextlist.append(i)
        
        # getting the data and header for each of the science frames
        data_lst = []
        header_lst = []
        for j in sciextlist:
            data, header = fits.getdata(fits_file, ext = j, header=True)
            # getting rid of one second exposures
            if header["EXPTIME"] !=  1.0:
                if header["ASN_MTYP"] == "REPEATOBS":
                    data_lst.append(data)
                    header_lst.append(header)
                else:
                    continue
            
            else:
                continue
                
        # getting the error frames
        error_lst = []
        for e in errextlist:
            err = fits.getdata(fits_file, ext = e, header = False)
            error_lst.append(err)

        # getting the data quality frames
        if dq == True:
            dq_lst = []
            for k in dqextlist:
                dqs = fits.getdata(fits_file, ext = k, header = False)
                dq_lst.append(dqs)
                
        if jit == True:
            # gets corresponding .jit file for each .fits file
            data, header = fits.getdata(fits_file, ext = j, header=True)
            # getting rid of one second exposures
            if header["EXPTIME"] !=  1.0:
                if header["ASN_MTYP"] == "REPEATOBS":
                    corresponding_jit_file = fits.open(fits_file.replace("flt","jit")) 

                    # gets the names of the different jitter vectors
                    jitter_vector_list = corresponding_jit_file[1].columns.names 

                    # initialize an intermediate list
                    jitter_lst = []

                    for jitter_hdu in corresponding_jit_file: # iterates through each exposure of each file
                        if jitter_hdu.name == 'jit':
                            dummy_jit_array = []
                            for jitvect in jitter_vector_list: # iterates through each jitter vector name
                                jitter_points = jitter_hdu.data[jitvect]
                                jitter_points[jitter_points > 1e30] = np.median(jitter_points) # kills weird edge cases
                                dummy_jit_array.append(np.mean(jitter_points)) # saves the mean jitter value inside of each exposure
                            jitter_lst.append(dummy_jit_array)
                else:
                    jitter_lst = []
            else:
                jitter_lst = []
        
        
        #i'm not returning this at the moment but can if they're needed
        visits_idx.append(len(data_lst)) 
        
        # adding each orbit's data to master list
        data_hold = data_hold + data_lst
        headers_hold = headers_hold + header_lst
        error_hold = error_hold + error_lst
        
        if dq == True:
            dqs_hold = dqs_hold + dq_lst
            
        if jit == True:
            jitter_hold = jitter_hold + jitter_lst
        
    if jit == True:
        # sort jitter vectors
        jitter_dict = {}
        for i in range(len(jitter_vector_list)):
            jitter_dict[jitter_vector_list[i]] = [item[i] for item in jitter_hold]
        
        
    
    if dq == True and jit == True:
        return data_hold, headers_hold, jitter_dict, dqs_hold, error_hold, visits_idx
    elif dq == True and jit == False:
        return data_hold, headers_hold, dqs_hold, error_hold, visits_idx
    elif dq == False and jit == True:
        return data_hold, headers_hold, jitter_dict, error_hold, visits_idx
    else:
        return data_hold, headers_hold, error_hold, visits_idx



# using the best medfilt window size of 5, find the points for which the residual value is 
# greater than 5 sigma and mark them - 1d
def residual_outliers_1d(spectra_cut, n = 5, window = 5): 
    medfilt_result = medfilt(spectra_cut, window)
    
    residuals = medfilt_result-spectra_cut
    stdev_residuals = 1.4826*np.nanmedian(np.abs(residuals - np.nanmedian(residuals)))
    #stdev_residuals = np.sqrt(np.var(residuals))
    outlier_locations = np.where(abs(residuals)>stdev_residuals*n)
    
    # option to plot positions of the outliers 
    #plt.figure(figsize=(15,10))
    #plt.imshow(spectra_cut)
    #plt.scatter(outlier_locations[1],outlier_locations[0], color = "red")
    #plt.show()
    
    return outlier_locations

def spectrum_outliers(spectra, n = 5):
    cut_s = np.copy(spectra)
    for i in range(len(spectra)):
        s = residual_outliers_1d(spectra[i][1], n = n)
        for j in s[0]:
            if j + 1 in s[0]:
                cut_s[i][1][j] = (spectra[i][1][j-1] + spectra[i][1][j-2])/2
            elif j == 1023:
                cut_s[i][1][j] = (spectra[i][1][j-1] + spectra[i][1][j-2])/2
            else:
                cut_s[i][1][j] = (spectra[i][1][j-1] + spectra[i][1][j+1])/2
                
    return cut_s

# function to do a basic centroid trace fit to the data
def trace_spectrum(data, xi, xf, y_guess, profile_radius = 20, gauss_filter_width = 10):
    # x-axis
    x = np.arange(xi,xf)
    # y-axis
    y = np.arange(data.shape[0])
    
    # Define array that will save centroids at each x:
    y_vals = np.zeros(len(x))
    
    for i in range(len(x)):
        # Find centroid within profile_radius pixels of the initial guess:
        idx = np.where(np.abs(y-y_guess)<profile_radius)[0]
        y_vals[i] = np.nansum(y[idx]*data[:,x[i]][idx])/np.nansum(data[:,x[i]][idx])
        y_guess = y_vals[i]
    return x,y_vals


# function to mark bad pixels in files based on data quality (dq) frames included in the fits files
# default flagged pixel to use is 16, but can add a list of whichever you want to remove
def dq_clean(files, dqs, flags):
    
    # initializing list of bad pixels
    bads = []
    
    # in each of the included files, mark the location of the pixels with the value given in the flags kwarg
    for i in range(len(files)):
        for j in flags:
            bad = np.where(dqs[i] == j)
            bad_indices = list(zip(bad[0], bad[1]))
            bads.append(bad_indices)
    bads = np.array(bads)
    
    # make a copy of the original fed in files array shape
    files_clean = np.zeros_like(files)
    
    # make the value of each bad pixel -1, to be fit over later
    for j in range(len(files)):
        c_frame = files[j]
        for index in bads[j]:
            c_frame[index] = -1
        files_clean[j] = c_frame
    
    # return the files with the bad pixel locations set as -1
    return files_clean

# function to do "difference cleaning" between frames
def difference_clean(files, wind_size, sigma):
    
    # initializing lists
    differences = []
    labels = []
    
    # each image will be taken with a difference with the rest of the images in files
    # i know it's redundant to take differences both ways, but i think it makes sense for the median files
    for i in range(len(files)):
        standard = files[i]
        for j in range(len(files)):
            if j == i:
                continue
            else:
                diff = standard - files[j]
                differences.append(diff)
                label = [i,j]
                labels.append(label)
    
    medians = []
    
    # create median frames using all of the subtractions from each frame together
    for k in range(len(files)):
        sublist = []
        for n in range(len(labels)):
            if labels[n][0] == k:
                sublist.append(differences[n])
            else:
                continue
            
        median_frame = np.nanmedian(sublist, axis = 0)

        medians.append(median_frame)
    
    # in each median frame, in each row sigma reject in windows given by wind_size and sigma
    cr_loc = []
    cr_loc_frame = []
    files_clean = np.zeros_like(files)
    frame_num = 0
    for m in medians:
        frame = np.copy(files[frame_num])
        cr_loc_single = []
        cr_loc_frame_single = np.zeros_like(medians[0])
        for row_idx in range(len(m)):
            row = m[row_idx]
            row_med = np.nanmedian(row)
            row_stdev = np.sqrt(np.var(row))
            row_cut = row_med+(row_stdev*sigma)
            q = wind_size
            for p in range(len(row)-wind_size):
                window = row[p:q]
                wind_med = np.nanmedian(window)
                wind_stdev = np.sqrt(np.var(m))
                wind_cut = wind_med+(wind_stdev*sigma)
                cut_use = max(wind_cut, row_cut)
                for val in range(len(window)):
                    if window[val] > cut_use:
                        cr_loc_single.append([m,p+val])
                        cr_loc_frame_single[row_idx][p+val] = 1
                        frame[row_idx][p+val] = -2
                p = p+wind_size
                q = q+wind_size
            
        cr_loc.append(cr_loc_single)
        cr_loc_frame.append(cr_loc_frame_single)
        files_clean[frame_num] = frame 
        frame_num = frame_num + 1
    
    return files_clean


# function to do hot and cold pixel marking
def hc_clean(files, hc_sigma, hc_window_size):
    
    # initializing list
    hcs = []
    
    splines_hcs = np.zeros_like(files)
    files_clean = np.zeros_like(files)
    
    # hot and cold pixels
    for frame_idx in range(len(files)):
        frame = np.copy(files[frame_idx])
        for column in range(len(files[0][0])):
            test_spline = UnivariateSpline(np.arange(0,len(frame[:,column]),1), frame[:,column], s=100)
            splines_hcs[frame_idx][:,column] = (test_spline(np.arange(0,len(frame[:,column]),1)))

        # leave borders around the edges to not mess up the box, we don't really care about edges anyways
        for p in range(hc_window_size+1,len(splines_hcs[frame_idx])-hc_window_size-1,1):
            for q in range(hc_window_size+1,len(splines_hcs[frame_idx][0])-hc_window_size-1,1):
                box = splines_hcs[frame_idx][p-hc_window_size:p+hc_window_size, q-hc_window_size:q+hc_window_size]
                box_less = box[box != splines_hcs[frame_idx][p,q]]
                box_med = np.nanmedian(box_less)
                box_stdev = np.sqrt(np.var(box_less))
                
                if splines_hcs[frame_idx][p,q] < box_med - (box_stdev * hc_sigma) or splines_hcs[frame_idx][p,q] > box_med + (box_stdev * hc_sigma):
                    hcs.append([p,q])
                    frame[p][q] = -3
        
        files_clean[frame_idx] = frame
    
    return files_clean

# function to mark values offset from a spline fit
def spline_mark(files, traces, spline_sigma, inner_factor, outer_factor):

    files_clean = np.zeros_like(files)
    # take average of spline fits across frames
    splines = np.zeros_like(files[0])
    # for each column
    for column in range(len(files[0][0])):
        # create temporary list to hold the individual splines for that column
        spline_fits = []
        for i in files:
            test_spline = UnivariateSpline(np.arange(0,len(i[:,column]),1), i[:,column])
            spline_fits.append(test_spline(np.arange(0,len(i[:,column]),1)))

        # take the median of the splines from each frame
        med = np.nanmedian(spline_fits, axis = 0)

        # normalize the spline before appending
        norm_med = med/np.nanmax(med)
        splines[:,column] = norm_med
    
    # now, use the splines to go through each frame and reject problems
    crs = []
    spline_use = np.zeros_like(splines)
    for frame_idx in range(len(files)):
        frame = np.copy(files[frame_idx])
        # use the median spline to reject cosmic rays for each frame's column
        for k in range(2,len(frame[0])-2,1):
            spline_use_single = np.nanmedian(splines[:,k-2:k+2], axis = 1)
            spline_use[:,k] = spline_use_single*np.nanmax(frame[:,k])

            # scale the normalized median spline to the max of the frame before taking residual
            resid = frame[:,k] - (spline_use[:,k])
            resid_stdev = np.sqrt(np.var(resid))
            cutoff = resid_stdev * spline_sigma
            #j = 0
            for m in range(len(resid)):
                # if the residual value is greater than residual stdev * sigma, then mark it as a problem
                if m > traces[frame_idx][1][m]+4 or m < traces[frame_idx][1][m]-4:
                    if resid[m] > cutoff:
                        frame[m,k] = -4
                elif m > traces[frame_idx][1][m]+2 or m < traces[frame_idx][1][m]-2:
                    if resid[m] > cutoff * outer_factor:
                        frame[m,k] = -4
                else:
                    if resid[m] > cutoff * inner_factor:
                        frame[m,k] = -4
                
        files_clean[frame_idx] = frame
        
    return files_clean

# function that does the cleaning for the marked "bad" pixels
def marked_clean(files, manual_badcolumn, s):
    bad_columns = []
    for frame_idx in range(len(files)):
        frame = np.copy(files[frame_idx])
        bad_columns_inb = []
        if manual_badcolumn == True:
            bad_columns_inb = [49, 89, 179, 180, 348, 399, 400, 491, 541, 771, 836, 841, 885, 943]
        else:
            for column in range(2,len(frame[0])-2,1):
                bad_counter = 0
                for pixel in range(len(frame)):
                    if frame[pixel][column] == -1:
                        bad_counter = bad_counter + 1

                if bad_counter >= len(frame)/6:
                    bad_columns_inb.append(column)
        bad_columns.append(bad_columns_inb)
    
    # there are columns full of bad pixels - if an entire column has more than 50% pixels marked as bad (=-1)
    # then use the average of the two surrounding columns 
    files_clean = np.zeros_like(files)

    frame_clean_1 = []
    for frame_idx in range(len(files)):
        frame = np.copy(files[frame_idx])
        for column in range(2,len(frame[0])-2,1):
            
            for val in range(len(frame)):
                
                if frame[val][column] < 0:#in [-1,-2,-3,-4]:
                    
                    if frame[val][column+1] in [-1,-2,-3,-4]:
                        frame[val][column] = ((frame[val][column-2]+frame[val][column-1])/2)

                    elif frame[val][column-1] in [-1,-2,-3,-4]:
                        frame[val][column] = ((frame[val][column+1]+frame[val][column+2])/2)
                    
                    else:
                        frame[val][column] = ((frame[val][column+1]+frame[val][column-1])/2)

        frame_clean_1.append(frame)   
        
    files_clean = []
    for frame_idx in range(len(files)):
        frame = np.copy(frame_clean_1[frame_idx])
        for column in range(2,len(frame[0])-2,1):
            if column in bad_columns[frame_idx]:
                if column-1 in bad_columns[frame_idx]:
                    frame[:,column] = (frame[:,column-2]+frame[:,column+1])/2
                elif column+1 in bad_columns[frame_idx]:
                    frame[:,column] = (frame[:,column-1]+frame[:,column+2])/2
                else:
                    frame[:,column] = (frame[:,column-1]+frame[:,column+1])/2
                    
        files_clean.append(frame)
        
    return files_clean


# master cleaning function, calls all of the above single functions, all of which can be turned off
def clean_data(files, dq_correct = True, dqs = None,  flags = [16], difference_correct = True, wind_size = 20, wind_sigma = 5, hc_correct = True, hc_sigma = 3, hc_wind_size = 2, spline_correct = True, traces = None, spline_sigma = 3, s = 7e5, manual_badcolumn = False, inner_factor = 4, outer_factor = 2, return_marked = False):
    
    if dq_correct == True:
        print("Starting dq_correct.")
        if dqs is None:
            print("Oops! You need the data quality frames corresponding to each exposure for the dq_correct." +                   "You can get these from the get_data function with dq = True")
            return
        else:
            marked_1 = dq_clean(files, dqs, flags)
            print("dq_correct complete.")
    else:
        marked_1 = np.copy(files)
        
    if difference_correct == True:
        print("Starting difference_correct.")
        marked_2 = difference_clean(marked_1, wind_size, wind_sigma)
        print("difference_correct complete.")
    else: 
        marked_2 = np.copy(marked_1)
    
    if hc_correct == True:
        print("Starting hc_correct.")
        marked_3 = hc_clean(marked_2, hc_sigma, hc_wind_size)
        print("hc_correct complete.")
    else:
        marked_3 = np.copy(marked_2)
        
    if spline_correct == True:
        if traces is None:
            print("Oops! You need basic spectral traces for the spline fit option :]")
            return
        else:
            print("Starting spline_correct.")
            marked_4 = spline_mark(marked_3, traces, spline_sigma, inner_factor, outer_factor)
            print("spline_correct complete.")
            print("Starting overwrite of marked pixels.")
            cleaned_data = marked_clean(marked_4, manual_badcolumn, s)
            print("Done!")
    else:
        marked_4 = np.copy(marked_3)
        print("Starting overwrite of marked pixels.")
        cleaned_data = marked_clean(marked_4, manual_badcolumn, s)
        print("Done!")
    
    if return_marked == True:
        return cleaned_data, marked_4
    
    else:
        return cleaned_data


# converting header times to bjd
def times_to_bjd(headers, starname = "WASP-69"):
    times = []
    exptimes = []
    expstart = []
    expend = []
    for i in headers:

        times.append(i["DATE-OBS"]+"T"+i["TIME-OBS"])

        exptimes.append(i["EXPTIME"])
        expstart.append(i["EXPSTART"])
        expend.append(i["EXPEND"])

    # change this time to bjd using package barycorr

    jd_conv = 2400000.5
    t_start = Time(np.array(expstart)+jd_conv, format='jd', scale='utc')
    t_end = Time(np.array(expend)+jd_conv, format='jd', scale='utc')

    t_start_bjd = utc_tdb.JDUTC_to_BJDTDB(t_start,starname = starname)#hip_id=8102 , lat=-30.169283, longi=-70.806789, alt=2241.9)
    t_end_bjd = utc_tdb.JDUTC_to_BJDTDB(t_end,starname = starname)# , lat=-30.169283, longi=-70.806789, alt=2241.9)

    return t_start_bjd, t_end_bjd

# spectral extraction, with options for optimal or simple box
def spectral_extraction(data, trace, method = "optimal", correct_bkg = False, aperture_radius = 15., ron = 1., gain = 1.,                         nsigma = 12, polynomial_spacing = 0.75, polynomial_order = 3, errors = None, background_radius=30):
    
    if errors is not None:
        if method == "optimal":
            spectrum = spectroscopy.getOptimalSpectrum(data, trace[1], aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order, data_variance = np.array(errors)**2)#, min_column = 600)
        elif method == "simple":
            spectrum = spectroscopy.getSimpleSpectrum(data, trace[0], trace[1], aperture_radius, background_radius = background_radius, correct_bkg = correct_bkg, error_data = errors)#, min_column = 600)
        
        
    else:
        if method == "optimal":
            spectrum = spectroscopy.getOptimalSpectrum(data, trace[1], aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order)#, min_column = 600)
        elif method == "simple": 
            spectrum = spectroscopy.getSimpleSpectrum(data, trace[0], trace[1], aperture_radius, background_radius = background_radius, correct_bkg = correct_bkg)#, min_column = 600)
        

    return spectrum

# this was for finding median p values for optimal extraction and fixing them, but it didn't work that great
# but I'll leave it as an option
def spectral_extraction_bulk(data_lst, trace_lst, method = "optimal", correct_bkg = False, aperture_radius = 15., ron = 1., gain = 1., nsigma = 12, polynomial_spacing = 0.75, polynomial_order = 3, errors = None, median_p = True):
    spectra = []
    
    if median_p == True:
        p_lst = []
        for i in range(len(data_lst)):
            data = data_lst[i]
            trace = trace_lst[i]
            
            if errors is not None:
                spectrum, p = spectroscopy.getOptimalSpectrum(data, trace[1], aperture_radius, ron, gain, 3, polynomial_spacing, polynomial_order, data_variance = np.array(errors)**2, return_P = True)
            
            else:
                spectrum, p = spectroscopy.getOptimalSpectrum(data, trace[1], aperture_radius, ron, gain, 3, polynomial_spacing, polynomial_order, return_P = True)
                
            p_lst.append(np.array(p))
        
        med_p = np.nanmedian(p_lst, axis = 0)    
        
        for i in range(len(data_lst)):
            data = data_lst[i]
            trace = trace_lst[i]
            
            if errors is not None:
                spectrum = spectroscopy.getOptimalSpectrum(data, trace[1], aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order, data_variance = np.array(errors)**2, P = med_p)
            
            else:
                spectrum = spectroscopy.getOptimalSpectrum(data, trace[1], aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order, P = med_p)
                
            spectra.append(spectrum)
            
        

    else:
        for i in range(len(data_lst)):
            data = data_lst[i]
            trace = trace_lst[i]

            if errors is not None:
                if method == "optimal":                    
                    spectrum = spectroscopy.getOptimalSpectrum(data, trace[1], aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order, data_variance = np.array(errors)**2)#, min_column = 600)
                elif method == "simple":
                    spectrum = spectroscopy.getSimpleSpectrum(data, trace[0], trace[1], aperture_radius, correct_bkg = correct_bkg, error_data = errors)#, min_column = 600)



            else:
                if method == "optimal":
                    spectrum = spectroscopy.getOptimalSpectrum(data, trace[1], aperture_radius, ron, gain, nsigma, polynomial_spacing, polynomial_order)#, min_column = 600)
                elif method == "simple": 
                    spectrum = spectroscopy.getSimpleSpectrum(data, trace[0], trace[1], aperture_radius, correct_bkg = correct_bkg)#, min_column = 600)

            spectra.append(spectrum)
            
    return spectra     

# cross-correlation
def xcorr(x,y):
    """
    Perform Cross-Correlation on x and y Deviations
    x    : 1st signal
    y    : 2nd signal

    returns
    lags : lags of correlation
    corr : coefficients of correlation
    maxlag :  Lag at which cross-correlation function is maximized
    """
    corr = signal.correlate(x-np.mean(x), y-np.mean(y), mode="full")
    scx=np.sum((y-np.mean(y))**2)
    scy=np.sum((x-np.mean(x))**2)
    corr = corr/(np.sqrt(scx*scy))
    lags = signal.correlation_lags(len(x), len(y), mode="full")
    mindx=np.argmax(corr) #index of cross corr peak
    
    srad = 3 #Find peak shift from central region srad pixels wide on each side
    sublag=lags[mindx-srad:mindx+(srad+1)]
    subcf=corr[mindx-srad:mindx+(srad+1)]
    result=np.polyfit(sublag, subcf, 2)
    maxlag = - result[1]/(2.*result[0])
    return lags, corr,maxlag


def impact_param(i, a_Rs):
    return a_Rs*np.cos(np.deg2rad(i))

def inclination(b, a_Rs):
    return np.rad2deg(np.arccos(b/a_Rs))

# master single light curve fitting function
def white_light_fit(input_params, times, lc, errors, detrenders, sys_method = "linear", limb_darkening = "fixed", gp_kernel = "Matern", N_iters = 3, juliet_name = None, instrument_name = "STIS", sampler = "dynamic_dynesty", gp_priors = "exponential"):
    if os.path.exists("juliet_fits") != True:
        os.mkdir("juliet_fits")
    if sys_method == "linear":
        if sampler == "LM":
            p = lmfit.Parameters()
            fit_param = {} # dictionary to save our fit params
            fit_uncs = {} # dictionary to save our fit uncertainties
            params = batman.TransitParams()
            N_iters = N_iters

            # initializing lmfit parameters
            # vary = 0 for fixed, 1 for floating
            # keeping bounds fairly wide is important. central guess not so much

            # orbital params
            for i in input_params.keys():
                if i in ["c1", "c2"]:
                    continue
                else:
                    if len(input_params[i]) == 1:
                        p.add(i, value = input_params[i][0], vary = 0)
                    elif len(input_params[i]) == 2:
                        p.add(i, value = input_params[i][0], vary = 1, min = input_params[i][0] - input_params[i][1], max = input_params[i][0] + input_params[i][1])

            if limb_darkening == "fixed":
                params.limb_dark = "quadratic"
                p.add('c1', value = input_params["c1"][0], vary = 0) 
                p.add('c2', value = input_params["c1"][0], vary = 0)
                params.u = [p['c1'], p['c2']]


            # systematics params
            for j in detrenders.keys():
                p.add(j, value = 0, vary = 1)

            # linear systematics term
            #p.add("linear", value = 0, vary = 1)

            # baseline flux level
            p.add('f0', value = lc[0], vary = 1)


            # we iterate N times, fitting the data, and estimating errorbars. we re-fit from the best-fit and error of the previous fit
            # this is to ensure that we never get stuck in some tiny global minimum, and explore the space better
            # this also gives better uncertainties
            err = None
            for _ in range(N_iters-1):
                result = lmfit.minimize(residual, params = p, args = (times, params, lc, err, detrenders)) # fit data
                err = np.std(lc - model_light_curve(p, times, params, detrenders)[0])
                for name, param in result.params.items(): # iterate through our lmfit parameters and update the variables
                    p[name].value = param.value

            # one final fit to be sure. these Nth fits are fast.
            err = np.std(lc - model_light_curve(p, times, params, detrenders)[0])
            result = lmfit.minimize(residual, params = p, args = (times, params, lc, err, detrenders))

            # iterate through our two dictionaries to save fits and fit uncertainties
            for name, param in result.params.items():
                if param.vary == 1:
                    p[name].value = param.value
                    fit_param[name] = param.value
                    fit_uncs[name+'_unc'] = param.stderr

            t_final = np.linspace(times[0], times[-1], 1000)        
            model_fit = model_light_curve(p, times, params, detrenders) # stores the final best fitting model
            model_final = transit_final(p, t_final, params, detrenders)
            data_fit_cube = [fit_param, fit_uncs, model_final, lc] # stores all our stuff in a neat cube
            return data_fit_cube
        
        else:
            # Create dictionaries:
            time, fluxes, fluxes_error, sys = {},{},{},{}
            # Save data into those dictionaries:
            time[instrument_name], fluxes[instrument_name], fluxes_error[instrument_name] = times, lc/np.median(lc[:10]), \
            errors/np.median(lc[:10])
            
            sys[instrument_name]= np.vstack(detrenders.values()).T

            input_param_names = []
            input_dist = []
            hyperps_vals = []
            for i in input_params.keys():
                if i in ["c1", "c2"]:
                    continue
                elif i == "rho":
                    name = i
                    input_param_names.append(name)

                    if len(input_params[i]) == 1:
                        input_dist.append("fixed")
                        hyperps_vals.append(input_params[i][0])

                    elif len(input_params[i]) == 2:
                        input_dist.append("normal")
                        hyperps_vals.append(input_params[i])
                else:
                    name = i + "_p1"
                    input_param_names.append(name)

                    if len(input_params[i]) == 1:
                        input_dist.append("fixed")
                        hyperps_vals.append(input_params[i][0])

                    elif len(input_params[i]) == 2:
                        input_dist.append("normal")
                        hyperps_vals.append(input_params[i])
            
            input_param_names_sys = []
            input_dist_sys = []
            hyperps_vals_sys = []
            for i in range(len(detrenders.keys())):
                name = "theta" + str(i) + "_" + instrument_name
                input_param_names_sys.append(name)
                input_dist_sys.append("uniform")
                hyperps_vals_sys.append([-100, 100])

            if limb_darkening == "fixed":
                ld_names = ["q1_" + instrument_name, "q2_" + instrument_name]
                ld_dist = ['fixed','fixed']
                q1 = (input_params["c1"][0] + input_params["c2"][0])**2 
                q2 = input_params["c1"][0]/(2*(input_params["c1"][0]+input_params["c2"][0]))
                ld_hyperps = [q1, q2]
            elif limb_darkening == "fit":
                ld_names = ["q1_" + instrument_name, "q2_" + instrument_name]
                ld_dist = ['uniform','uniform']
                ld_hyperps = [[0,1], [0,1]]

                
            other_params = ["mdilution_" + instrument_name, "mflux_" + instrument_name, "sigma_w_" + instrument_name]

            other_params_dist = ['fixed', 'normal', 'loguniform']

            other_params_hyperps = [1.0, [0.,1.0], [0.1, 10000.]]    
            # if you don't want to define a prior on rho for some reason, you can uncomment this section
            #if "rho" in input_params.keys():
            #    
            #    other_params = ["mdilution_" + instrument_name, "mflux_" + instrument_name, "sigma_w_" + instrument_name]

            #    other_params_dist = ['fixed', 'normal', 'loguniform']

            #    other_params_hyperps = [1.0, [0.,1.0], [0.1, 10000.]]

            #else:
            #    other_params = ["rho", "mdilution_" + instrument_name, "mflux_" + instrument_name, "sigma_w_" + instrument_name]

            #    other_params_dist = ['loguniform', 'fixed', 'normal', 'loguniform']

            #    other_params_hyperps = [[100, 10000.], 1.0, [0.,1.0], [0.1, 10000.]]

            params = input_param_names + ld_names + other_params + input_param_names_sys

            dists = input_dist + ld_dist + other_params_dist + input_dist_sys

            hyperps = hyperps_vals + ld_hyperps + other_params_hyperps + hyperps_vals_sys
            
            priors = {}
            for param, dist, hyperp in zip(params, dists, hyperps):
                priors[param] = {}
                priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

            # Perform the juliet fit. Load dataset first:
            if juliet_name is None:
                print("Oops! you need to enter a name for the output folder for your GP fit")
            else:
                dataset = juliet.load(priors=priors, t_lc = time, y_lc = fluxes, \
                                  yerr_lc = fluxes_error, linear_regressors_lc = sys,
                                  out_folder = "juliet_fits/" + juliet_name)

            if sampler == "dynesty":
                results = dataset.fit(sampler = "dynesty", nthreads = 2, verbose = True)
            elif sampler == "dynamic_dynesty":
                results = dataset.fit(sampler = "dynamic_dynesty", nthreads = 2, verbose = True)
            else: 
                results = dataset.fit(n_live_points = 500, verbose = True)
            return results
            
    
    elif sys_method == "gp":
        # Create dictionaries:
        time, fluxes, fluxes_error, sys = {},{},{},{}
        # Save data into those dictionaries:
        time[instrument_name], fluxes[instrument_name], fluxes_error[instrument_name] = times, lc/np.median(lc[:10]), \
        errors/np.median(lc[:10])

        sys[instrument_name]= np.vstack(detrenders.values()).T

        input_param_names = []
        input_dist = []
        hyperps_vals = []
        for i in input_params.keys():
            if i in ["c1", "c2"]:
                continue
            elif i == "rho":
                name = i
                input_param_names.append(name)

                if len(input_params[i]) == 1:
                    input_dist.append("fixed")
                    hyperps_vals.append(input_params[i][0])

                elif len(input_params[i]) == 2:
                    input_dist.append("normal")
                    hyperps_vals.append(input_params[i])
            else:
                name = i + "_p1"
                input_param_names.append(name)

                if len(input_params[i]) == 1:
                    input_dist.append("fixed")
                    hyperps_vals.append(input_params[i][0])

                elif len(input_params[i]) == 2:
                    input_dist.append("normal")
                    hyperps_vals.append(input_params[i])
        input_param_names_sys = []
        input_dist_sys = []
        hyperps_vals_sys = []
        for i in range(len(detrenders.keys())):
            if gp_kernel == "ExpSquared":
                name = "GP_alpha" + str(i) + "_" + instrument_name
            elif gp_kernel == "Matern":
                name = "GP_malpha" + str(i) + "_" + instrument_name
            
            if gp_priors == "exponential":
                input_param_names_sys.append(name)
                input_dist_sys.append("exponential")
                hyperps_vals_sys.append([1])
            
            elif gp_priors == "loguniform":
                input_param_names_sys.append(name)
                input_dist_sys.append("loguniform")
                hyperps_vals_sys.append([1e-5, 1e5])
        
        if limb_darkening == "fixed":
            ld_names = ["q1_" + instrument_name, "q2_" + instrument_name]
            ld_dist = ['fixed','fixed']
            q1 = (input_params["c1"][0] + input_params["c2"][0])**2 
            q2 = input_params["c1"][0]/(2*(input_params["c1"][0]+input_params["c2"][0]))
            ld_hyperps = [q1, q2]
        elif limb_darkening == "fit":
            ld_names = ["q1_" + instrument_name, "q2_" + instrument_name]
            ld_dist = ['uniform','uniform']
            ld_hyperps = [[0,1], [0,1]]

            
            
        other_params = ["mdilution_" + instrument_name, "mflux_" + instrument_name, "sigma_w_" + instrument_name, \
                       "GP_sigma_" + instrument_name]

        other_params_dist = ['fixed', 'normal', 'loguniform', 'loguniform']

        other_params_hyperps = [1.0, [0.,1.0], [0.1, 10000.], [1e-6, 1e6]]
        # if you don't want to define a prior on rho for some reason, you can uncomment this section
        #if "rho" in input_params.keys():
                
        #    other_params = ["mdilution_" + instrument_name, "mflux_" + instrument_name, "sigma_w_" + instrument_name, \
        #               "GP_sigma_" + instrument_name]

        #    other_params_dist = ['fixed', 'normal', 'loguniform', 'loguniform']

        #    other_params_hyperps = [1.0, [0.,1.0], [0.1, 10000.], [1e-6, 1e6]]
        
        #else:
        #    other_params = ["rho", "mdilution_" + instrument_name, "mflux_" + instrument_name, "sigma_w_" + instrument_name, \
        #               "GP_sigma_" + instrument_name]

        #    other_params_dist = ['loguniform', 'fixed', 'normal', 'loguniform', 'loguniform']

        #    other_params_hyperps = [[100, 10000.], 1.0, [0.,1.0], [0.1, 10000.], [1e-6, 1e6]]


        params = input_param_names + ld_names + other_params + input_param_names_sys

        dists = input_dist + ld_dist + other_params_dist + input_dist_sys

        hyperps = hyperps_vals + ld_hyperps + other_params_hyperps + hyperps_vals_sys

        priors = {}
        for param, dist, hyperp in zip(params, dists, hyperps):
            priors[param] = {}
            priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

        # Perform the juliet fit. Load dataset first:
        if juliet_name is None:
            print("Oops! you need to enter a name for the output folder for your GP fit")
        else:
            dataset = juliet.load(priors=priors, t_lc = time, y_lc = fluxes, \
                              yerr_lc = fluxes_error, GP_regressors_lc = sys,
                              out_folder = "juliet_fits/" + juliet_name)

        if sampler == "dynesty":
            results = dataset.fit(sampler = "dynesty", nthreads = 2, verbose = True)
        elif sampler == "dynamic_dynesty":
            results = dataset.fit(sampler = "dynamic_dynesty", nthreads = 2, verbose = True)
        else:
            results = dataset.fit(n_live_points = 500, verbose = True)
        return results
        
    
# master joint light curve fitting function   
def joint_white_light_fit(input_params, times1, lc1, errors1, detrenders1, times2, lc2, errors2, detrenders2, sys_method = "linear", limb_darkening = "fixed", gp_kernel = "Matern", N_iters = 3, juliet_name = None, instrument_name1 = "STIS1", instrument_name2 = "STIS2", sampler = "dynamic_dynesty", gp_priors = "exponential"):
    if os.path.exists("juliet_fits") != True:
        os.mkdir("juliet_fits")
    if sys_method == "linear":
        if sampler == "LM":
            # systematics will act on individual light curves, which will then be phased into one light curve for the transit fit
            p = lmfit.Parameters()
            fit_param = {} # dictionary to save our fit params
            fit_uncs = {} # dictionary to save our fit uncertainties
            params = batman.TransitParams()
            N_iters = N_iters

            # initializing lmfit parameters
            # vary = 0 for fixed, 1 for floating
            # keeping bounds fairly wide is important. central guess not so much

            # orbital params
            for i in input_params.keys():
                if i in ["c1", "c2"]:
                    continue
                else:
                    if len(input_params[i]) == 1:
                        p.add(i, value = input_params[i][0], vary = 0)
                    elif len(input_params[i]) == 2:
                        p.add(i, value = input_params[i][0], vary = 1, min = input_params[i][0] - input_params[i][1], max = input_params[i][0] + input_params[i][1])

            if limb_darkening == "fixed":
                params.limb_dark = "quadratic"
                p.add('c1', value = input_params["c1"][0], vary = 0) 
                p.add('c2', value = input_params["c1"][0], vary = 0)
                params.u = [p['c1'], p['c2']]


            # systematics params
            for j in detrenders1.keys():
                p.add(j+"_"+str(1), value = 0, vary = 1)

            for j in detrenders2.keys():
                p.add(j+"_"+str(2), value = 0, vary = 1)

            # linear systematics term
            #p.add("linear", value = 0, vary = 1)

            # baseline flux level
            p.add('f0', value = lc1[0], vary = 1)


            # we iterate N times, fitting the data, and estimating errorbars. we re-fit from the best-fit and error of the previous fit
            # this is to ensure that we never get stuck in some tiny global minimum, and explore the space better
            # this also gives better uncertainties
            err = None
            for _ in range(N_iters-1):
                result = lmfit.minimize(residual_joint, params = p, args = (times1, times2, params, lc1, lc2, err, detrenders1, detrenders2)) # fit data
                err = np.std(lc1 - model_light_curve_joint(p, times1, times2, params, detrenders1, detrenders2)[0][0]) + np.std(lc2 - model_light_curve_joint(p, times1, times2, params, detrenders1, detrenders2)[0][1])
                for name, param in result.params.items(): # iterate through our lmfit parameters and update the variables
                    p[name].value = param.value

            # one final fit to be sure. these Nth fits are fast.
            err = np.std(lc1 - model_light_curve_joint(p, times1, times2, params, detrenders1, detrenders2)[0][0]) + np.std(lc2 - model_light_curve_joint(p, times1, times2, params, detrenders1, detrenders2)[0][1])
            result = lmfit.minimize(residual_joint, params = p, args = (times1, times2, params, lc1, lc2, err, detrenders1, detrenders2)) 

            # iterate through our two dictionaries to save fits and fit uncertainties
            for name, param in result.params.items():
                if param.vary == 1:
                    p[name].value = param.value
                    fit_param[name] = param.value
                    fit_uncs[name+'_unc'] = param.stderr

            t_final = np.linspace(times1[0], times1[-1], 1000)        
            model_fit = model_light_curve_joint(p, times1, times2, params, detrenders1, detrenders2) # stores the final best fitting model
            model_final = transit_final_joint(p, t_final, params, detrenders1, detrenders2)
            data_fit_cube = [fit_param, fit_uncs, model_final, [lc1, lc2]] # stores all our stuff in a neat cube
            return data_fit_cube
        
        else:
            # Create dictionaries:
            time, fluxes, fluxes_error, sys = {},{},{},{}
            # Save data into those dictionaries:
            time[instrument_name1], fluxes[instrument_name1], fluxes_error[instrument_name1] = times1, lc1/np.median(lc1[:10]), \
            errors1/np.median(lc1[:10])
            time[instrument_name2], fluxes[instrument_name2], fluxes_error[instrument_name2] = times2, lc2/np.median(lc2[:10]), \
            errors2/np.median(lc2[:10])
            
            sys[instrument_name1]= np.vstack(detrenders1.values()).T
            sys[instrument_name2]= np.vstack(detrenders2.values()).T

            input_param_names = []
            input_dist = []
            hyperps_vals = []
            for i in input_params.keys():
                if i in ["c1", "c2"]:
                    continue
                elif i == "rho":
                    name = i
                    input_param_names.append(name)

                    if len(input_params[i]) == 1:
                        input_dist.append("fixed")
                        hyperps_vals.append(input_params[i][0])

                    elif len(input_params[i]) == 2:
                        input_dist.append("normal")
                        hyperps_vals.append(input_params[i])
                else:
                    name = i + "_p1"
                    input_param_names.append(name)

                    if len(input_params[i]) == 1:
                        input_dist.append("fixed")
                        hyperps_vals.append(input_params[i][0])

                    elif len(input_params[i]) == 2:
                        input_dist.append("normal")
                        hyperps_vals.append(input_params[i])
            
            input_param_names_sys1 = []
            input_dist_sys1 = []
            hyperps_vals_sys1 = []
            
            for i in range(len(detrenders1.keys())):
                name = "theta" + str(i) + "_" + instrument_name1
                input_param_names_sys1.append(name)
                input_dist_sys1.append("uniform")
                hyperps_vals_sys1.append([-100, 100])
                    
                
            input_param_names_sys2 = []
            input_dist_sys2 = []
            hyperps_vals_sys2 = []
            
            for i in range(len(detrenders2.keys())):
                name = "theta" + str(i) + "_" + instrument_name2
                input_param_names_sys2.append(name)
                input_dist_sys2.append("uniform")
                hyperps_vals_sys2.append([-100, 100])


            if limb_darkening == "fixed":
                ld_names = ["q1_" + instrument_name1 + "_" + instrument_name2, "q2_" + instrument_name1 + "_" + instrument_name2]
                ld_dist = ['fixed','fixed']
                q1 = (input_params["c1"][0] + input_params["c2"][0])**2 
                q2 = input_params["c1"][0]/(2*(input_params["c1"][0]+input_params["c2"][0]))
                ld_hyperps = [q1, q2]
            elif limb_darkening == "fit":
                ld_names = ["q1_" + instrument_name1 + "_" + instrument_name2, "q2_" + instrument_name1 + "_" + instrument_name2]
                ld_dist = ['uniform','uniform']
                ld_hyperps = [[0,1], [0,1]]
            
            
            other_params = ["mdilution_" + instrument_name1, "mflux_" + instrument_name1, "sigma_w_" + instrument_name1, "mdilution_" + instrument_name2, "mflux_" + instrument_name2, "sigma_w_" + instrument_name2]

            other_params_dist = ['fixed', 'normal', 'loguniform', 'fixed', 'normal', 'loguniform']

            other_params_hyperps = [1.0, [0.,1.0], [0.1, 10000.], 1.0, [0.,1.0], [0.1, 10000.]]
            # if you don't want to define a prior on rho for some reason, you can uncomment this section
            #if "rho" in input_params.keys():

            #    other_params = ["mdilution_" + instrument_name1, "mflux_" + instrument_name1, "sigma_w_" + instrument_name1, "mdilution_" + instrument_name2, "mflux_" + instrument_name2, "sigma_w_" + instrument_name2]

            #    other_params_dist = ['fixed', 'normal', 'loguniform', 'fixed', 'normal', 'loguniform']

            #    other_params_hyperps = [1.0, [0.,1.0], [0.1, 10000.], 1.0, [0.,1.0], [0.1, 10000.]]

            #else:

            #    other_params = ["rho", "mdilution_" + instrument_name1, "mflux_" + instrument_name1, "sigma_w_" + instrument_name1, "mdilution_" + instrument_name2, "mflux_" + instrument_name2, "sigma_w_" + instrument_name2]

            #    other_params_dist = ['loguniform', 'fixed', 'normal', 'loguniform', 'fixed', 'normal', 'loguniform']

             #   other_params_hyperps = [[100, 10000.], 1.0, [0.,1.0], [0.1, 10000.], 1.0, [0.,1.0], [0.1, 10000.]]

                
            params = input_param_names + ld_names + other_params + input_param_names_sys1 + input_param_names_sys2

            dists = input_dist + ld_dist + other_params_dist + input_dist_sys1 + input_dist_sys2

            hyperps = hyperps_vals + ld_hyperps + other_params_hyperps + hyperps_vals_sys1 + hyperps_vals_sys2
            
            priors = {}
            for param, dist, hyperp in zip(params, dists, hyperps):
                priors[param] = {}
                priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

            # Perform the juliet fit. Load dataset first:
            if juliet_name is None:
                print("Oops! you need to enter a name for the output folder for your GP fit")
            else:
                dataset = juliet.load(priors=priors, t_lc = time, y_lc = fluxes, \
                                  yerr_lc = fluxes_error, linear_regressors_lc = sys,
                                  out_folder = "juliet_fits/" + juliet_name)
                
            if sampler == "dynesty":
                results = dataset.fit(sampler = "dynesty", nthreads = 2, verbose = True)
            elif sampler == "dynamic_dynesty":
                results = dataset.fit(sampler = "dynamic_dynesty", nthreads = 2, verbose = True)
            else: 
                results = dataset.fit(n_live_points = 500, verbose = True)
            return results
            
        
    
    elif sys_method == "gp":
            
        # Create dictionaries:
        time, fluxes, fluxes_error, sys = {},{},{},{}
        # Save data into those dictionaries:
        time[instrument_name1], fluxes[instrument_name1], fluxes_error[instrument_name1] = times1, lc1/np.median(lc1[:10]), \
        errors1/np.median(lc1[:10])
        time[instrument_name2], fluxes[instrument_name2], fluxes_error[instrument_name2] = times2, lc2/np.median(lc2[:10]), \
        errors2/np.median(lc2[:10])

        sys[instrument_name1]= np.vstack(detrenders1.values()).T
        sys[instrument_name2]= np.vstack(detrenders2.values()).T

        input_param_names = []
        input_dist = []
        hyperps_vals = []
        for i in input_params.keys():
            if i in ["c1", "c2"]:
                continue
            elif i == "rho":
                name = i
                input_param_names.append(name)

                if len(input_params[i]) == 1:
                    input_dist.append("fixed")
                    hyperps_vals.append(input_params[i][0])

                elif len(input_params[i]) == 2:
                    input_dist.append("normal")
                    hyperps_vals.append(input_params[i])
            else:
                name = i + "_p1"
                input_param_names.append(name)

                if len(input_params[i]) == 1:
                    input_dist.append("fixed")
                    hyperps_vals.append(input_params[i][0])

                elif len(input_params[i]) == 2:
                    input_dist.append("normal")
                    hyperps_vals.append(input_params[i])

        input_param_names_sys1 = []
        input_dist_sys1 = []
        hyperps_vals_sys1 = []
        for i in range(len(detrenders1.keys())):
            if gp_kernel == "ExpSquared":
                name = "GP_alpha" + str(i) + "_" + instrument_name1
            elif gp_kernel == "Matern":
                name = "GP_malpha" + str(i) + "_" + instrument_name1

            if gp_priors == "exponential":
                input_param_names_sys1.append(name)
                input_dist_sys1.append("exponential")
                hyperps_vals_sys1.append([1])

            elif gp_priors == "loguniform":
                input_param_names_sys1.append(name)
                input_dist_sys1.append("loguniform")
                hyperps_vals_sys1.append([1e-5, 1e5])

        input_param_names_sys2 = []
        input_dist_sys2 = []
        hyperps_vals_sys2 = []
        for i in range(len(detrenders2.keys())):
            if gp_kernel == "ExpSquared":
                name = "GP_alpha" + str(i) + "_" + instrument_name2
            elif gp_kernel == "Matern":
                name = "GP_malpha" + str(i) + "_" + instrument_name2

            if gp_priors == "exponential":
                input_param_names_sys2.append(name)
                input_dist_sys2.append("exponential")
                hyperps_vals_sys2.append([1])

            elif gp_priors == "loguniform":
                input_param_names_sys2.append(name)
                input_dist_sys2.append("loguniform")
                hyperps_vals_sys2.append([1e-5, 1e5])

        if limb_darkening == "fixed":
            ld_names = ["q1_" + instrument_name1 + "_" + instrument_name2, "q2_" + instrument_name1 + "_" + instrument_name2]
            ld_dist = ['fixed','fixed']
            q1 = (input_params["c1"][0] + input_params["c2"][0])**2 
            q2 = input_params["c1"][0]/(2*(input_params["c1"][0]+input_params["c2"][0]))
            ld_hyperps = [q1, q2]
        elif limb_darkening == "fit":
            ld_names = ["q1_" + instrument_name1 + "_" + instrument_name2, "q2_" + instrument_name1 + "_" + instrument_name2]
            ld_dist = ['uniform','uniform']
            ld_hyperps = [[0,1], [0,1]]

        other_params = ["mdilution_" + instrument_name1, "mflux_" + instrument_name1, "sigma_w_" + instrument_name1, "GP_sigma_" + instrument_name1, "mdilution_" + instrument_name2, "mflux_" + instrument_name2, "sigma_w_" + instrument_name2, "GP_sigma_" + instrument_name2]

        other_params_dist = ['fixed', 'normal', 'loguniform', 'loguniform', 'fixed', 'normal', 'loguniform', 'loguniform']

        other_params_hyperps = [1.0, [0.,1.0], [0.1, 10000.], [1e-6, 1e6], 1.0, [0.,1.0], [0.1, 10000.], [1e-6, 1e6]]
        
        # if you don't want to define a prior on rho for some reason, you can uncomment this section
        #if "rho" in input_params.keys():

        #    other_params = ["mdilution_" + instrument_name1, "mflux_" + instrument_name1, "sigma_w_" + instrument_name1, "GP_sigma_" + instrument_name1, "mdilution_" + instrument_name2, "mflux_" + instrument_name2, "sigma_w_" + instrument_name2, "GP_sigma_" + instrument_name2]

        #    other_params_dist = ['fixed', 'normal', 'loguniform', 'loguniform', 'fixed', 'normal', 'loguniform', 'loguniform']

        #    other_params_hyperps = [1.0, [0.,1.0], [0.1, 10000.], [1e-6, 1e6], 1.0, [0.,1.0], [0.1, 10000.], [1e-6, 1e6]]

        #else:

        #    other_params = ["rho", "mdilution_" + instrument_name1, "mflux_" + instrument_name1, "sigma_w_" + instrument_name1, "GP_sigma_" + instrument_name1, "mdilution_" + instrument_name2, "mflux_" + instrument_name2, "sigma_w_" + instrument_name2, "GP_sigma_" + instrument_name2]

        #    other_params_dist = ['loguniform', 'fixed', 'normal', 'loguniform', 'loguniform', 'fixed', 'normal', 'loguniform', 'loguniform']

        #    other_params_hyperps = [[100, 10000.], 1.0, [0.,1.0], [0.1, 10000.], [1e-6, 1e6], 1.0, [0.,1.0], [0.1, 10000.], [1e-6, 1e6]]


        params = input_param_names + ld_names + other_params + input_param_names_sys1 + input_param_names_sys2

        dists = input_dist + ld_dist + other_params_dist + input_dist_sys1 + input_dist_sys2

        hyperps = hyperps_vals + ld_hyperps + other_params_hyperps + hyperps_vals_sys1 + hyperps_vals_sys2


        priors = {}
        for param, dist, hyperp in zip(params, dists, hyperps):
            priors[param] = {}
            priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

        # Perform the juliet fit. Load dataset first:
        if juliet_name is None:
            print("Oops! you need to enter a name for the output folder for your GP fit")
        else:
            dataset = juliet.load(priors=priors, t_lc = time, y_lc = fluxes, \
                              yerr_lc = fluxes_error, GP_regressors_lc = sys,
                              out_folder = "juliet_fits/" + juliet_name)

        # Fit:
        if sampler == "dynesty":
            results = dataset.fit(sampler = "dynesty", nthreads = 2, verbose = True)
        elif sampler == "dynamic_dynesty":
            results = dataset.fit(sampler = "dynamic_dynesty", nthreads = 2, verbose = True)
        else:
            results = dataset.fit(n_live_points = 500, verbose = True)
        return results

# master single spectroscopic light curve fitting function
def spectroscopic_lightcurve_fit(params, wl, times, spectra, detrenders, bins, sld, bin_unit = "nm",\
                                 sys_method = "gp", juliet_name = None, mode = None, plot = False,\
                                 vertical_offset = 0.015, figure_offset = 0.015, savefig = False,\
                                 fig_name = None, method = None, sampler = "dynamic_dynesty"):
    ordered_binned_spectrum, ordered_binned_errors, bin_centers = binning(wl, spectra, bins, bin_unit = bin_unit)
    bin_lst = np.loadtxt(bins, delimiter = "\t")
    if bin_unit == "nm":
        bin_lst = bin_lst*10
        
    fits = []
    depths = []
    depth_e = []
    fit_value = []
    if os.path.exists("juliet_fits/" + juliet_name) != True:
        os.mkdir("juliet_fits/" + juliet_name)
    
    colormap = plt.cm.nipy_spectral
    colors = colormap(np.linspace(0.1, 0.9, len(ordered_binned_spectrum)))
        
    plt.figure(figsize = (8, 24))
    v_off = 0
    fig_off = 0
    for i in range(len(ordered_binned_spectrum)):
        if mode == None:
            print("For limb darkening, please pass mode = which HST mode you're using :)")
            break
        c1, c2 = sld.compute_quadratic_ld_coeffs(
        wavelength_range=np.array([bin_lst[i][0], bin_lst[i][1]]),
        mode=mode)
        params["c1"] = [c1]
        params["c2"] = [c2]

        if sys_method == "gp":
            name = juliet_name + "/" + juliet_name + "_bin" + str(i+1).zfill(3)
            wl_lc = white_light_fit(params, times, ordered_binned_spectrum[i], ordered_binned_errors[i], detrenders, sys_method = "gp", juliet_name=name, sampler = sampler)
            fits.append(wl_lc)
            fit_value.append(np.array([wl_lc.posteriors['lnZ'], wl_lc.posteriors['lnZerr']]))

                
        elif sys_method == "linear":
            if method == "LM":
                wl_lc = white_light_fit(params, times, ordered_binned_spectrum[i], detrenders, sys_method = "linear", method = "LM")
                fits.append(wl_lc)
            else:
                name = juliet_name + "/" + juliet_name + "_bin" + str(i+1).zfill(3)
                wl_lc = white_light_fit(params, times, ordered_binned_spectrum[i], ordered_binned_errors[i], detrenders, sys_method = "linear", juliet_name = name, sampler = sampler)
                fits.append(wl_lc)
                fit_value.append(np.array([wl_lc.posteriors['lnZ'], wl_lc.posteriors['lnZerr']]))

        if plot == True:
            t_final = np.linspace(times[0], times[-1], 1000)
            # then the gp detrending
            transit_plus_GP_model = wl_lc.lc.evaluate('STIS')
            transit_model = wl_lc.lc.evaluate('STIS', evaluate_transit = True)
            transit_model_resamp, error68_up, error68_down = wl_lc.lc.evaluate('STIS', evaluate_transit = True, t = t_final, return_err = True)

            # there may be a small vertical offset between the two models because of the different normalizations - account for that
            offset = np.nanmedian(wl_lc.posteriors['posterior_samples']['mflux_STIS'])
            plt.scatter(times, (ordered_binned_spectrum[i]/np.median(ordered_binned_spectrum[i][:10]))+v_off, label = "raw data", color = colors[i], alpha = 0.3, s = 15)
            plt.scatter(times, (ordered_binned_spectrum[i]/np.median(ordered_binned_spectrum[i][:10])) - (transit_plus_GP_model - transit_model) + offset+v_off,alpha=1, label = "gp detrending", color = colors[i], s = 15)
            #plt.plot(t_final, batman_model(t_final), color='black',zorder=10, label = "gp transit model (batman)")
            plt.plot(np.linspace(times[0], times[-1], 1000), transit_model_resamp + offset+v_off, color = colors[i],zorder=10, label = "gp + transit model")
            plt.fill_between(t_final, error68_up+offset+v_off, error68_down+offset+v_off, color = colors[i], alpha = 0.2, label = "68% error")

            txt = plt.text(times[0], 1.004 + fig_off ,str(bin_lst[i]/10000), size=11, color='black', weight = "bold")
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

            v_off -= vertical_offset
            fig_off -= figure_offset
        
    if plot == True:
        plt.ylabel("Relative Flux + Vertical Offset")
        plt.xlabel("Time (BJD-TBD)")
        plt.ylim(1.004 + fig_off - figure_offset, 1.004 + figure_offset)
        if savefig == True:
            plt.savefig(fig_name, dpi = 300, facecolor = "white", bbox_inches="tight")
        else:
            plt.show()  
    
    
    if sys_method == "gp":
        for file in sorted(glob.glob("juliet_fits/" + juliet_name + "/*bin*/*posteriors.dat*")):
            fit = np.genfromtxt(file, dtype=None)
            for param in fit:
                if param[0].decode() == "p_p1":
                    depths.append(param[1])
                    depth_e.append((param[2], param[3]))
        return(bin_centers, depths, depth_e, fit_value)
    
    elif sys_method == "linear":
        if method == "LM":
            print("not implemented :]")
        else:
            for file in sorted(glob.glob("juliet_fits/" + juliet_name + "/*bin*/*posteriors.dat*")):
                fit = np.genfromtxt(file, dtype=None)
                for param in fit:
                    if param[0].decode() == "p_p1":
                        depths.append(param[1])
                        depth_e.append((param[2], param[3]))
            return(bin_centers, depths, depth_e, fit_value)

        
# master joint spectroscopic light curve fitting function
def joint_spectroscopic_lightcurve_fit(params, wl, times1, spectra1, detrenders1, times2, spectra2, detrenders2,\
                                       bins, sld, bin_unit = "nm", sys_method = "gp", juliet_name = None,\
                                       mode = None, plot = False, vertical_offset = 0.015, figure_offset = 0.015,\
                                       savefig = False, fig_name = None, sampler = "dynamic_dynesty"):
    
    ordered_binned_spectrum1, ordered_binned_errors1, bin_centers = binning(wl, spectra1, bins, bin_unit = bin_unit)
    ordered_binned_spectrum2, ordered_binned_errors2, bin_centers = binning(wl, spectra2, bins, bin_unit = bin_unit)
    
    bin_lst = np.loadtxt(bins, delimiter = "\t")
    if bin_unit == "nm":
        bin_lst = bin_lst*10
        
    fits = []
    depths = []
    depth_e = []
    fit_value = []
    if os.path.exists("juliet_fits/" + juliet_name) != True:
        os.mkdir("juliet_fits/" + juliet_name)
    
    colormap = plt.cm.nipy_spectral
    colors = colormap(np.linspace(0.1, 0.9, len(ordered_binned_spectrum1)))
        
    plt.figure(figsize = (8, 24))
    v_off = 0
    fig_off = 0
        
    for i in range(len(ordered_binned_spectrum1)):
        if mode == None:
            print("For limb darkening, please pass mode = which HST mode you're using :)")
            break
        c1, c2 = sld.compute_quadratic_ld_coeffs(
        wavelength_range=np.array([bin_lst[i][0], bin_lst[i][1]]),
        mode=mode)
        params["c1"] = [c1]
        params["c2"] = [c2]
        if sys_method == "gp":
            name = juliet_name + "/" + juliet_name + "_bin" + str(i+1).zfill(3)
            wl_lc = joint_white_light_fit(params, times1, ordered_binned_spectrum1[i], ordered_binned_errors1[i], detrenders1, times2, ordered_binned_spectrum2[i], ordered_binned_errors2[i], detrenders2, sys_method = "gp", juliet_name=name, limb_darkening = "fixed", sampler = sampler)
            fits.append(wl_lc)
            fit_value.append(np.array([wl_lc.posteriors['lnZ'], wl_lc.posteriors['lnZerr']]))
                
        elif sys_method == "linear":
            name = juliet_name + "/" + juliet_name + "_bin" + str(i+1).zfill(3)
            wl_lc = joint_white_light_fit(params, times1, ordered_binned_spectrum1[i], ordered_binned_errors1[i], detrenders1, times2, ordered_binned_spectrum2[i], ordered_binned_errors2[i], detrenders2, sys_method = "linear",juliet_name=name, sampler = sampler)
            fits.append(wl_lc)
            fit_value.append(np.array([wl_lc.posteriors['lnZ'], wl_lc.posteriors['lnZerr']]))

        if plot == True:
            phases = juliet.utils.get_phases(times1, params["P"], params["t0"])
            phases_2 = juliet.utils.get_phases(times2, params["P"], params["t0"])

            transit_plus_GP_model1 = wl_lc.lc.evaluate('STIS1')
            transit_model1 = wl_lc.lc.evaluate('STIS1', evaluate_transit = True)

            gp_model1 = transit_plus_GP_model1 - transit_model1

            transit_plus_GP_model2 = wl_lc.lc.evaluate('STIS2')
            transit_model2 = wl_lc.lc.evaluate('STIS2', evaluate_transit = True)

            gp_model2 = transit_plus_GP_model2 - transit_model2

            offset1 = np.nanmedian(wl_lc.posteriors['posterior_samples']['mflux_STIS1'])
            offset2 = np.nanmedian(wl_lc.posteriors['posterior_samples']['mflux_STIS2'])
            if times1[0]%params["P"] < times2[0]%params["P"]:
                t_diff = ((times2[-1] - times1[-1])%params["P"])/params["P"]
            else:
                t_diff = ((times1[-1] - times2[-1])%params["P"])/params["P"]

            t_final = np.linspace(times1[0], (times1[-1]+t_diff)[0], 1000)
            transit_model2_resamp, error68_up, error68_down = wl_lc.lc.evaluate('STIS2', evaluate_transit = True, t = t_final, return_err = True)
            model_phases = juliet.utils.get_phases(t_final, params["P"], params["t0"])

            offset3 = np.nanmedian(wl_lc.posteriors['posterior_samples']['mflux_STIS2'])
            plt.scatter(phases, (ordered_binned_spectrum1[i]/np.median(ordered_binned_spectrum1[i][:10])) + v_off,alpha=0.3, label = "raw data (transit 1)", color = colors[i], s = 15)
            plt.scatter(phases, (ordered_binned_spectrum1[i]/np.median(ordered_binned_spectrum1[i][:10])) - (transit_plus_GP_model1 - transit_model1) + offset1 + v_off,alpha=1, label = "gp detrending (transit 1)", color = colors[i], s = 15)
            
            plt.scatter(phases_2, (ordered_binned_spectrum2[i]/np.median(ordered_binned_spectrum2[i][:10])) + v_off,alpha=0.3, label = "raw data (transit 2)", color = colors[i], s = 15)
            plt.scatter(phases_2, (ordered_binned_spectrum2[i]/np.median(ordered_binned_spectrum2[i][:10])) - (transit_plus_GP_model2 - transit_model2) + offset2 + v_off,alpha=1, label = "gp detrending (transit 2)", color = colors[i], s = 15)

            plt.plot(model_phases, transit_model2_resamp + offset3 + v_off, color = colors[i], label = "gp + transit model", zorder = 0)
            plt.fill_between(model_phases, error68_up+offset3 + v_off, error68_down+offset3 + v_off, color = colors[i], alpha = 0.2, label = "68% error")

            txt = plt.text(phases[0], 1.004 + fig_off ,str(bin_lst[i]/10000), size=11, color='black', weight = "bold")
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

            v_off -= vertical_offset
            fig_off -= figure_offset
    
    if plot == True:    
        plt.ylabel("Relative Flux + Vertical Offset")
        plt.xlabel("Phase")
        plt.ylim(1.004 + fig_off - figure_offset, 1.004 + figure_offset)
        if savefig == True:
            plt.savefig(fig_name, dpi = 300, facecolor = "white", bbox_inches="tight")
        else:
            plt.show()  
    
    if sys_method == "gp":
        for file in sorted(glob.glob("juliet_fits/" + juliet_name + "/*bin*/*posteriors.dat*")):
            fit = np.genfromtxt(file, dtype=None)
            for param in fit:
                if param[0].decode() == "p_p1":
                    depths.append(param[1])
                    depth_e.append((param[2], param[3]))
                    
        return(bin_centers, depths, depth_e, fit_value)
    elif sys_method == "linear":
        if sys_method == "LM":
            print("not implemented :]")
        else:
            for file in sorted(glob.glob("juliet_fits/" + juliet_name + "/*bin*/*posteriors.dat*")):
                fit = np.genfromtxt(file, dtype=None)
                for param in fit:
                    if param[0].decode() == "p_p1":
                        depths.append(param[1])
                        depth_e.append((param[2], param[3]))
            
            return(bin_centers, depths, depth_e, fit_value)    
    
# batman model for lm fits -- mostly just used for testing, use a sampler + juliet :]
def model_light_curve(p, t, params, detrenders):
    """
    Generates a light curve with systematics. Uses the lmfit 'p' dictionary.
    """
    params.t0  = p['t0']
    params.per = p['P']
    params.rp  = p['p']
    params.a   = p['a']
    params.inc = inclination(p['b'], p['a'])
    params.ecc = p['ecc']
    params.w   = p['omega']
    #params.limb_dark = "quadratic"
    #params.u = [0.1, 0.3]

    if p['p'] == 0:
        light_curve = np.ones(len(t)) # if it doesn't want a transit, give it a flat line
    else:
        light_curve = batman.TransitModel(params, t).light_curve(params)
    
    # there must be a way to make this more robust
    systematics =  p["V2_roll"]*np.array(detrenders["V2_roll"]) + p["V3_roll"]*np.array(detrenders["V3_roll"]) + \
    p["Latitude"]*np.array(detrenders["Latitude"]) + p["Longitude"]*np.array(detrenders["Longitude"]) + \
    p["RA"]*np.array(detrenders["RA"]) + p["DEC"]*np.array(detrenders["DEC"]) + 1 #p["linear"]*t + 1 #
    
    model = p['f0'] * light_curve * systematics
    t_final = np.linspace(t[0], t[-1], 1000)  
    light_curve_plot = batman.TransitModel(params, t_final).light_curve(params)

    return model, systematics, light_curve

    
def residual(p, t, params, data, err, detrenders):
    """
    Outputs the residual of the model and data.
    """
    
    model = model_light_curve(p, t, params, detrenders)[0]
    sys = model_light_curve(p, t, params, detrenders)[1]
    
    '''
    plt.scatter(t, data/sys, color = "purple", label = "with systematics")
    plt.legend()
    plt.xlabel("Time (BJD-TBD)")
    plt.ylabel("Counts")
    plt.show()
    plt.plot(t, model)
    plt.scatter(t, data, color = "blue", label = "Original", alpha = 0.5)
    plt.xlabel("Time (BJD-TBD)")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()
    '''
    if err == None:
        err = np.sqrt(p['f0']) # if no errorbars specified, assume shot noise uncertainty from baseline flux

    chi2 = sum((data-model)**2/err**2)
    res = np.std((data-model)/max(model))
    
    return (data-model)/err

def transit_final(p, t, params, detrenders):
    transit_model_final = batman.TransitModel(params, t).light_curve(params)
    systematics = p["V2_roll"]*np.array(detrenders["V2_roll"]) + p["V3_roll"]*np.array(detrenders["V3_roll"]) + \
    p["Latitude"]*np.array(detrenders["Latitude"]) + p["Longitude"]*np.array(detrenders["Longitude"]) + \
    p["RA"]*np.array(detrenders["RA"]) + p["DEC"]*np.array(detrenders["DEC"]) + 1
    
    #final_model = p["f0"] * transit_model_final * systematics
    
    return [p["f0"], transit_model_final, systematics]


def model_light_curve_joint(p, t1, t2, params, detrenders1, detrenders2):
    """
    Generates a light curve with systematics. Uses the lmfit 'p' dictionary.
    """
    params.t0  = p['t0']
    params.per = p['P']
    params.rp  = p['p']
    params.a   = p['a']
    params.inc = inclination(p['b'], p['a'])
    params.ecc = p['ecc']
    params.w   = p['omega']
    
    #phases1 = juliet.utils.get_phases(t1, p['P'], p['t0'])
    #phases2 = juliet.utils.get_phases(t2, p['P'], p['t0'])

    if p['p'] == 0:
        light_curve1 = np.ones(len(t)) # if it doesn't want a transit, give it a flat line
    else:
        light_curve1 = batman.TransitModel(params, t1).light_curve(params)
        light_curve2 = batman.TransitModel(params, t2).light_curve(params)
        
    
    # there must be a way to make this more robust
    systematics1 =  p["V2_roll_1"]*np.array(detrenders1["V2_roll"]) + p["V3_roll_1"]*np.array(detrenders1["V3_roll"]) + \
    p["Latitude_1"]*np.array(detrenders1["Latitude"]) + p["Longitude_1"]*np.array(detrenders1["Longitude"]) + \
    p["RA_1"]*np.array(detrenders1["RA"]) + p["DEC_1"]*np.array(detrenders1["DEC"]) + 1 
    
    systematics2 =  p["V2_roll_2"]*np.array(detrenders2["V2_roll"]) + p["V3_roll_2"]*np.array(detrenders2["V3_roll"]) + \
    p["Latitude_2"]*np.array(detrenders2["Latitude"]) + p["Longitude_2"]*np.array(detrenders2["Longitude"]) + \
    p["RA_2"]*np.array(detrenders2["RA"]) + p["DEC_2"]*np.array(detrenders2["DEC"]) + 1 
    
    transit1 = p['f0'] * light_curve1 * systematics1
    transit2 = p['f0'] * light_curve2 * systematics2
    phases = juliet.utils.get_phases(t1, p["P"], p["t0"])
    phases2 = juliet.utils.get_phases(t2, p["P"], p["t0"])
    
    return [transit1, transit2], [systematics1, systematics2], [light_curve1, light_curve2]

    
def residual_joint(p, t1, t2, params, data1, data2, err, detrenders1, detrenders2):
    """
    Outputs the residual of the model and data.
    """
    
    model1 = model_light_curve_joint(p, t1, t2, params, detrenders1, detrenders2)[0][0]
    model2 = model_light_curve_joint(p, t1, t2, params, detrenders1, detrenders2)[0][1]
    sys1 = model_light_curve_joint(p, t1, t2, params, detrenders1, detrenders2)[1][0]
    sys2 = model_light_curve_joint(p, t1, t2, params, detrenders1, detrenders2)[1][1]
    
    phases = juliet.utils.get_phases(t1, p["P"], p["t0"])
    phases2 = juliet.utils.get_phases(t2, p["P"], p["t0"])
    
    '''
    plt.scatter(t, data/sys, color = "purple", label = "with systematics")
    plt.legend()
    plt.xlabel("Time (BJD-TBD)")
    plt.ylabel("Counts")
    plt.show()
    plt.plot(t, model)
    plt.scatter(t, data, color = "blue", label = "Original", alpha = 0.5)
    plt.xlabel("Time (BJD-TBD)")
    plt.ylabel("Counts")
    plt.legend()
    plt.show()
    '''
    if err == None:
        err = np.sqrt(p['f0']) # if no errorbars specified, assume shot noise uncertainty from baseline flux

    chi2 = sum((data1-model1)**2/err**2) + sum((data2-model2)**2/err**2)
    res = np.std((data1-model1)/max(model1)) + np.std((data2-model2)/max(model2))
    
    return ((data1-model1)**2/err**2) + ((data2-model2)**2/err**2)

def transit_final_joint(p, t_final, params, detrenders1, detrenders2):
    transit_model_final = batman.TransitModel(params, t_final).light_curve(params)
    
    systematics1 =  p["V2_roll_1"]*np.array(detrenders1["V2_roll"]) + p["V3_roll_1"]*np.array(detrenders1["V3_roll"]) + \
    p["Latitude_1"]*np.array(detrenders1["Latitude"]) + p["Longitude_1"]*np.array(detrenders1["Longitude"]) + \
    p["RA_1"]*np.array(detrenders1["RA"]) + p["DEC_1"]*np.array(detrenders1["DEC"]) + 1 
    
    systematics2 =  p["V2_roll_2"]*np.array(detrenders2["V2_roll"]) + p["V3_roll_2"]*np.array(detrenders2["V3_roll"]) + \
    p["Latitude_2"]*np.array(detrenders2["Latitude"]) + p["Longitude_2"]*np.array(detrenders2["Longitude"]) + \
    p["RA_2"]*np.array(detrenders2["RA"]) + p["DEC_2"]*np.array(detrenders2["DEC"]) + 1 
    
    #final_model = p["f0"] * transit_model_final * systematics
    
    return [p["f0"], transit_model_final, [systematics1, systematics2]]

# binning function for spectroscopic fits
def binning(wl, spectra, bins, bin_unit = "nm"):
    bin_lst = np.loadtxt(bins, delimiter = "\t")
    if bin_unit == "nm":
        bin_lst = bin_lst*10
    
    #binned_spectrum = []
    bin_centers = []
    
    for i in range(len(bin_lst)):
        idx1 = np.where(wl > bin_lst[i][0])[0][0]
        idx2 = np.where(wl > bin_lst[i][1])[0][0]
        
        bin_centers.append((bin_lst[i][0]+bin_lst[i][1])/2)
        
        inbet = []
        inbet_err = []
        for spectrum in spectra:
            binned = np.nansum(spectrum[1][idx1:idx2])
            binned_error = np.sqrt(np.nansum((1/np.sqrt(spectrum[2][idx1:idx2]))**2))
            inbet.append([binned])
            inbet_err.append([binned_error])
        
        if i == 0:
            binned_spectrum = np.copy(inbet)
            binned_errors = np.copy(inbet_err)
        else:
            binned_spectrum = np.concatenate((binned_spectrum, inbet), axis = 1)
            binned_errors = np.concatenate((binned_errors, inbet_err), axis = 1)
        
    # lol my brain can't think of a better way to do this
    ordered_binned_spectrum = []
    ordered_binned_errors = []
    for j in range(len(binned_spectrum[0])):
        ordered_binned_spectrum.append(binned_spectrum[:,j])
        ordered_binned_errors.append(binned_errors[:,j])

    return(ordered_binned_spectrum, ordered_binned_errors, bin_centers)
    
    
#Limb Darkeneing
@custom_model
def nonlinear_limb_darkening(x, c0=0.0, c1=0.0, c2=0.0, c3=0.0):
    """
    Define non-linear limb darkening model with four parameters c0, c1, c2, c3.
    """
    model = (1. - (c0 * (1. - x ** (1. / 2)) + c1 * (1. - x ** (2. / 2)) + c2 * (1. - x ** (3. / 2)) + c3 *
                   (1. - x ** (4. / 2))))
    return model


@custom_model
def quadratic_limb_darkening(x, aLD=0.0, bLD=0.0):
    """
    Define linear limb darkening model with parameters aLD and bLD.
    """
    model = 1. - aLD * (1. - x) - bLD * (1. - x) ** (4. / 2.)
    return model

def limb_dark_fit(grating, wsdata, M_H, Teff, logg, dirsen, ld_model='1D'):
    """
    Calculates stellar limb-darkening coefficients for a given wavelength bin.

    Currently supports:
    HST STIS G750L, G750M, G430L gratings
    HST WFC3 UVIS/G280, IR/G102, IR/G141 grisms

    What is used for 1D models - Kurucz (?)
    Procedure from Sing et al. (2010, A&A, 510, A21).
    Uses 3D limb darkening from Magic et al. (2015, A&A, 573, 90).
    Uses photon FLUX Sum over (lambda*dlamba).
    :param grating: string; grating to use ('G430L','G750L','G750M', 'G280', 'G102', 'G141')
    :param wsdata: array; data wavelength solution
    :param M_H: float; stellar metallicity
    :param Teff: float; stellar effective temperature (K)
    :param logg: float; stellar gravity
    :param dirsen: string; path to main limb darkening directory
    :param ld_model: string; '1D' or '3D', makes choice between limb darkening models; default is 1D
    :return: uLD: float; linear limb darkening coefficient
    aLD, bLD: float; quadratic limb darkening coefficients
    cp1, cp2, cp3, cp4: float; three-parameter limb darkening coefficients
    c1, c2, c3, c4: float; non-linear limb-darkening coefficients
    """

    print('You are using the', str(ld_model), 'limb darkening models.')

    if ld_model == '1D':

        direc = os.path.join(dirsen, 'Kurucz')

        print('Current Directories Entered:')
        print('  ' + dirsen)
        print('  ' + direc)

        # Select metallicity
        M_H_Grid = np.array([-0.1, -0.2, -0.3, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
        M_H_Grid_load = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 17, 20, 21, 22, 23, 24])
        optM = (abs(M_H - M_H_Grid)).argmin()
        MH_ind = M_H_Grid_load[optM]

        # Determine which model is to be used, by using the input metallicity M_H to figure out the file name we need
        direc = 'Kurucz'
        file_list = 'kuruczlist.sav'
        sav1 = readsav(os.path.join(dirsen, file_list))
        model = bytes.decode(sav1['li'][MH_ind])  # Convert object of type "byte" to "string"

        # Select Teff and subsequently logg
        Teff_Grid = np.array([3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500])
        optT = (abs(Teff - Teff_Grid)).argmin()

        logg_Grid = np.array([4.0, 4.5, 5.0])
        optG = (abs(logg - logg_Grid)).argmin()

        if logg_Grid[optG] == 4.0:
            Teff_Grid_load = np.array([8, 19, 30, 41, 52, 63, 74, 85, 96, 107, 118, 129, 138])

        elif logg_Grid[optG] == 4.5:
            Teff_Grid_load = np.array([9, 20, 31, 42, 53, 64, 75, 86, 97, 108, 119, 129, 139])

        elif logg_Grid[optG] == 5.0:
            Teff_Grid_load = np.array([10, 21, 32, 43, 54, 65, 76, 87, 98, 109, 120, 130, 140])

        # Where in the model file is the section for the Teff we want? Index T_ind tells us that.
        T_ind = Teff_Grid_load[optT]
        header_rows = 3    #  How many rows in each section we ignore for the data reading
        data_rows = 1221   # How  many rows of data we read
        line_skip_data = (T_ind + 1) * header_rows + T_ind * data_rows   # Calculate how many lines in the model file we need to skip in order to get to the part we need (for the Teff we want).
        line_skip_header = T_ind * (data_rows + header_rows)

        # Read the header, in case we want to have the actual Teff, logg and M_H info.
        # headerinfo is a pandas object.
        headerinfo = pd.read_csv(os.path.join(dirsen, direc, model), delim_whitespace=True, header=None,
                                 skiprows=line_skip_header, nrows=1)

        Teff_model = headerinfo[1].values[0]
        logg_model = headerinfo[3].values[0]
        MH_model = headerinfo[6].values[0]
        MH_model = float(MH_model[1:-1])

        print('\nClosest values to your inputs:')
        print('Teff: ', Teff_model)
        print('M_H: ', MH_model)
        print('log(g): ', logg_model)

        # Read the data; data is a pandas object.
        data = pd.read_csv(os.path.join(dirsen, direc, model), delim_whitespace=True, header=None,
                              skiprows=line_skip_data, nrows=data_rows)

        # Unpack the data
        ws = data[0].values * 10   # Import wavelength data
        f0 = data[1].values / (ws * ws)
        f1 = data[2].values * f0 / 100000.
        f2 = data[3].values * f0 / 100000.
        f3 = data[4].values * f0 / 100000.
        f4 = data[5].values * f0 / 100000.
        f5 = data[6].values * f0 / 100000.
        f6 = data[7].values * f0 / 100000.
        f7 = data[8].values * f0 / 100000.
        f8 = data[9].values * f0 / 100000.
        f9 = data[10].values * f0 / 100000.
        f10 = data[11].values * f0 / 100000.
        f11 = data[12].values * f0 / 100000.
        f12 = data[13].values * f0 / 100000.
        f13 = data[14].values * f0 / 100000.
        f14 = data[15].values * f0 / 100000.
        f15 = data[16].values * f0 / 100000.
        f16 = data[17].values * f0 / 100000.

        # Make single big array of them
        fcalc = np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16])
        phot1 = np.zeros(fcalc.shape[0])

        # Define mu
        mu = np.array([1.000, .900, .800, .700, .600, .500, .400, .300, .250, .200, .150, .125, .100, .075, .050, .025, .010])

        # Passed on to main body of function are: ws, fcalc, phot1, mu

    elif ld_model == '3D':

        direc = os.path.join(dirsen, '3DGrid')

        print('Current Directories Entered:')
        print('  ' + dirsen)
        print('  ' + direc)

        # Select metallicity
        M_H_Grid = np.array([-3.0, -2.0, -1.0, 0.0])  # Available metallicity values in 3D models
        M_H_Grid_load = ['30', '20', '10', '00']  # The according identifiers to individual available M_H values
        optM = (abs(M_H - M_H_Grid)).argmin()  # Find index at which the closes M_H values from available values is to the input M_H.

        # Select Teff
        Teff_Grid = np.array([4000, 4500, 5000, 5500, 5777, 6000, 6500, 7000])  # Available Teff values in 3D models
        optT = (abs(Teff - Teff_Grid)).argmin()  # Find index at which the Teff values is, that is closest to input Teff.

        # Select logg, depending on Teff. If several logg possibilities are given for one Teff, pick the one that is
        # closest to user input (logg).

        if Teff_Grid[optT] == 4000:
            logg_Grid = np.array([1.5, 2.0, 2.5])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 4500:
            logg_Grid = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 5000:
            logg_Grid = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 5500:
            logg_Grid = np.array([3.0, 3.5, 4.0, 4.5, 5.0])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 5777:
            logg_Grid = np.array([4.4])
            optG = 0

        elif Teff_Grid[optT] == 6000:
            logg_Grid = np.array([3.5, 4.0, 4.5])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 6500:
            logg_Grid = np.array([4.0, 4.5])
            optG = (abs(logg - logg_Grid)).argmin()

        elif Teff_Grid[optT] == 7000:
            logg_Grid = np.array([4.5])
            optG = 0

        # Select Teff and Log g. Mtxt, Ttxt and Gtxt are then put together as string to load correct files.
        Mtxt = M_H_Grid_load[optM]
        Ttxt = "{:2.0f}".format(Teff_Grid[optT] / 100)
        if Teff_Grid[optT] == 5777:
            Ttxt = "{:4.0f}".format(Teff_Grid[optT])
        Gtxt = "{:2.0f}".format(logg_Grid[optG] * 10)

        #
        file = 'mmu_t' + Ttxt + 'g' + Gtxt + 'm' + Mtxt + 'v05.flx'
        print('Filename:', file)

        # Read data from IDL .sav file
        sav = readsav(os.path.join(direc, file))  # readsav reads an IDL .sav file
        ws = sav['mmd'].lam[0]  # read in wavelength
        flux = sav['mmd'].flx  # read in flux
        Teff_model = Teff_Grid[optT]
        logg_model = logg_Grid[optG]
        MH_model = str(M_H_Grid[optM])

        print('\nClosest values to your inputs:')
        print('Teff  : ', Teff_model)
        print('M_H   : ', MH_model)
        print('log(g): ', logg_model)

        f0 = flux[0]
        f1 = flux[1]
        f2 = flux[2]
        f3 = flux[3]
        f4 = flux[4]
        f5 = flux[5]
        f6 = flux[6]
        f7 = flux[7]
        f8 = flux[8]
        f9 = flux[9]
        f10 = flux[10]

        # Make single big array of them
        fcalc = np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10])
        phot1 = np.zeros(fcalc.shape[0])

        # Mu from grid
        # 0.00000    0.0100000    0.0500000     0.100000     0.200000     0.300000   0.500000     0.700000     0.800000     0.900000      1.00000
        mu = sav['mmd'].mu

        # Passed on to main body of function are: ws, fcalc, phot1, mu

    ### Load response function and interpolate onto kurucz model grid

    # FOR STIS
    if grating == 'G430L':
        sav = readsav(os.path.join(dirsen, 'G430L.STIS.sensitivity.sav'))  # wssens,sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 3

    if grating == 'G750M':
        sav = readsav(os.path.join(dirsen, 'G750M.STIS.sensitivity.sav'))  # wssens, sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 0.554

    if grating == 'G750L':
        sav = readsav(os.path.join(dirsen, 'G750L.STIS.sensitivity.sav'))  # wssens, sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 4.882

    # FOR WFC3
    if grating == 'G141':  # http://www.stsci.edu/hst/acs/analysis/reference_files/synphot_tables.html
        sav = readsav(os.path.join(dirsen, 'G141.WFC3.sensitivity.sav'))  # wssens, sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 1

    if grating == 'G102':  # http://www.stsci.edu/hst/acs/analysis/reference_files/synphot_tables.html
        sav = readsav(os.path.join(dirsen, 'G141.WFC3.sensitivity.sav'))  # wssens, sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 1

    if grating == 'G280':  # http://www.stsci.edu/hst/acs/analysis/reference_files/synphot_tables.html
        sav = readsav(os.path.join(dirsen, 'G280.WFC3.sensitivity.sav'))  # wssens, sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 1

    # FOR JWST
    if grating == 'NIRSpecPrism':  # http://www.stsci.edu/hst/acs/analysis/reference_files/synphot_tables.html
        sav = readsav(os.path.join(dirsen, 'NIRSpec.prism.sensitivity.sav'))  # wssens, sensitivity
        wssens = sav['wssens']
        sensitivity = sav['sensitivity']
        wdel = 12


    widek = np.arange(len(wsdata))
    wsHST = wssens
    wsHST = np.concatenate((np.array([wsHST[0] - wdel - wdel, wsHST[0] - wdel]),
                            wsHST,
                            np.array([wsHST[len(wsHST) - 1] + wdel,
                                      wsHST[len(wsHST) - 1] + wdel + wdel])))

    respoutHST = sensitivity / np.max(sensitivity)
    respoutHST = np.concatenate((np.zeros(2), respoutHST, np.zeros(2)))
    inter_resp = interp1d(wsHST, respoutHST, bounds_error=False, fill_value=0)
    respout = inter_resp(ws)  # interpolate sensitivity curve onto model wavelength grid

    wsdata = np.concatenate((np.array([wsdata[0] - wdel - wdel, wsdata[0] - wdel]), wsdata,
                             np.array([wsdata[len(wsdata) - 1] + wdel, wsdata[len(wsdata) - 1] + wdel + wdel])))
    respwavebin = wsdata / wsdata * 0.0
    widek = widek + 2  # need to add two indicies to compensate for padding with 2 zeros
    respwavebin[widek] = 1.0
    data_resp = interp1d(wsdata, respwavebin, bounds_error=False, fill_value=0)
    reswavebinout = data_resp(ws)  # interpolate data onto model wavelength grid

    # Integrate over the spectra to make synthetic photometric points.
    for i in range(fcalc.shape[0]):  # Loop over spectra at diff angles
        fcal = fcalc[i, :]
        Tot = int_tabulated(ws, ws * respout * reswavebinout)
        phot1[i] = (int_tabulated(ws, ws * respout * reswavebinout * fcal, sort=True)) / Tot

    if ld_model == '1D':
        yall = phot1 / phot1[0]
    elif ld_model == '3D':
        yall = phot1 / phot1[10]

    Co = np.zeros((6, 4))   # NOT-REUSED

    A = [0.0, 0.0, 0.0, 0.0]  # c1, c2, c3, c4      # NOT-REUSED
    x = mu[1:]     # wavelength
    y = yall[1:]   # flux
    weights = x / x   # NOT-REUSED

    # Start fitting the different models
    fitter = LevMarLSQFitter()

    # Fit a four parameter non-linear limb darkening model and get fitted variables, c1, c2, c3, c4.
    corot_4_param = nonlinear_limb_darkening()
    corot_4_param = fitter(corot_4_param, x, y)
    c1, c2, c3, c4 = corot_4_param.parameters

    # Fit a three parameter non-linear limb darkening model and get fitted variables, cp2, cp3, cp4 (cp1 = 0).
    corot_3_param = nonlinear_limb_darkening()
    corot_3_param.c0.fixed = True  # 3 param is just 4 param with c0 = 0.0
    corot_3_param = fitter(corot_3_param, x, y)
    cp1, cp2, cp3, cp4 = corot_3_param.parameters

    # Fit a quadratic limb darkening model and get fitted parameters aLD and bLD.
    quadratic = quadratic_limb_darkening()
    quadratic = fitter(quadratic, x, y)
    aLD, bLD = quadratic.parameters

    # Fit a linear limb darkening model and get fitted variable uLD.
    linear = nonlinear_limb_darkening()
    linear.c0.fixed = True
    linear.c2.fixed = True
    linear.c3.fixed = True
    linear = fitter(linear, x, y)
    uLD = linear.c1.value

    print('\nLimb darkening parameters:')
    print("4param \t{:0.8f}\t{:0.8f}\t{:0.8f}\t{:0.8f}".format(c1, c2, c3, c4))
    print("3param \t{:0.8f}\t{:0.8f}\t{:0.8f}".format(cp2, cp3, cp4))
    print("Quad \t{:0.8f}\t{:0.8f}".format(aLD, bLD))
    print("Linear \t{:0.8f}".format(uLD))

    return uLD, c1, c2, c3, c4, cp1, cp2, cp3, cp4, aLD, bLD


def int_tabulated(X, F, sort=False):
    Xsegments = len(X) - 1

    # Sort vectors into ascending order.
    if not sort:
        ii = np.argsort(X)
        X = X[ii]
        F = F[ii]

    while (Xsegments % 4) != 0:
        Xsegments = Xsegments + 1

    Xmin = np.min(X)
    Xmax = np.max(X)

    # Uniform step size.
    h = (Xmax + 0.0 - Xmin) / Xsegments
    # Compute the interpolates at Xgrid.
    # x values of interpolates >> Xgrid = h * FINDGEN(Xsegments + 1L) + Xmin
    z = splev(h * np.arange(Xsegments + 1) + Xmin, splrep(X, F))

    # Compute the integral using the 5-point Newton-Cotes formula.
    ii = (np.arange((len(z) - 1) / 4, dtype=int) + 1) * 4

    return np.sum(2.0 * h * (7.0 * (z[ii - 4] + z[ii]) + 32.0 * (z[ii - 3] + z[ii - 1]) + 12.0 * z[ii - 2]) / 45.0)


# PCA TOOLS:
def get_sigma(x):
    """
    This function returns the MAD-based standard-deviation.
    """
    median = np.median(x)
    mad = np.median(np.abs(x-median))
    return 1.4826*mad

def standarize_data(input_data):
    output_data = np.copy(input_data)
    averages = np.median(input_data,axis=1)
    for i in range(len(averages)):
        sigma = get_sigma(output_data[i,:])
        output_data[i,:] = output_data[i,:] - averages[i]
        output_data[i,:] = output_data[i,:]/sigma
    return output_data

# typically the data will already be standardized
def classic_PCA(Input_Data, standarize = False):
    """
    classic_PCA function

    Description

    This function performs the classic Principal Component Analysis on a given dataset.
    """
    if standarize:
        Data = standarize_data(Input_Data)
    else:
        Data = np.copy(Input_Data)
    eigenvectors_cols,eigenvalues,eigenvectors_rows = np.linalg.svd(np.cov(Data))
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx[::-1]]
    eigenvectors_cols = eigenvectors_cols[:,idx[::-1]]
    eigenvectors_rows = eigenvectors_rows[idx[::-1],:]
    # Return: V matrix, eigenvalues and the principal components.
    return eigenvectors_rows,eigenvalues,np.dot(eigenvectors_rows,Data)



