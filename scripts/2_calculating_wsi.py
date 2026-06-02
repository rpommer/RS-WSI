
"""
Author: rpommer
Date: August 13, 2025
Description:
    Calculate WSI and perform linear regression between yield data and WSI, SR, LE, NDVI, and SAVI. 
    For 2018, perform linear regression between matric potential data and WSI. 
"""

print("Start of Script.")
import geopandas as gpd
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import Point, mapping
from scipy import stats
# from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import spearmanr, rankdata, pearsonr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import math

''' Setting up directories '''

if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

base_dir = os.path.abspath(os.path.join(script_dir, ".."))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


''' user input '''

ext = 'golfcourseMonthly_LR_optimalwindows'

# Matric Potential Location
locs = os.path.join(base_dir, "input", "matric_pot", "Soil_Moisture_locations.shp")

# CSV containing start/end date information
input_info_path = os.path.join(base_dir, "input", "input_info_months_optimalwindows.csv")

''' Functions '''

def makedirs(directory_list):    
    """    Make directory (if not exists) from a list of directory.    
    :param directory_list: A list of directories to create.    
    :return: None.    """    
    for directory in directory_list:        
        if not os.path.exists(directory):            
            os.makedirs(directory)
                      
def write_array_to_raster(raster_arr, raster_file, transform, output_path, dtype=None,
                          ref_file=None, nodata=None):
    """
    Write raster array to Geotiff format.

    :param raster_arr: Raster array data to be written.
    :param raster_file: Original rasterio raster file containing geo-coordinates.
    :param transform: Affine transformation matrix.
    :param output_path: Output filepath.
    :param dtype: Output raster data type. Default set to None.
    :param ref_file: Write output raster considering parameters from reference raster file.
    :param nodata: no_data_value set as -9999.

    :return: Output filepath.
    """
    if dtype is None:
        dtype = raster_arr.dtype

    if ref_file:
        raster_file = rasterio.open(ref_file)
        transform = raster_file.transform

    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=raster_arr.shape[0],
            width=raster_arr.shape[1],
            dtype=dtype,
            count=raster_file.count,
            crs=raster_file.crs,
            transform=transform,
            nodata=nodata
    ) as dst:
        dst.write(raster_arr, raster_file.count)

    return output_path

def zScores(array):
    ''' Normalize using zScores
    :parms: array
    :return: normalized array
    '''
    zscores = (array - np.nanmean(array))/np.nanstd(array)
    return zscores

def windowExtract(locationShp, rasterFile, windowSize=1, missingDataval=0, prefix=None):
    ''' Extract information from rasters using xy locations at point and for window around point.
    :parms: locationShp: input shapefile
    :parms: rasterFile: raster file to be extracted
    :parms: windowSize: number of pixels in either direction to extract information from. default = 1
    :parms: missingDataval: missing data value; default = 0
    :parms: prefix: string
    :return: df of statistics about extracted raster information
    :return: arrays of extracted values '''
    # extract raster points using shp
    loc = locationShp.copy() # create copy of location info
    siteRes, minRes,meanRes, maxRes, stdRes, outArrays = [], [], [], [], [], [] # create empty lists
    for idx, row in loc.iterrows():
        point = row.geometry
        site = row['Site_Ryan']
        row_pix, col_pix = rasterFile.index(point.x, point.y) # convert point to raster coords
        window = rasterio.windows.Window(col_pix - windowSize, row_pix - windowSize, 3,3) # define window
        
        out_image = rasterFile.read(1, window=window) # apply window
        out_image = np.where(out_image == missingDataval, np.nan, out_image) # remove missing data values
        outArrays.append(out_image)

        # min_val = np.nanmin(out_image)
        # mean_val = np.nanmean(out_image)
        # max_val = np.nanmax(out_image)
        # std_val = np.nanstd(out_image)
        
        if np.all(np.isnan(out_image)):
            min_val = mean_val = max_val = std_val = np.nan
        else:
            min_val = np.nanmin(out_image)
            mean_val = np.nanmean(out_image)
            max_val = np.nanmax(out_image)
            std_val = np.nanstd(out_image)
        minRes.append(min_val)
        meanRes.append(mean_val)
        maxRes.append(max_val)
        stdRes.append(std_val)
        siteRes.append(site)
        dict = {'Site': siteRes, 
                f'{prefix}_min': minRes,
                f'{prefix}_mean': meanRes, 
                f'{prefix}_max': maxRes, 
                f'{prefix}_std': stdRes
                }
        df = pd.DataFrame(dict)
        
    return df, outArrays

def NoWindowExtract(locationShp, rasterFile, windowSize=1, missingDataval=0, prefix=None):
    ''' Extract information from rasters using xy locations.
    :parms: locationShp: input shapefile
    :parms: rasterFile: raster file to be extracted
    :parms: windowSize: number of pixels in either direction to extract information from. default = 1
    :parms: missingDataval: missing data value; default = 0
    :parms: prefix: string
    :return: df of statistics about extracted raster information
    :return: arrays of extracted values '''
    # extract raster points using shp
    loc = locationShp.copy() # create copy of location info
    siteRes, minRes,meanRes, maxRes, stdRes, outArrays = [], [], [], [], [], [] # create empty lists
    for idx, row in loc.iterrows():
        point = row.geometry
        site = row['Site_Ryan']
        row_pix, col_pix = rasterFile.index(point.x, point.y) # convert point to raster coords
        window = rasterio.windows.Window(col_pix - windowSize, row_pix - windowSize, 1,1) # define window
        
        out_image = rasterFile.read(1, window=window) # apply window
        out_image = np.where(out_image == missingDataval, np.nan, out_image) # remove missing data values
        outArrays.append(out_image)
        
        if np.all(np.isnan(out_image)):
            min_val = mean_val = max_val = std_val = np.nan
        else:
            min_val = np.nanmin(out_image)
            mean_val = np.nanmean(out_image)
            max_val = np.nanmax(out_image)
            std_val = np.nanstd(out_image)
        
        minRes.append(min_val)
        meanRes.append(mean_val)
        maxRes.append(max_val)
        stdRes.append(std_val)
        siteRes.append(site)
        dict = {'Site': siteRes, 
                f'{prefix}_min': minRes,
                f'{prefix}_mean': meanRes, 
                f'{prefix}_max': maxRes, 
                f'{prefix}_std': stdRes
                }
        df = pd.DataFrame(dict)
        
    return df, outArrays

def linear_regression_residuals(arr1, arr2, label1, label2, year, plot=False, binned=True):
    ''' Perform linear regression on two arrays and plot with data and binned means of data.

    '''
    x_data = arr1
    y_data = arr2
    x_label= str(label1)
    y_label = str(label2)

    plot_title = f'{year}'
    box_position = [0.03, 0.21]
    x_clean = x_data
    y_clean = y_data
    valid_mask = (x_clean != 0) & (y_clean != 0) & ~np.isnan(x_clean) & ~np.isnan(y_clean)
    x_valid = x_clean[valid_mask]
    y_valid = y_clean[valid_mask]
    
    if x_valid.size == 0 or y_valid.size == 0:
        return np.array([]), np.array([]), np.nan
    
    x_valid_reshaped = x_valid.reshape(-1,1)
    y_valid_reshaped = y_valid
    model = LinearRegression()
    model.fit(x_valid_reshaped, y_valid_reshaped)
    residuals = y_valid_reshaped - model.predict(x_valid_reshaped)
    return residuals

def et2le(et_arr):
    ''' Convert ET to Latent Heat Flux (LE)
    :parms: array of ET values in mm
    :return: array of LE values in W/m2 '''
    rho = 1000 # kg/m3, density of water
    lv = 2.5e6 # J/kg, latent heat of vaporization
    le_arr = rho * lv * et_arr * (1/86400) * (1/1000)     # latent heat flux in W/m2
    return le_arr

def windowExtract_BoxPlot(location_path, input_raster_path, windowSize=1, missingDataval=0, prefix=None, fontsize=15, color='cornflowerblue', ylabel='WSI'):
    ''' Extract information from rasters using xy locations and output boxplot.
    :parms: location_path: path to shapefile
    :parms: input_raster_path: path to raster file
    :parms: windowSize: number of pixels in either direction to extract information from. default = 1
    :parms: missingDataval: missing data value; default = 0
    :parms: prefix: string
    :parms: fontsize: int
    :parms: color: plotting color, see matplotlib colors
    :parms: ylabel: string, axis label
    :return: df of statistics about extracted raster information
    :return: arrays of extracted values '''
    
    site_names = ['Site 1', 'Site 2', 'Site 3', 'Site 4','Site 5', 'Site 6']
    loc_shp = gpd.read_file(location_path)
    loc_shp['geometry'] = [Point(xy) for xy in zip(loc_shp['Lon'], loc_shp['Lat'])]
    loc_shp = gpd.GeoDataFrame(loc_shp, geometry='geometry', crs="EPSG:4326")
    with rasterio.open(input_raster_path) as index_ras:
        if loc_shp.crs != index_ras.crs:
            print(f'{loc_shp.crs}')
            print('CRS does not match. Reprojecting...')
            loc_shp = loc_shp.to_crs(index_ras.crs)
            print(f'New CRS for locations: {loc_shp.crs}')
        index_outArrays = windowExtract(loc_shp, index_ras, windowSize=1, missingDataval=0, prefix=None)
        data_boxPlot = [arr.flatten() for arr in index_outArrays[1]]
        fig, ax = plt.subplots(figsize=(6.5,3.5))
        bbox = ax.boxplot(data_boxPlot, tick_labels=site_names, patch_artist=True,            
                          boxprops=dict(facecolor=color, color='black'),
                          capprops=dict(color='black'),
                          whiskerprops=dict(color='black'),
                          flierprops=dict(color=color, markeredgecolor='black'),
                          medianprops=dict(color='black'),)
        ax.tick_params('both', labelsize=fontsize)
        ax.set_title(f'{ylabel}', fontsize=fontsize)
        ax.set_ylabel(f'{ylabel}', fontsize=fontsize)
        plt.tight_layout()
        plt.show()
    return data_boxPlot
        
def MatPot_extractRas(input_raster_path, location_path, medians, xlabel_str="NDVI", def_fontSize = 22, boxPositionX = 0.03, boxPositionY = 0.18, color='cornflowerblue', windowExtractFlag=False):
    """ Perform linear regression on matric potential data and extracted raster values. Create xy plot of results.
    :parms: input_raster_path: input raster path
    :parms: location_path: input location path
    :parms: medians: median daily matric potential value for each sensor as list of floats
    :parms: xlabel_str: string, xlabel
    :parms: def_fontSize: int, font size
    :parms: boxPositionX, boxPositionY: flaots (0.03), (0.18)
    :parms: color: matplotlib color
    :parms: windowExtractFlag: True/False 
    :return: linear regression plot
    """
    loc_shp = gpd.read_file(location_path)
    loc_shp['geometry'] = [Point(xy) for xy in zip(loc_shp['Lon'], loc_shp['Lat'])]
    loc_shp = gpd.GeoDataFrame(loc_shp, geometry='geometry', crs="EPSG:4326")
    with rasterio.open(input_raster_path) as index_ras:
        if loc_shp.crs != index_ras.crs:
            print(f'{loc_shp.crs}')
            print('CRS does not match. Reprojecting...')
            loc_shp = loc_shp.to_crs(index_ras.crs)
            print(f'New CRS for locations: {loc_shp.crs}')
        if windowExtractFlag:
            vals_index, arr_index = windowExtract(loc_shp, index_ras, prefix='index')
        else:
            vals_index, arr_index = NoWindowExtract(loc_shp, index_ras, prefix='index')
           
    with rasterio.open(yield_path) as yield_ras:   
        if windowExtractFlag:
            vals_yield, arr_yield = windowExtract(loc_shp, yield_ras, prefix='yield')
        else:
            vals_yield, arr_yield = NoWindowExtract(loc_shp, yield_ras, prefix='yield')

    ext = output_extension.split('_')[-1]
    # Adding median information to dataframe and merging with extracted raster values
    order = [6, 7, 9, 10, 11, 12]
    df_temp = pd.DataFrame(list(zip(medians, stations, stress_days)), columns=['Median_Mat_Pot', 'station','Tot_Stress_Days']) # build dataframe
    df_temp.index = df_temp['station']
    df_temp_sorted = df_temp.loc[order] # sort by defined order
    df_temp_sorted['Site'] = [1,2,3,4,5,6]
    df_temp_sorted.set_index('Site', inplace=True)
    df_extract = vals_yield
    df_list = [vals_index]
    for data in df_list:
        df_extract = df_extract.merge(data, on='Site')
    df_final = df_extract.merge(df_temp_sorted, on='Site') # merge dataframes on Site column
    try:
        x_var = 'index_mean'
        y_var = 'Median_Mat_Pot'
        xlabel= xlabel_str
        ylabel= 'Median Matric Potential (kPa)'
        x_reshaped = df_final[x_var].to_numpy().reshape(-1, 1)
        y_reshaped = df_final[y_var].to_numpy()
        model = LinearRegression().fit(x_reshaped, y_reshaped)
        slope = model.coef_[0]
        intercept = model.intercept_
        r_squared = model.score(x_reshaped, y_reshaped)
        print(f"\t{xlabel_str} R² Mat Pot: {round(r_squared,2)}")

        plt.close('all')
        fig, ax = plt.subplots(figsize=(8.5, 8))
        # Scatter
        ax.scatter(df_final[x_var], df_final[y_var], s=100, color=color)
        # Regression line
        x_line = np.linspace(df_final[x_var].min(), df_final[x_var].max(), 200).reshape(-1, 1)
        ax.plot(x_line, model.predict(x_line), color='black', linewidth=3, alpha=0.5)
        # Ensure limits are set before annotation offset logic
        plt.draw()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Annotate points
        for nom, i, j in zip(df_final['Site'], df_final[x_var], df_final[y_var]):
            x_offset = 5
            y_offset = -20
            if i > xlim[1] - 0.05 * (xlim[1] - xlim[0]):
                x_offset = -40
            elif i < xlim[0] + 0.05 * (xlim[1] - xlim[0]):
                x_offset = 10
            if j > ylim[1] - 0.05 * (ylim[1] - ylim[0]):
                y_offset = -20
            elif j < ylim[0] + 0.05 * (ylim[1] - ylim[0]):
                y_offset = 20
            ann = ax.transData.transform((i,j)) + np.array([x_offset, y_offset])
            ann_data_coords = ax.transData.inverted().transform(ann)
            ax.plot([i, ann_data_coords[0]], [j, ann_data_coords[1]],
            linestyle='-', linewidth=1, color='gray', alpha=0.6)
            ax.annotate(f"Site {nom}", xy=(i, j), xytext=(x_offset, y_offset),
                        textcoords='offset points', fontsize=def_fontSize, clip_on=False)
        # Regression info box
        # textstr = f"Slope={slope:.2f}\nIntercept={intercept:.2f}\nR²={r_squared:.2f}"
        textstr = f"R² = {r_squared:.2f}"
        ax.text(boxPositionX, boxPositionY, textstr, transform=ax.transAxes, fontsize=def_fontSize,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        # Labels, styles
        ax.set_xlabel(xlabel_str, fontsize=def_fontSize)
        ax.set_ylabel("Matric Potential (kPa)", fontsize=def_fontSize)
        ax.tick_params(labelsize=def_fontSize)
        plt.title(f'Matric Potential vs {xlabel_str}', fontsize=def_fontSize)
        plt.setp(ax.spines.values(), lw=1.5, color='black')
        plt.tight_layout()
        plt.show()
        
    except Exception:
        r_squared = np.nan
    
    return r_squared

def open_raster_clean(path, ref_shape=None):
    '''
    Reads raster with rasterio and replaces no data values with nan.
    '''
        
    if os.path.exists(path):
        with rasterio.open(path) as src:
            arr = src.read(1).astype('float32')
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            return arr, arr.shape
    else:
        if ref_shape is None:
            raise ValueError("Reference shape must be provided when raster is missing.")
    return np.full(ref_shape, np.nan, dtype='float32'), ref_shape

def shapiroWilks(array, pThreshold):
    try:
        valid_mask = (array != 0) & ~np.isnan(array)
        array_clean = array[valid_mask]
        stat, pval = stats.shapiro(array_clean)
    except Exception:
        return np.nan, np.nan, "n/a"
    
    if np.isnan(pval):
        ver = "n/a"
    elif pval < pThreshold:
        ver = "not normal"
    else:
        ver = "normal"
    return stat, pval, ver

def linearregression_spearman_pearson(arr1, arr2, label1, label2, year, binned=True):
    """
    model: "linear" 
    Returns: (x_means, y_means, y_pred_means, r2, gam_params, lin_params)
    """
    
    # nan_params = {"lam": np.nan, "edf": np.nan, "AIC": np.nan, "n": 0, "r2": np.nan, "noise_floor": np.nan}
    nan_lin = {"r2": np.nan, "r2_adj": np.nan, "slope": np.nan, "intercept": np.nan, "AIC": np.nan, "n": 0}
    nan_spearman = {"rho_spear": np.nan, "pval_spear": np.nan, "ci_low_spear": np.nan, "ci_high_spear": np.nan}
    nan_pearson = {"rho_pear": np.nan, "pval_pear": np.nan, "ci_low_pear": np.nan, "ci_high_pear": np.nan}
    empty = (np.array([]), np.array([]), np.array([]), nan_lin, nan_spearman, nan_pearson)
    if arr1 is None or arr2 is None:
        return empty
    x_data = arr1.flatten()
    y_data = arr2.flatten()
    if x_data.size == 0 or np.isnan(x_data).all():
        return empty
    valid_mask = (x_data != 0) & (y_data != 0) & ~np.isnan(x_data) & ~np.isnan(y_data)
    x_valid = x_data[valid_mask]
    y_valid = y_data[valid_mask]
    
    if x_valid.size < 10:
        print(f"{label1}-{label2} ({year}): too few points ({x_valid.size}), skipping.")
        return empty
    x2 = x_valid.reshape(-1, 1)
    n  = len(x_valid)
    # Linear Regression
    try:
        lin = LinearRegression().fit(x2, y_valid)
        y_pred_lin  = lin.predict(x2)
        r2_lin      = r2_score(y_valid, y_pred_lin)
        adj_r2_lin = 1 - (1 - r2_lin) * (n-1) / (n - 2 - 1)
        sse_lin     = np.sum((y_valid - y_pred_lin) ** 2)
        sigma2_lin  = sse_lin / n
        # Match pygam's AIC scale: -2 * full Gaussian log-likelihood + 2k
        aic_lin     = n * np.log(sigma2_lin) + n * np.log(2 * np.pi) + n + 2 * 2
        lin_params  = {
            "r2":        r2_lin,
            "r2_adj":    adj_r2_lin,
            "slope":     lin.coef_[0],
            "intercept": lin.intercept_,
            "AIC":       aic_lin,
            "n":         n,
        }
        print(f"{label1}-{label2} ({year}) LIN : R²={round(adj_r2_lin,3)}")
        
    except Exception as e:
        print(f"{label1}-{label2} ({year}): Linear failed - {e}")
        lin_params  = nan_lin
        y_pred_lin  = np.full(n, np.nan)
        
    # SPEARMAN  
    try:
        # 1D arrays only
        rho_spear, pval_spear = spearmanr(x_valid, y_valid)
        
        # # Pre-rank once (faster + correct)
        # xr = rankdata(x_valid)
        # yr = rankdata(y_valid)
        
        # boot strapping
        n_boot = 2000
        rng = np.random.default_rng()
        idx = rng.integers(0, len(x_valid), size=(n_boot, len(x_valid)))
        
        rhos_spear = np.array([np.corrcoef(rankdata(x_valid[i]), rankdata(y_valid[i]))[0, 1] for i in idx])
        ci_low_spear, ci_high_spear = np.percentile(rhos_spear, [2.5, 97.5])
        spearman_params = {"rho_spear": rho_spear,
                           "pval_spear": pval_spear, 
                           "ci_low_spear": ci_low_spear,
                           "ci_high_spear": ci_high_spear,
                           }
        print(f"{label1}-{label2} ({year}) SPEARMAN: rho={rho_spear:.3f}, p={pval_spear:.3g}, CI[{ci_low_spear:.2f}, {ci_high_spear:.2f}]")
    except Exception as e:
        print(f"{label1}-{label2} ({year}): Spearman failed - {e}")
        spearman_params = nan_spearman
        
    # PEARSON-R
    try:
        rho_pear, pval_pear = pearsonr(x_valid, y_valid)
        
        # boot strapping
        n_boot = 2000
        rng = np.random.default_rng()
        idx = rng.integers(0, len(x_valid), size=(n_boot, len(x_valid)))
        rhos_pear = np.array([np.corrcoef(x_valid[i], y_valid[i])[0, 1] for i in idx])
        ci_low_pear, ci_high_pear = np.percentile(rhos_pear, [2.5, 97.5])
        pearson_params = {"rho_pear": rho_pear, 
                          "pval_pear": pval_pear,
                          "ci_low_pear": ci_low_pear,
                          "ci_high_pear": ci_high_pear,
                          }
        print(f"{label1}-{label2} ({year}) PEARSON: R={rho_pear:.3f}, p={pval_pear:.3g}, CI[{ci_low_pear:.2f}, {ci_high_pear:.2f}]")
    except Exception as e:
        print(f"{label1}-{label2} ({year}): Pearson-R failed - {e}")
        pearson_params = nan_pearson
        
    # Binned output
    y_pred_active = y_pred_lin
    x_means, y_means, y_pred_means = [], [], []
    if binned:
        n_bins    = 10
        bin_edges = np.linspace(x_valid.min(), x_valid.max(), n_bins + 1)
        bin_idx   = np.digitize(x_valid, bin_edges)
        for i in range(1, n_bins + 1):
            mask = bin_idx == i
            if np.any(mask):
                x_means.append(np.mean(x_valid[mask]))
                y_means.append(np.mean(y_valid[mask]))
                y_pred_means.append(np.mean(y_pred_active[mask]))

    return (np.array(x_means), np.array(y_means), np.array(y_pred_means),
            lin_params, spearman_params, pearson_params)

def linearregression_loglike_spearman_pearson(arr1, arr2, label1, label2, year, binned=True):
    """
    model: "linear" 
    Returns: (x_means, y_means, y_pred_means, r2, gam_params, lin_params, log_params)
    """
    
    # nan_params = {"lam": np.nan, "edf": np.nan, "AIC": np.nan, "n": 0, "r2": np.nan, "noise_floor": np.nan}
    nan_lin = {"r2": np.nan, "r2_adj": np.nan, "slope": np.nan, "intercept": np.nan, "AIC": np.nan, "n": 0}
    nan_spearman = {"rho_spear": np.nan, "pval_spear": np.nan, "ci_low_spear": np.nan, "ci_high_spear": np.nan}
    nan_pearson = {"rho_pear": np.nan, "pval_pear": np.nan, "ci_low_pear": np.nan, "ci_high_pear": np.nan}
    nan_log = {"r2_log": np.nan, "r2_adj_log": np.nan, "slope_log": np.nan, "intercept_log": np.nan, "AIC_log": np.nan, "n_log": 0}
    empty = (np.array([]), np.array([]), np.array([]), nan_lin, nan_spearman, nan_pearson, nan_log)
    if arr1 is None or arr2 is None:
        return empty
    x_data = arr1.flatten()
    y_data = arr2.flatten()
    if x_data.size == 0 or np.isnan(x_data).all():
        return empty
    valid_mask = (x_data != 0) & (y_data != 0) & ~np.isnan(x_data) & ~np.isnan(y_data)
    x_valid = x_data[valid_mask]
    y_valid = y_data[valid_mask]
    
    if x_valid.size < 10:
        print(f"{label1}-{label2} ({year}): too few points ({x_valid.size}), skipping.")
        return empty
    x2 = x_valid.reshape(-1, 1)
    n  = len(x_valid)
    # Linear Regression
    try:
        lin = LinearRegression().fit(x2, y_valid)
        y_pred_lin  = lin.predict(x2)
        r2_lin      = r2_score(y_valid, y_pred_lin)
        adj_r2_lin = 1 - (1 - r2_lin) * (n-1) / (n - 2 - 1)
        sse_lin     = np.sum((y_valid - y_pred_lin) ** 2)
        sigma2_lin  = sse_lin / n
        # Match pygam's AIC scale: -2 * full Gaussian log-likelihood + 2k
        aic_lin     = n * np.log(sigma2_lin) + n * np.log(2 * np.pi) + n + 2 * 2
        lin_params  = {
            "r2":        r2_lin,
            "r2_adj":    adj_r2_lin,
            "slope":     lin.coef_[0],
            "intercept": lin.intercept_,
            "AIC":       aic_lin,
            "n":         n,
        }
        print(f"{label1}-{label2} ({year}) LIN : R²={round(adj_r2_lin,3)}")
        
    except Exception as e:
        print(f"{label1}-{label2} ({year}): Linear failed - {e}")
        lin_params  = nan_lin
        y_pred_lin  = np.full(n, np.nan)
        
    # SPEARMAN  
    try:
        # 1D arrays only
        rho_spear, pval_spear = spearmanr(x_valid, y_valid)
        
        # # Pre-rank once (faster + correct)
        # xr = rankdata(x_valid)
        # yr = rankdata(y_valid)
        
        # boot strapping
        n_boot = 2000
        rng = np.random.default_rng()
        idx = rng.integers(0, len(x_valid), size=(n_boot, len(x_valid)))
        
        rhos_spear = np.array([np.corrcoef(rankdata(x_valid[i]), rankdata(y_valid[i]))[0, 1] for i in idx])
        ci_low_spear, ci_high_spear = np.percentile(rhos_spear, [2.5, 97.5])
        spearman_params = {"rho_spear": rho_spear,
                           "pval_spear": pval_spear, 
                           "ci_low_spear": ci_low_spear,
                           "ci_high_spear": ci_high_spear,
                           }
        print(f"{label1}-{label2} ({year}) SPEARMAN: rho={rho_spear:.3f}, p={pval_spear:.3g}, CI[{ci_low_spear:.2f}, {ci_high_spear:.2f}]")
    except Exception as e:
        print(f"{label1}-{label2} ({year}): Spearman failed - {e}")
        spearman_params = nan_spearman
        
    # PEARSON-R
    try:
        rho_pear, pval_pear = pearsonr(x_valid, y_valid)
        
        # boot strapping
        n_boot = 2000
        rng = np.random.default_rng()
        idx = rng.integers(0, len(x_valid), size=(n_boot, len(x_valid)))
        rhos_pear = np.array([np.corrcoef(x_valid[i], y_valid[i])[0, 1] for i in idx])
        ci_low_pear, ci_high_pear = np.percentile(rhos_pear, [2.5, 97.5])
        pearson_params = {"rho_pear": rho_pear, 
                          "pval_pear": pval_pear,
                          "ci_low_pear": ci_low_pear,
                          "ci_high_pear": ci_high_pear,
                          }
        print(f"{label1}-{label2} ({year}) PEARSON: R={rho_pear:.3f}, p={pval_pear:.3g}, CI[{ci_low_pear:.2f}, {ci_high_pear:.2f}]")
    except Exception as e:
        print(f"{label1}-{label2} ({year}): Pearson-R failed - {e}")
        pearson_params = nan_pearson
        
    # Log-like Linear Regression
    try:
        # log transform: keep only positive values (log undefined for 0 or negative values)
        mask = (x2.flatten() > 0) & (y_valid > 0)
        x_log = np.log(x2[mask].reshape(-1, 1))
        y_log = np.log(y_valid[mask])
        n_log = len(y_log)
        lin_log = LinearRegression().fit(x_log, y_log)
        y_pred_lin_log  = lin_log.predict(x_log)
        r2_lin_log      = r2_score(y_log, y_pred_lin_log)
        adj_r2_lin_log = 1 - (1 - r2_lin_log) * (n-1) / (n - 2 - 1)
        sse_lin_log     = np.sum((y_log - y_pred_lin_log) ** 2)
        sigma2_lin_log  = sse_lin_log / n
        # Match pygam's AIC scale: -2 * full Gaussian log-likelihood + 2k
        aic_lin_log     = n * np.log(sigma2_lin_log) + n * np.log(2 * np.pi) + n + 2 * 2
        log_params  = {
            "r2_log":        r2_lin_log,
            "r2_adj_log":    adj_r2_lin_log,
            "slope_log":     lin_log.coef_[0],
            "intercept_log": lin_log.intercept_,
            "AIC_log":       aic_lin_log,
            "n_log":         n,
        }
        print(f"{label1}-{label2} ({year}) Log LIN : R²={round(adj_r2_lin,3)}")
        
    except Exception as e:
        print(f"{label1}-{label2} ({year}): Log Linear failed - {e}")
        log_params  = nan_log

        
    # Binned output
    y_pred_active = y_pred_lin
    x_means, y_means, y_pred_means = [], [], []
    if binned:
        n_bins    = 10
        bin_edges = np.linspace(x_valid.min(), x_valid.max(), n_bins + 1)
        bin_idx   = np.digitize(x_valid, bin_edges)
        for i in range(1, n_bins + 1):
            mask = bin_idx == i
            if np.any(mask):
                x_means.append(np.mean(x_valid[mask]))
                y_means.append(np.mean(y_valid[mask]))
                y_pred_means.append(np.mean(y_pred_active[mask]))

    return (np.array(x_means), np.array(y_means), np.array(y_pred_means),
            lin_params, spearman_params, pearson_params, log_params)

''' Calling functions '''

df_input_info = pd.read_csv(input_info_path)

all_results=[]
all_dates_rows = []
res_list_raw, res_list_residuals = [], [] # shapiro Wilks test results 
results_rows = [] 
for idx, row in df_input_info.iterrows():
    year = row['year']
    output_extension = row['output_extension']
    start_date = row['start_date']
    end_date = row['end_date']
    crop = row['crop']
    vNum = row['vNum']
    mean_days_in_month_for_timeperiod = row['mean_days_in_month_for_timeperiod']

    # Assign median (keep NaN if missing)
    median = row['median'] if not pd.isna(row['median']) else np.nan

    # matPot_start_date
    if pd.isna(row['matPot_start_date']):
        matPot_start_date = np.nan
    else:
        matPot_start_date = pd.to_datetime(row['matPot_start_date'], format='%m/%d/%Y')
        matPot_start_date = matPot_start_date.strftime('%Y-%m-%d')

    # matPot_end_date
    if pd.isna(row['matPot_end_date']):
        matPot_end_date = np.nan
    else:
        matPot_end_date = pd.to_datetime(row['matPot_end_date'], format='%m/%d/%Y')
        matPot_end_date = matPot_end_date.strftime('%Y-%m-%d')
        
    start_date = pd.to_datetime(start_date, format='%m/%d/%Y')
    start_date = start_date.strftime('%Y-%m-%d')
        
    end_date = pd.to_datetime(end_date, format='%m/%d/%Y')
    end_date = end_date.strftime('%Y-%m-%d')

    print(f'Processing: {start_date} - {end_date}')

    base_dir_input = os.path.join(base_dir, "output", f"gv_mz_{year}_golfcourse_{output_extension}")
    os.makedirs(base_dir_input, exist_ok = True)
    
    base_dir_output = os.path.join(base_dir,"output", f"gv_mz_{year}_golfcourse_{output_extension}")
    os.makedirs(base_dir_output, exist_ok = True)
    
    # Call data
    sr_ss = os.path.join(base_dir_input,  'topo', 'solarRadSS_topo_final.tif')
    gridMET = os.path.join(base_dir_input, 'topo', 'gridmet_topo_final.tif')
    ETensemble = os.path.join(base_dir_input, 'openet', 'openet_ensemble_final.tif')
    NDVI = os.path.join(base_dir_input,  'indices', f'L8_{year}_median_NDVI_final.tif')
    SAVI = os.path.join(base_dir_input, 'indices', f'L8_{year}_median_SAVI_final.tif')

    rasters = [NDVI, SAVI, ETensemble]
    
    # Study area and yield
    yield_path = os.path.join(base_dir, "input", "yield", f"{year}", f"yield_gridded_smooth_{year}_reproj_WgtAvg.tif")
    field_bound = os.path.join(base_dir, "input", "yield", f"{year}", "RC_field_bound.shp")
    matPot_folder_path = os.path.join(base_dir, "input", "matric_pot")
    
    ''' Calling Data ''' 
    with rasterio.open(yield_path) as ref:
        ref_shape = (ref.height, ref.width)
        
        # Solar Radiation
        sr_clean, ref_shape = open_raster_clean(sr_ss, ref_shape)
        # gridMET
        grid_clean, _ = open_raster_clean(gridMET, ref_shape)
        # ET ensemble
        et_clean, _ = open_raster_clean(ETensemble, ref_shape)
        # NDVI
        ndvi_clean, _ = open_raster_clean(NDVI, ref_shape)
        # SAVI
        savi_clean, _ = open_raster_clean(SAVI, ref_shape)
           
    yield_live = rasterio.open(yield_path)
    yield_live_arr = yield_live.read(1)
    
    start_date = f"{start_date}"
    end_date = f"{end_date}"
    
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)
    
    ref_raster = rasterio.open(yield_path)
    
    ''' Calculating Index ''' 
    
    _et = et_clean / mean_days_in_month_for_timeperiod
    _le = et2le(_et)
    _ndvi = ndvi_clean
    _savi = savi_clean
    _sr_rad = sr_clean
    _grid = np.nanmean(grid_clean)
    sr = (_sr_rad/ np.nanmean(_sr_rad)) * _grid
    le = _le
    c = 1
    d = 1
    
    wsi = zScores(c*sr - d*le)
    
    _sr = c*sr
    _le = d*le

    wsi_df = pd.DataFrame(wsi.flatten())
    wsi_path = write_array_to_raster(wsi,       ref_raster, ref_raster.transform, os.path.join(base_dir_output, f'wsi_v{vNum}.tif'))
    sr_path =  write_array_to_raster(_sr,       ref_raster, ref_raster.transform, os.path.join(base_dir_output, f'sr_{vNum}.tif'))
    le_path =  write_array_to_raster(_le,       ref_raster, ref_raster.transform, os.path.join(base_dir_output, f'le_{vNum}.tif'))
    ndvi_path =  write_array_to_raster(_ndvi,   ref_raster, ref_raster.transform, os.path.join(base_dir_output, f'ndvi_{vNum}.tif'))
    savi_path =  write_array_to_raster(_savi,   ref_raster, ref_raster.transform, os.path.join(base_dir_output, f'savi_{vNum}.tif'))

    ''' Statistical Analysis ''' 

    # clean yield as needed
    yield_masked = np.where(mask, yield_live_arr[:wsi.shape[0], :], np.nan)
    
    # check for normality
    p_threshold = 0.05
    check_list = [wsi,_sr, _le, ndvi_clean, savi_clean, yield_masked]
    names_list = ['wsi','sr','le','ndvi','savi', 'yield']
    
    for check, name in zip(check_list, names_list):
        checkStat, checkPval, checkVerdict = shapiroWilks(check, p_threshold)
        raw_name = "rawdata"
        checkRes = [raw_name, start_date, end_date, name, checkStat, checkPval, checkVerdict]
        res_list_raw.append(checkRes)
        
        model_residuals = linear_regression_residuals(check, yield_masked, 'arr', 'yield', year, plot=False, binned=False)
        resStat, resPval, resVerdict = shapiroWilks(model_residuals, p_threshold)
        raw_name = "residual"
        checkResiduals = [raw_name, start_date, end_date, name, resStat, resPval, resVerdict]
        res_list_residuals.append(checkResiduals)
        
    df_shapirowilks_rawData = pd.DataFrame(res_list_raw, columns=['data_type', 'start_date', 'end_date', 'name', 'stat', 'pval', 'verdict'])  
    df_shapirowilks_residuals = pd.DataFrame(res_list_residuals, columns=['data_type','start_date', 'end_date', 'name', 'stat', 'pval', 'verdict'])  
    df_shapirowilks_out = pd.concat( [df_shapirowilks_rawData, df_shapirowilks_residuals])
    df_shapirowilks_out.to_csv(os.path.join(base_dir, "output", "results_shapirowilks_LR.csv"), index=False)


    #### Linear Regression + Spearman Rank + Pearson
    wsi_yield_xmeans,  wsi_yield_ymeans,  wsi_yield_predMeans,  wsi_yield_linparams,  wsi_yield_spearman,   wsi_yield_pearson   = linearregression_spearman_pearson(wsi,        yield_masked, 'WSI', 'Yield', year, binned=True)
    sr_yield_xmeans,   sr_yield_ymeans,   sr_yield_predMeans,   sr_yield_linparams,   sr_yield_spearman,    sr_yield_pearson    = linearregression_spearman_pearson(_sr,        yield_masked, 'SR',    'Yield', year, binned=True)
    le_yield_xmeans,   le_yield_ymeans,   le_yield_predMeans,   le_yield_linparams,   le_yield_spearman,    le_yield_pearson    = linearregression_spearman_pearson(_le,        yield_masked, 'LE',    'Yield', year, binned=True)
    ndvi_yield_xmeans, ndvi_yield_ymeans, ndvi_yield_predMeans, ndvi_yield_linparams, ndvi_yield_spearman,  ndvi_yield_pearson  = linearregression_spearman_pearson(ndvi_clean, yield_masked, 'NDVI',  'Yield', year, binned=True)
    savi_yield_xmeans, savi_yield_ymeans, savi_yield_predMeans, savi_yield_linparams, savi_yield_spearman,  savi_yield_pearson  = linearregression_spearman_pearson(savi_clean, yield_masked, 'SAVI',  'Yield', year, binned=True)
    
    # #### Linear Regression + Spearman Rank + Pearson + Log Linear Regression
    # wsi_yield_xmeans,  wsi_yield_ymeans,  wsi_yield_predMeans,  wsi_yield_linparams,  wsi_yield_spearman,   wsi_yield_pearson, wsi_yield_linlog    = linearregression_loglike_spearman_pearson(wsi,        yield_masked, 'WSI', 'Yield', year, binned=True)
    # sr_yield_xmeans,   sr_yield_ymeans,   sr_yield_predMeans,   sr_yield_linparams,   sr_yield_spearman,    sr_yield_pearson, sr_yield_linlog      = linearregression_loglike_spearman_pearson(_sr,        yield_masked, 'SR',    'Yield', year, binned=True)
    # le_yield_xmeans,   le_yield_ymeans,   le_yield_predMeans,   le_yield_linparams,   le_yield_spearman,    le_yield_pearson, le_yield_linlog      = linearregression_loglike_spearman_pearson(_le,        yield_masked, 'LE',    'Yield', year, binned=True)
    # ndvi_yield_xmeans, ndvi_yield_ymeans, ndvi_yield_predMeans, ndvi_yield_linparams, ndvi_yield_spearman,  ndvi_yield_pearson, ndvi_yield_linlog  = linearregression_loglike_spearman_pearson(ndvi_clean, yield_masked, 'NDVI',  'Yield', year, binned=True)
    # savi_yield_xmeans, savi_yield_ymeans, savi_yield_predMeans, savi_yield_linparams, savi_yield_spearman,  savi_yield_pearson, savi_yield_linlog  = linearregression_loglike_spearman_pearson(savi_clean, yield_masked, 'SAVI',  'Yield', year, binned=True)
    
    
    #### data_dict for downstream plotting / array padding
    data_dict = {
        'wsi_yield_XMeans':    wsi_yield_xmeans,
        'wsi_yield_YMeans':    wsi_yield_ymeans,
        'wsi_yield_predMeans': wsi_yield_predMeans,
        'sr_yield_XMeans':     sr_yield_xmeans,
        'sr_yield_YMeans':     sr_yield_ymeans,
        'sr_yield_predMeans':  sr_yield_predMeans,
        'le_yield_XMeans':     le_yield_xmeans,
        'le_yield_YMeans':     le_yield_ymeans,
        'le_yield_predMeans':  le_yield_predMeans,
        'ndvi_yield_XMeans':   ndvi_yield_xmeans,
        'ndvi_yield_YMeans':   ndvi_yield_ymeans,
        'ndvi_yield_predMeans':ndvi_yield_predMeans,
        'savi_yield_XMeans':   savi_yield_xmeans,
        'savi_yield_YMeans':   savi_yield_ymeans,
        'savi_yield_predMeans':savi_yield_predMeans,
    }
    
    #### results row for CSV
    model_results = {
        'wsi':   (wsi_yield_linparams,  wsi_yield_spearman,  wsi_yield_pearson),
        'sr':    (sr_yield_linparams,   sr_yield_spearman,   sr_yield_pearson),
        'le':    (le_yield_linparams,   le_yield_spearman,   le_yield_pearson),
        'ndvi':  (ndvi_yield_linparams, ndvi_yield_spearman, ndvi_yield_pearson),
        'savi':  (savi_yield_linparams, savi_yield_spearman, savi_yield_pearson),
    }
    
    # #### results row for CSV - Including log-linear analysis
    # model_results = {
    #     'wsi':   (wsi_yield_linparams,  wsi_yield_spearman,  wsi_yield_pearson, wsi_yield_linlog),
    #     'sr':    (sr_yield_linparams,   sr_yield_spearman,   sr_yield_pearson, sr_yield_linlog),
    #     'le':    (le_yield_linparams,   le_yield_spearman,   le_yield_pearson, le_yield_linlog),
    #     'ndvi':  (ndvi_yield_linparams, ndvi_yield_spearman, ndvi_yield_pearson, ndvi_yield_linlog),
    #     'savi':  (savi_yield_linparams, savi_yield_spearman, savi_yield_pearson, savi_yield_linlog),
    # }
    
    row = {
        'start_date':       start_date,
        'end_date':         end_date,
        'year':             year,
        'output_extension': output_extension,
        'crop':             crop,
    }
    
    for key, (lin_p, spear_p, pear_p) in model_results.items():
    # for key, (lin_p, spear_p, pear_p, linlog_p) in model_results.items():
        # Linear columns: {key}_lin_r2, {key}_lin_slope, {key}_lin_intercept, {key}_lin_AIC, {key}_lin_n
        for param, val in lin_p.items():
            row[f"{key}_lin_{param}"] = val
        
        # Spearman columns
        for param, val in spear_p.items():
            row[f"{key}_{param}"] = val
            
        # Pearson columns
        for param, val in pear_p.items():
            row[f"{key}_{param}"] = val
            
        # # Log Transform Linear Regression
        # for param, val in linlog_p.items():
        #     row[f"{key}_{param}"] = val
    
    results_rows.append(row)
     
    data_dict = {'wsi_yield_XMeans':    wsi_yield_xmeans,
                 'wsi_yield_YMeans':    wsi_yield_ymeans,
                 'wsi_yield_predMeans': wsi_yield_predMeans,
                 'sr_yield_XMeans':     sr_yield_xmeans,
                 'sr_yield_YMeans':     sr_yield_ymeans,
                 'sr_yield_predMeans':  sr_yield_predMeans,
                 'le_yield_XMeans':     le_yield_xmeans,
                 'le_yield_YMeans':     le_yield_ymeans,
                 'le_yield_predMeans':  le_yield_predMeans,
                 'ndvi_yield_XMeans':   ndvi_yield_xmeans,
                 'ndvi_yield_YMeans':   ndvi_yield_ymeans,
                 'ndvi_yield_predMeans': ndvi_yield_predMeans,
                 'savi_yield_XMeans':   savi_yield_xmeans,
                 'savi_yield_YMeans':   savi_yield_ymeans,
                 'savi_yield_predMeans':savi_yield_predMeans,
                          }
    # padding arrays
    max_len = max(len(lst) for lst in data_dict.values())
    for key in data_dict:
        var = list(data_dict[key])
        if len(var) < max_len:
            var = var + [np.nan] * (max_len - len(var))
        data_dict[key] = var
            
    df_out = pd.DataFrame(data_dict)
    
    df_out_path3 = os.path.join(base_dir,"output", "wsi_bins")
    makedirs([df_out_path3])
    
    df_out_path = os.path.join(df_out_path3, f"bins_{start_date}_{end_date}_{ext}.csv")
    df_out.to_csv(df_out_path, index=False)
    
    X = wsi.flatten().reshape(-1, 1)
    y = yield_masked.flatten()
    
    mask_loc = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    
    X_clean = X[mask_loc]
    y_clean = y[mask_loc]
 
    array_dict_2 = {
        f"{year}_wsi": wsi.flatten(),
        f"{year}_sr": _sr.flatten(),
        f"{year}_le": _le.flatten(),
        f"{year}_ndvi": ndvi_clean.flatten(),
        f"{year}_savi": _savi.flatten(),
        f"{year}_yield": yield_masked.flatten(),
        }
    
    df_out_path1 = os.path.join(base_dir,"output", "wsi_arrays")
    makedirs([df_out_path1])
    df_arrays_2 = pd.DataFrame(array_dict_2)
    df_arrays_2_outPath = os.path.join(df_out_path1, f"rasterData_{crop}_{start_date}_{end_date}_{ext}.csv")
    df_arrays_2.to_csv(df_arrays_2_outPath, index=False)
    
    array_dict = {
        f'wsi_{start_date}_{end_date}': wsi.flatten(),
        f'sr_{start_date}_{end_date}': _sr.flatten(),
        f'le_{start_date}_{end_date}': _le.flatten(),
        f'ndvi_{start_date}_{end_date}': ndvi_clean.flatten(),
        f'savi_{start_date}_{end_date}': _savi.flatten(),
        f'yield_{start_date}_{end_date}': yield_masked.flatten(),
        }
    
    df_out_path2 = os.path.join(base_dir,"output", "wsi_timeslices")
    makedirs([df_out_path2])
    df_arrays = pd.DataFrame(array_dict)
    df_out_path = os.path.join(df_out_path2, f"{crop}_{start_date}_{end_date}_{ext}.csv")
    df_arrays.to_csv(df_out_path, index=False)
    
    ''' Cleaning Matric Potential Data ''' 
    if year == 2018:

        start_datetime = pd.to_datetime(matPot_start_date)
        end_datetime = pd.to_datetime(matPot_end_date)
        
        # Matric Potential Cleaning Parameters: 
        window_size = 8                     # moving average and stdev window size
        std_threshold = 1                   # outlier tolerance thershold
        reasonable_threshold_lower = -3000  # below threshold, values nan
        reasonable_threshold_upper = 0      # above threshold, values nan
        
        search_var = f"site*.csv"           # for 2018 data
        target_col_name = 'MatPot_2'        # for 2018 sensor data 45 cm bgs

        fig_data, field_names, stations, fig_data_full, medians, stress_days  = [], [], [], [], [], []
        # Call matric potential data as array
        for file in glob(os.path.join(matPot_folder_path, search_var)):
            if year == 2023:
              field_name = os.path.basename(file).split('_')[0] + " "+ os.path.basename(file).split('_')[1]
              field_names.append(field_name)
            else:
                field_name = os.path.basename(file).split('.')[0]
                field_names.append(field_name)
            field_names.append(field_name)
            station = os.path.basename(file).split('_')[-1].split('.')[0].split('e')[-1]
        
            df = pd.read_csv(file)                              # call file as DataFrame
            df['Timestamps'] = pd.to_datetime(df['Timestamps']) # convert times to datetime
            df = df.set_index(df['Timestamps'])                 # make time index
            df = df.drop(['Timestamps'], axis=1)                # drop old time column
            fig_data_full.append(df)
            df_new = df.copy()                                  # copy dataframe
             
            col = df_new[target_col_name].copy() # create copy of target column
            col.replace(-99999, np.nan, inplace=True) # replace any known error codes with nan values
            reasonable = col.where((col > reasonable_threshold_lower) & (col < reasonable_threshold_upper)) # define where values are within lower and upper bounds
          
            rolling_mean = reasonable.rolling(window=window_size, center=True, min_periods=1).mean() # calculte rolling mean based on reasonable values and windows size
            rolling_std = reasonable.rolling(window=window_size, center=True, min_periods=1).std() # calculte rolling stdev based on reasonable values and windows size
              
            rolling_mean = rolling_mean.reindex(col.index)  # force index of col and rolling mean to match
            rolling_std = rolling_std.reindex(col.index)    # force index of col and rolling stdev to match
                    
            rolling_mean = rolling_mean.fillna(0)   # fill any missing values
            rolling_std = rolling_std.fillna(0)     # fill any missing values
              
            outliers = (col - rolling_mean).abs() > std_threshold * rolling_std # identify outliers: abs(data - mean) > stdev threshold * stdev
              
            col[outliers] = np.nan # set outliers to nan
             
            df_new[f'{target_col_name}_updated'] = col # write out working column to dataframe
             
            # Aggregating to daily timestep
            if median == True:
                df_new = df_new.resample('D').median()
                title = "Median Daily"
            else:
                df_new = df_new
                title = "Hourly"
                  
            df_new = df_new.loc[(df_new.index >= start_datetime) & (df_new.index <= end_datetime)] # trim dates
            medians.append(np.nanmedian(df_new[f'{target_col_name}_updated']))
            
            stress_thres = -100 # kPa
            df_new['above_stress_thres'] = np.where(df_new[f'{target_col_name}_updated'] <= stress_thres, 1, 0)
            
            # need to group by day first and then sum
            stress_day = np.sum(df_new['above_stress_thres'])
            
            stress_days.append(stress_day)
            fig_data.append(df_new)
            stations.append(int(station))
            
            
        harvest_date_2018 = pd.to_datetime('2018-09-25')
        points_labels=['Site 4', 'Site 5', 'Site 6', 'Site 1', 'Site 2', 'Site 3']
        # reorder figure data
        desired_order = ['Site 1', 'Site 2', 'Site 3', 'Site 4', 'Site 5', 'Site 6']
        label_to_df = dict(zip(points_labels, fig_data))
        fig_data_reordered = [label_to_df[label] for label in desired_order]
        
        fontsize=25
        fig, axes = plt.subplots(2, 3, figsize=(15,8), tight_layout=True, sharex=True, sharey=True)
        axes = axes.flatten()
        for i, site_data in enumerate(fig_data_reordered):
            ax = axes[i]
            
            site_data['days_before_harvest'] = [(harvest_date_2018 - date).days for date in site_data.index]
            site_label = desired_order[i] 
            y_abs = np.abs(site_data[f'{target_col_name}_updated'])
            ax.plot(site_data['days_before_harvest'], y_abs, label='Matric Potential', color='cornflowerblue')
            ax.axhline(y=60, color='brown', linestyle='-', label='-60 kPa Stress Threshold', alpha = 0.5)
            ax.axhline(y=600, color='brown', linestyle='-', label='PWP Young Plants', alpha = 0.5)
            ax.axhline(y=1000, color='brown', linestyle='--', label='PWP Old Plants', alpha = 0.5)
            min_xlim = np.min(site_data['days_before_harvest'])
            max_xlim = np.max(site_data['days_before_harvest'])
            ax.set_title(desired_order[i], fontsize=fontsize)
            ax.set_xticks(np.linspace(min_xlim, max_xlim, 7, dtype=int))
            
            ax.tick_params(axis= "x", labelrotation=90, labelsize=fontsize)
            ax.tick_params(axis= "y", labelsize=fontsize)
            ax.set_ylim([10, 1500])
            ax.invert_yaxis()
            ax.set_yscale('log')
            ax.grid(which='minor', axis='both', alpha=0.2)
            
            if i == 0:
                ax.legend(loc='lower left', fontsize=fontsize-10)
                
        plt.gca().invert_xaxis()
        fig.supylabel('Matric Potential (-kPa)', fontsize=fontsize)
        fig.supxlabel('Days Before Harvest', fontsize=fontsize)
        plt.show()
        
        ''' Linear Regression - Matric Potential ''' 
    
        windowExtractFlag = False
        if windowExtractFlag == True:
            print("Window extracted.")
        else:
            print("Point location extracted.")
        
        wsi_matpot_r2 = MatPot_extractRas(wsi_path,     locs, medians, 'WSI',       def_fontSize= 28, boxPositionX=0.03, boxPositionY=0.08, windowExtractFlag=windowExtractFlag)
        sr_matpot_r2 = MatPot_extractRas(sr_path,       locs, medians, 'SR (W/m²)', def_fontSize= 28, boxPositionX=0.76, boxPositionY=0.08, windowExtractFlag=windowExtractFlag)
        le_matpot_r2 = MatPot_extractRas(le_path,       locs, medians, 'LE (W/m²)', def_fontSize= 28, boxPositionX=0.76, boxPositionY=0.08, windowExtractFlag=windowExtractFlag)
        ndvi_matpot_r2 = MatPot_extractRas(ndvi_path,   locs, medians, 'NDVI',      def_fontSize= 28, boxPositionX=0.03, boxPositionY=0.08, color='limegreen', windowExtractFlag=windowExtractFlag)
        savi_matpot_r2 = MatPot_extractRas(savi_path,   locs, medians, 'SAVI',      def_fontSize= 28, boxPositionX=0.03, boxPositionY=0.08, color='darkgreen', windowExtractFlag=windowExtractFlag)
        yield_matpot_r2 = MatPot_extractRas(yield_path, locs, medians, 'Yield',     def_fontSize= 28, boxPositionX=0.73, boxPositionY=0.08, color='orange', windowExtractFlag=windowExtractFlag)

        # add r2 values to dict
        r2_dict_matpot = {
            'wsi_matpot_r2':    wsi_matpot_r2,
            'sr_matpot_r2':     sr_matpot_r2,
            'le_matpot_r2':     le_matpot_r2,
            'ndvi_matpot_r2':   ndvi_matpot_r2,
            'savi_matpot_r2':   savi_matpot_r2,
            'yield_matpot_r2':  yield_matpot_r2,
            }
        
        matpot_r2_df = pd.DataFrame([r2_dict_matpot])
        
        wsi_bPlot = windowExtract_BoxPlot(locs,     wsi_path,   windowSize=1, missingDataval=0, prefix=None, fontsize=25, color='cornflowerblue', ylabel='WSI')
        SR_bPlot = windowExtract_BoxPlot(locs,      sr_path,    windowSize=1, missingDataval=0, prefix=None, fontsize=25, color='cornflowerblue', ylabel='SR (W/m²)')
        LH_bPlot = windowExtract_BoxPlot(locs,      le_path,    windowSize=1, missingDataval=0, prefix=None, fontsize=25, color='cornflowerblue', ylabel='LE (W/m²)')
        NDVI_bPlot = windowExtract_BoxPlot(locs,    ndvi_path,  windowSize=1, missingDataval=0, prefix=None, fontsize=25, color='limegreen', ylabel='NDVI')
        SAVI_bPlot = windowExtract_BoxPlot(locs,    savi_path,  windowSize=1, missingDataval=0, prefix=None, fontsize=25, color='darkgreen', ylabel='SAVI')
        yield_bPlot = windowExtract_BoxPlot(locs,   yield_path, windowSize=1, missingDataval=0, prefix=None, fontsize=25, color='orange', ylabel='Yield')
        
        data_list = [wsi_bPlot, SR_bPlot, LH_bPlot, NDVI_bPlot, SAVI_bPlot, yield_bPlot]
        
        n_sites = len(wsi_bPlot)
        n_vars = len(data_list)
        
        width = 0.05
        fig, axes = plt.subplots(
            nrows=len(data_list), ncols=1, sharex=True, figsize=(3, 10))
        
        datasets = [wsi_bPlot, SR_bPlot,LH_bPlot, NDVI_bPlot, SAVI_bPlot, yield_bPlot]
        labels = ['WSI', 'SR (W/m²)','LE (W/m²)', 'NDVI', 'SAVI', 'Yield',]
        colors = ['cornflowerblue', 'cornflowerblue', 'cornflowerblue', 'limegreen', 'darkgreen', 'orange']
        
        for ax, data, label, color in zip(axes, datasets, labels, colors):
            bp = ax.boxplot(data, patch_artist=True, showfliers=False)
            for box in bp['boxes']:
                box.set_facecolor(color)
            ax.set_ylabel(label)
            ax.grid(axis='y', alpha=0.3)
        num_sites = len(wsi_bPlot)
        axes[-1].set_xticks(range(1, num_sites + 1))
        axes[-1].set_xticklabels([f'Site {i+1}' for i in range(len(wsi_bPlot))])
        axes[-1].set_xlabel('Site')
        plt.tight_layout()
        plt.show()
        
        with rasterio.open(yield_path) as rasterFile:
            loc_shp = gpd.read_file(locs)
            loc_shp['geometry'] = [Point(xy) for xy in zip(loc_shp['Lon'], loc_shp['Lat'])]
            locationShp = gpd.GeoDataFrame(loc_shp, geometry='geometry', crs="EPSG:4326")
            # extract yield data at specific points
            test = NoWindowExtract(locationShp, rasterFile, windowSize=1, missingDataval=0, prefix=None)
    else:
        print("Matric potential data does not overlap with study period.")
        
        r2_dict_matpot = {
            'wsi_matpot_r2': np.nan,
            'sr_matpot_r2': np.nan,
            'le_matpot_r2': np.nan,
            'ndvi_matpot_r2': np.nan,
            'savi_matpot_r2': np.nan,
            'yield_matpot_r2': np.nan,
            }
        
        matpot_r2_df = pd.DataFrame([r2_dict_matpot])
    
df_timeseries = pd.DataFrame(results_rows)
df_timeseries.to_csv(os.path.join(base_dir, "output", f"results_linear_regression_r2_{ext}.csv"))

print("End of Script.")