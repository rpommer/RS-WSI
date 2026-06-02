'''
Author: Rahel
Date: April 2026
Description:
    Article figures:
        1 - QGIS
        2 - Powerpoint
        3 - QGIS
        4 - Python
        5 - QGIS/Python/Inkscape
        6 - QGIS/Python/Inkscape
        7 - QGIS/Python/Inkscape
        8 - Python
        9 - Python (2_calculating_wsi.py)/Inkscape
        S1 - Python
        S2 - Python
        S3 - Python (2_calculating_wsi.py)/Inkscape
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mticker
import os
from scipy.stats import spearmanr, rankdata, pearsonr
from pygam import LinearGAM, s, f
import scipy.stats as stats
from matplotlib.lines import Line2D
import matplotlib as mpl
import math
from sklearn.metrics import r2_score

''' Set-up '''
if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

base_dir = os.path.abspath(os.path.join(script_dir, ".."))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

''' User input '''
version = 'LR_optimalwindows'
main_color = 'cornflowerblue'

# Results for optimized timewindows - month for each growing season that explains most variance in yield data
path = os.path.join(base_dir, "output", f"results_optimized_{version}.xlsx")

# Results for all timeslices analyzed
csv_path = os.path.join(base_dir, "output", f"results_linear_regression_r2_golfcourseMonthly_{version}.csv")

# PRISM data file path
file_path = os.path.join(base_dir, 'input', 'prism', 'GV_RC_PRISM.csv')

# List of years in analysis
year_list = [2015, 2016, 2017, 2018, 2019]   

# Figure 6/7 specific lists
label_list_short = [ 'sr', 'le', 'wsi']
label_names_list_short = ['Solar Radiation (W/m²)', 'Latent Heat Flux (W/m²)', 'WSI']

# Figure 4, 5, 8, S2 specific lists
color_list = ['orange', 'royalblue', 'cornflowerblue', 'limegreen', 'darkgreen']
label_list = ['sr', 'le', 'wsi', 'ndvi', 'savi']
label_names_list = ["Solar Radiation (W/m²)", "Latent Heat Flux (W/m²)", "WSI", 'NDVI', 'SAVI']
    
''' Calling data '''

all_arrays = pd.read_excel(path, 'arrays')
data = pd.read_csv(csv_path)
df_prism = pd.read_csv(file_path)

''' Functions '''
''' Figure 4 '''
def compute_days(row):
    hdate = harvest_dates[row['year']]
    return (hdate - row['start_date']).days

def plot_r2(ax, year):
    year_data = data[data['year'] == year].copy()
    year_data = year_data.sort_values('days_before_harvest')
    for col, cfg in r2_configs.items():
        print(col)
        clean = year_data.dropna(subset=['days_before_harvest', col])
        missing = year_data[year_data['days_before_harvest'].notna() & year_data[col].isna()]
        ax.plot(clean['days_before_harvest'], clean[col],
                color=cfg['color'], linewidth=3, label=cfg['label'], linestyle=cfg['linestyle'])
        if not missing.empty and len(clean) >= 2:
            interp_y = np.interp(missing['days_before_harvest'], clean['days_before_harvest'], clean[col])
        elif not missing.empty:
            interp_y = [np.nan] * len(missing)
        if not missing.empty:
            ax.scatter(missing['days_before_harvest'], interp_y, marker='^', facecolors='none', edgecolors=cfg['color'], linewidths=2, s=100, zorder=5, label=None)

    cfg = span_config.get(year, {})
    if "span" in cfg:
        a, b = cfg["span"]
        ax.axvspan(min(a, b), max(a, b), alpha=0.10, color='cornflowerblue')
    if cfg.get("vline"):
        ax.axvline(cfg["vline"], color='black', linestyle='--', linewidth=2)
    ax.set_xlim(0, 160)
    ax.invert_xaxis()
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    xticks = np.arange(0, 181, 30)
    ax.set_xticks(xticks)
    ax.set_yticks(np.arange(0,1.1,0.5), minor=True)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_title(f"{cfg['crop']} {year}", fontsize=fontsize+2)

def plot_rho(ax, year):
    year_data = data[data['year'] == year].copy()
    year_data = year_data.sort_values('days_before_harvest')
    for key, cfg in spear_configs.items():
        clean = year_data.dropna(subset=[
            'days_before_harvest',
            f"{key}_rho_spear",
            f"{key}_ci_low_spear",
            f"{key}_ci_high_spear" ])
        
       
        # absolute value of spearman
        for col in [f"{key}_rho_spear", f"{key}_ci_low_spear", f"{key}_ci_high_spear"]:
            clean[col] = clean[col].abs()
            
        missing = year_data[year_data['days_before_harvest'].notna() & year_data[f"{key}_rho_spear"].isna()]
        if not missing.empty and len(clean) >= 2:
            interp_y = np.interp(missing['days_before_harvest'], clean['days_before_harvest'], clean[f"{key}_rho_spear"])
        elif not missing.empty:
            interp_y = [np.nan] * len(missing)
        if not missing.empty:
            ax.scatter(missing['days_before_harvest'], interp_y, marker='^', facecolors='none', edgecolors=cfg['color'], linewidths=2, s=100, zorder=5, label=None)
            
        ax.plot(clean['days_before_harvest'], clean[f"{key}_rho_spear"],
                color=cfg['color'], linewidth=3, label=cfg['label'], linestyle=cfg['linestyle'])
        ci_low = clean[f"{key}_ci_low_spear"].copy()
        ci_high = clean[f"{key}_ci_high_spear"].copy()
        clean[f"{key}_ci_low_spear"] = np.minimum(ci_low, ci_high)
        clean[f"{key}_ci_high_spear"] = np.maximum(ci_low, ci_high)
        ax.fill_between(clean['days_before_harvest'], clean[f"{key}_ci_low_spear"], clean[f"{key}_ci_high_spear"], color=cfg['color'], alpha=0.10)
    
    cfg = span_config.get(year, {})
    if "span" in cfg:
        a, b = cfg["span"]
        ax.axvspan(min(a, b), max(a, b), alpha=0.10, color='cornflowerblue')
    if cfg.get("vline"):
        ax.axvline(cfg["vline"], color='black', linestyle='--', linewidth=2)
    ax.set_xlim(0, 160)
    ax.invert_xaxis()
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    xticks = np.arange(0, 181, 30)
    ax.set_xticks(xticks, minor=True)
    ax.set_yticks(np.arange(0, 1.1, 0.5), minor=True)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_title(f"{cfg['crop']} {year}", fontsize=fontsize+2)
      
def plot_pear(ax, year):
    year_data = data[data['year'] == year].copy()
    year_data = year_data.sort_values('days_before_harvest')
    for key, cfg in pear_configs.items():
        clean = year_data.dropna(subset=[
            'days_before_harvest',
            f"{key}_rho_pear",
            f"{key}_ci_low_pear",
            f"{key}_ci_high_pear" ])
        
        # absolute value of spearman
        for col in [f"{key}_rho_pear", f"{key}_ci_low_pear", f"{key}_ci_high_pear"]:
            clean[col] = clean[col].abs()
        
        ci_low = clean[f"{key}_ci_low_pear"].copy()
        ci_high = clean[f"{key}_ci_high_pear"].copy()
        clean[f"{key}_ci_low_pear"] = np.minimum(ci_low, ci_high)
        clean[f"{key}_ci_high_pear"] = np.maximum(ci_low, ci_high)
        
        ax.fill_between(clean['days_before_harvest'],
                        clean[f"{key}_ci_low_pear"],
                        clean[f"{key}_ci_high_pear"], color=cfg['color'], alpha=0.10)
        
        missing = year_data[year_data['days_before_harvest'].notna() & year_data[f"{key}_rho_pear"].isna()]
        if not missing.empty and len(clean) >= 2:
            interp_y = np.interp(missing['days_before_harvest'], clean['days_before_harvest'], clean[f"{key}_rho_pear"])
        elif not missing.empty:
            interp_y = [np.nan] * len(missing)
        if not missing.empty:
            ax.scatter(missing['days_before_harvest'], interp_y, marker='^', facecolors='none', edgecolors=cfg['color'], linewidths=2, s=100, zorder=5, label=None)
        
        ax.plot(clean['days_before_harvest'],
                clean[f"{key}_rho_pear"],
                color=cfg['color'],linewidth=3, linestyle=cfg['linestyle'],label=cfg['label'])
    cfg = span_config.get(year, {})
    if "span" in cfg:
        a, b = cfg["span"]
        ax.axvspan(min(a, b), max(a, b), alpha=0.10, color='cornflowerblue')
    if cfg.get("vline"):
        ax.axvline(cfg["vline"], color='black', linestyle='--', linewidth=2)
    ax.set_xlim(0, 160)
    ax.invert_xaxis()
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    xticks = np.arange(0, 181, 30)
    ax.set_xticks(xticks, minor=True)
    ax.set_yticks(np.arange(0, 1.1, 0.5), minor=True)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_title(f"{cfg['crop']} {year}", fontsize=fontsize+2)

''' Figure 5 '''
def regression_histogram_grid(full_array, year_list, label1, label2,fontsize=20, xaxislabel='WSI',color='cornflowerblue', box_position=[0.04,0.25]):
    fig, axs = plt.subplots(2, 5, figsize=(18,7), sharey='row')
    # Flatten rows
    axs_reg = axs[0]   # top row
    axs_hist = axs[1]  # bottom row
    all_x = []
    for year in year_list:
        arr = np.array(full_array[f'{year}_{label1}'])
        arr = arr[~np.isnan(arr)]
        all_x.append(arr)
    all_x = np.concatenate(all_x)
    global_min = np.nanmin(all_x)
    global_max = np.nanmax(all_x)
    xticks = np.linspace(global_min, global_max, 4)
    def linear_regression_only(arr1, arr2):
        x_data = arr1.flatten()
        y_data = arr2.flatten()
        valid_mask = (x_data != 0) & (y_data != 0) & ~np.isnan(x_data) & ~np.isnan(y_data)
        x_valid = x_data[valid_mask]
        y_valid = y_data[valid_mask]
        x2 = x_valid.reshape(-1, 1)
        n  = len(x_valid)
        # Linear Regression
        try:
            lin = LinearRegression().fit(x2, y_valid)
            y_pred_lin  = lin.predict(x2)
            r2_lin      = r2_score(y_valid, y_pred_lin)
            r_squared_adj = 1 - (1 - r2_lin) * (n-1) / (n - 2 - 1)
            slope = lin.coef_[0]
            intercept = lin.intercept_
            # print(f"{label1}-{label2} ({year}) LIN : R²={round(r_squared_adj,3)}")
        except Exception as e:
            print(f"{label1}-{label2} ({year}): Linear failed - {e}")
        return lin, slope, intercept, r_squared_adj, x_valid, y_valid

    for i, year in enumerate(year_list):
        arr1 = np.array(full_array[f'{year}_{label1}'])
        arr2 = np.array(full_array[f'{year}_{label2}'])
        model_, slope_, intercept_, r2_, x_valid_, y_valid_ = linear_regression_only(arr1, arr2)

        ax = axs_reg[i]
        ax.scatter(x_valid_, y_valid_, s=50, color=color, alpha=0.5)
        x_line = np.linspace(x_valid_.min(), x_valid_.max(), 100).reshape(-1,1)
        y_line = model_.predict(x_line)
        ax.plot(x_line, y_line, color='black', linewidth=2, label='Linear Reg.')
        textstr = f'$R_{{adj}}^{{2}} = {r2_:.2f}$'
        props = dict(boxstyle='round, pad=0.2', facecolor='white', alpha=0.7)
        ax.text(box_position[0], box_position[1], textstr,
                transform=ax.transAxes,
                fontsize=fontsize*0.7,
                verticalalignment='top',
                linespacing=0.5,  
                bbox=props)
        ax.set_title(year, fontsize=fontsize)
        ax.set_xticks(xticks)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        ax.tick_params(axis='both', labelsize=fontsize)
        axh = axs_hist[i]
        arr_hist = np.array(full_array[f'{year}_{label1}'])
        axh.hist(arr_hist, color=main_color, edgecolor='black', linewidth=0.5)
        axh.set_ylim([0,225])
        axh.tick_params(axis='both', labelsize=fontsize)
        axh.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    handles, labels = axs_reg[0].get_legend_handles_labels()
    axs_reg[0].legend(handles, labels, loc='upper right', fontsize=fontsize*0.6, frameon=True)
    fig.supxlabel(xaxislabel, fontsize=fontsize)
    fig.text(-0.02, 0.72, label2.capitalize(), va='center', rotation='vertical', fontsize=fontsize)
    fig.text(-0.02, 0.28, 'Frequency', va='center', rotation='vertical', fontsize=fontsize)

    plt.tight_layout()
    plt.show()

''' Figure 6/7 '''
def histogram_subplots(full_array, year, label1, label_list, fontsize=15):
    fig, axs = plt.subplots(1, 3, figsize=(12,4), sharey=True)
    axs = axs.flatten() 
    for ax, label1, label_name in zip(axs, label1, label_list):
        arr1 = np.array(full_array[f'{year}_'+label1])
        ax.hist(arr1, color=main_color, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(f"{label_name}", fontsize=fontsize)
        ax.minorticks_on()
        ax.tick_params(axis='both', labelsize=fontsize)    
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_ylim([0,225])
    for ax in axs[len(year_list):]:
        ax.set_visible(False)
    fig.suptitle(f"{year}")
    fig.supylabel('Frequency', fontsize=fontsize)
    plt.tight_layout()
    plt.show()

''' Figure 8 '''
def LR_spearman_pearson_year_grid(full_array,years,label1,label2,label_names,color_list, figsize=(18,3.5)):
    crops = {
        2015: "Potatoes",
        2016: "Winter Wheat",
        2017: "Winter Wheat",
        2018: "Potatoes",
        2019: "Winter Wheat",
        }
    fontsize = 18
    nrows = len(years)
    ncols = len(label1)
    fig, axs = plt.subplots(nrows, ncols,figsize=(figsize[0], figsize[1] * nrows),sharey=True)
    
    axs = np.atleast_2d(axs)
    def lr_spear_pear_fit(x, y):
        x_data = x.flatten()
        y_data = y.flatten()
        mask = (x_data != 0) & (y_data != 0) & ~np.isnan(x_data) & ~np.isnan(y_data)
        x_valid = x_data[mask]
        y_valid = y_data[mask]
        x_2d = x_valid.reshape(-1, 1)
        n = len(x_valid)
        
        # Linear Regression
        lin = LinearRegression().fit(x_2d, y_valid)
        y_pred_lin = lin.predict(x_2d)
        r2_lin = r2_score(y_valid, y_pred_lin)
        r2_lin_adj = 1 - (1 - r2_lin) * (n-1) / (n - 2 - 1)
        
        # Spearman
        rho_spear, _ = spearmanr(x_valid, y_valid)
        
        # Pearson
        rho_pear, _ = pearsonr(x_valid, y_valid)

        return lin, r2_lin_adj, rho_pear, rho_spear, x_valid, y_valid
    
    for i, year in enumerate(years):
        for j, (target_label, label_name) in enumerate(zip(label1, label_names)):
            ax = axs[i, j]
            arr1 = np.array(full_array[f'{year}_{target_label}'])
            arr2 = np.array(full_array[f'{year}_{label2}'])
            lin_model, r2_lin, rho_pear, rho_spear, x_valid, y_valid = lr_spear_pear_fit(arr1, arr2)
  
            # Linear Regression
            x_line = np.linspace(x_valid.min(), x_valid.max(), 100).reshape(-1,1)
            y_line = lin_model.predict(x_line)
            ax.plot(x_line, y_line, color='black', label='Linear regression', linewidth=2.0)
            ax.scatter(x_valid, y_valid, s=40, alpha=0.5, color=color_list[j])
            
            ax.text(0.03, 0.30,
                    f"LR R²={r2_lin:.2f}\n"
                    f"R = {rho_pear:.2f}\n"
                    f"ρ = {rho_spear:.2f}",
                    transform=ax.transAxes, fontsize=fontsize, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            if i == nrows - 1:
                ax.set_xlabel(label_name, fontsize=fontsize)
            if j == 0:
                ax.set_ylabel(f"{crops.get(year, {})} {year}\nYield", fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()
    plt.show()

''' Figure S2 '''
def GAM_LR_year_grid(full_array,years,label1,label2,label_names,color_list, figsize=(18,3.5)):
    crops = {
        2015: "Potatoes",
        2016: "Winter Wheat",
        2017: "Winter Wheat",
        2018: "Potatoes",
        2019: "Winter Wheat",
        }
    fontsize = 18
    nrows = len(years)
    ncols = len(label1)
    fig, axs = plt.subplots(nrows, ncols,figsize=(figsize[0], figsize[1] * nrows),sharey=True)
    
    axs = np.atleast_2d(axs)
    def gam_lr_fit(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        mask = (x != 0) & (y != 0) & ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        x_2d = x.reshape(-1, 1)
        n = len(x_2d)
        
        lin = LinearRegression().fit(x_2d, y)
        r2_lin = lin.score(x.reshape(-1,1), y)
        r2_lin_adj = 1 - (1 - r2_lin) * (n-1) / (n - 2 - 1)
        
        nf = (np.quantile(x, 0.90) - np.quantile(x, 0.10)) / 20
        clusters = (x.max() - x.min()) / nf
        n_splines = max(4, min(8, math.ceil(clusters / 2)))
        gam = LinearGAM(s(0, n_splines = n_splines))
        gam.gridsearch(x_2d, y, lam = np.logspace(-3, 3, 60))
        
        r2_gam = gam.score(x_2d, y)
        edf = gam.statistics_["edof"]
        r2_gam_adj = 1 - (1 - r2_gam) * (n-1) / (n - edf - 1)
        return gam, r2_gam_adj, lin, r2_lin_adj, x, y
    
    for i, year in enumerate(years):
        for j, (target_label, label_name) in enumerate(zip(label1, label_names)):
            ax = axs[i, j]
            arr1 = np.array(full_array[f'{year}_{target_label}'])
            arr2 = np.array(full_array[f'{year}_{label2}'])
            gam_model, r2_gam, lin_model, r2_lin, x_valid, y_valid = gam_lr_fit(arr1, arr2)
            y_valid = arr2[(arr1 != 0) & (arr2 != 0) & ~np.isnan(arr1) & ~np.isnan(arr2)]
            # GAM 
            ax.scatter(x_valid, y_valid,s=40, alpha=0.5, color=color_list[j])
            # smooth curve
            x_line = np.linspace(x_valid.min(), x_valid.max(), 200).reshape(-1, 1)
            y_line = gam_model.predict(x_line)
            ax.plot(x_line, y_line, color_list[j], linewidth=2)
            ax.plot(x_line, y_line, 'black', linewidth=2, linestyle='dashed')
            
            # Linear Regression
            x_line = np.linspace(x_valid.min(), x_valid.max(), 100).reshape(-1,1)
            y_line = lin_model.predict(x_line)
            ax.plot(x_line, y_line, color='black', label='Linear regression', linewidth=2.0)
            
            ax.text(0.03, 0.22,f"GAM Adj. R²={r2_gam:.2f} \n"
                    f"LR Adj. R²={r2_lin:.2f}"
                    ,transform=ax.transAxes, fontsize=fontsize, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            if i == nrows - 1:
                ax.set_xlabel(label_name, fontsize=fontsize)
            if j == 0:
                ax.set_ylabel(f"{crops.get(year, {})} {year}\nYield", fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize)
    plt.tight_layout()
    plt.show()

''' Figure 4 - Statistical Timeseries '''
data = data[data['output_extension'].astype(str).str.contains('slice', case=False, na=False)].copy()
data = data.replace(" ", np.nan)

harvest_dates = {
    2015: pd.Timestamp("2015-09-29"),
    2016: pd.Timestamp("2016-08-08"),
    2017: pd.Timestamp("2017-08-28"),
    2018: pd.Timestamp("2018-09-25"),
    2019: pd.Timestamp("2019-08-24"),
    }

data['start_date'] = pd.to_datetime(data['start_date'])
data['year'] = data['start_date'].dt.year
data['days_before_harvest'] = data.apply(compute_days, axis=1)

# Ensure numeric
num_cols = [c for c in data.columns if "_rho" in c or "_ci_low" in c or "_ci_high" in c]
data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')

pear_configs = {
    'wsi':  {'color': 'cornflowerblue', 'label': 'WSI',  'linestyle': '-'     },
    'le':   {'color': 'navy',           'label': 'LE',   'linestyle': 'dotted'},
    'sr':   {'color': 'orange',         'label': 'SR',   'linestyle': 'dashed'},
    'ndvi': {'color': 'limegreen',      'label': 'NDVI', 'linestyle': 'dashed'},
    'savi': {'color': 'darkgreen',      'label': 'SAVI', 'linestyle': 'dashed'},
    }

spear_configs = {
    'wsi':  {'color': 'cornflowerblue', 'label': 'WSI',  'linestyle': '-'     },
    'le':   {'color': 'navy',           'label': 'LE',   'linestyle': 'dotted'},
    'sr':   {'color': 'orange',         'label': 'SR',   'linestyle': 'dashed'},
    'ndvi': {'color': 'limegreen',      'label': 'NDVI', 'linestyle': 'dashed'},
    'savi': {'color': 'darkgreen',      'label': 'SAVI', 'linestyle': 'dashed'},
    }

r2_configs = {
    'wsi_lin_r2_adj':  {'color': 'cornflowerblue', 'label': 'WSI',  'linestyle': '-'     },
    'le_lin_r2_adj':   {'color': 'navy',           'label': 'LE',   'linestyle': 'dotted'},
    'sr_lin_r2_adj':   {'color': 'orange',         'label': 'SR',   'linestyle': 'dashed'},
    'ndvi_lin_r2_adj': {'color': 'limegreen',      'label': 'NDVI', 'linestyle': 'dashed'},
    'savi_lin_r2_adj': {'color': 'darkgreen',      'label': 'SAVI', 'linestyle': 'dashed'},
}

span_config = {
    2015: {"span": (60, 90),  "vline": 137,  "crop": "Potatoes"    }, # span: July
    2016: {"span": (69, 99),  "vline": None, "crop": "Winter Wheat"},
    2017: {"span": (59, 88),  "vline": None, "crop": "Winter Wheat"},
    2018: {"span": (25, 55),  "vline": 119,  "crop": "Potatoes"    }, # span: August
    2019: {"span": (24, 54),  "vline": None, "crop": "Winter Wheat"},}


fontsize=20
fig, axes = plt.subplots(nrows=len(year_list),ncols=3,figsize=(18, 20),sharex=True)
fig.subplots_adjust(top=0.95)
fig.text(0.24, 0.98, "(a) LR Adj. R² vs DBH",
         ha='center', va='top', fontsize=fontsize)
fig.text(0.50, 0.98, "(b) Spearman |ρ| vs DBH",
         ha='center', va='top', fontsize=fontsize)
fig.text(0.77, 0.98, "(c) Pearson R vs DBH",
         ha='center', va='top', fontsize=fontsize)

for i, year in enumerate(year_list):
    # Only first row gets titles
    plot_r2(axes[i, 0], year)
    plot_rho(axes[i, 1], year)
    plot_pear(axes[i, 2], year)
for ax in axes[:, 0]:
    ax.set_ylabel("Adj. R²", fontsize=fontsize)
for ax in axes[:, 1]:
    ax.set_ylabel("Spearman ρ (±95% CI)", fontsize=fontsize)
for ax in axes[:, 2]:
    ax.set_ylabel("Pearson R (±95% CI)", fontsize=fontsize)
for ax in axes[-1, :]:
    ax.set_xlabel("Days Before Harvest", fontsize=fontsize)
  
missing_handle = Line2D([0], [0], marker='^', color='gray',markerfacecolor='none', markersize=8,linewidth=0, label='Missing Data')
handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(missing_handle)
labels.append('Missing data')
axes[0, 0].legend(handles, labels, loc='upper left', fontsize=fontsize-2, frameon=True, ncols=2)

''' Figure 5 - Linear Regression Grid ''' 
regression_histogram_grid(all_arrays, year_list, label1='wsi', label2='yield', fontsize=25, xaxislabel='WSI', box_position=[0.04, 0.22])

''' Fig. 6: Histograms SR, LE, WSI (2016) '''
histogram_subplots(all_arrays, 2016, label_list_short, label_names_list_short, fontsize=19)

''' Fig. 7: Histograms SR, LE, WSI (2018) '''
histogram_subplots(all_arrays, 2018, label_list_short, label_names_list_short, fontsize=19)

''' Fig 8 - Scatter plot with fitted Linaer Regression, Pearson R, Spearman Rho '''
LR_spearman_pearson_year_grid(all_arrays, years=year_list, label1=label_list, label2='yield', label_names = label_names_list, color_list = color_list)

''' Figure S1 - PRISM Timeseries '''
mpl.rcParams["font.size"] = 14
df_p = df_prism[['month', 'ppt', 'tdmean', 'tmax', 'tmean', 'tmin','vpdmax', 'vpdmin']]
df_p['Datetime'] = pd.to_datetime(df_p['month'], format='%Y-%m')
# change index to Date
df_p.set_index(df_p['Datetime'], inplace=True)

# for each year, take mean of precip
df_p['year'] = df_p['Datetime'].dt.year
df_p['month'] = df_p['Datetime'].dt.month

### PRISM Plot: Monthly Comparison
data_2015 = df_p.loc[(df_p.index >= pd.to_datetime('2015-01-01')) & (df_p.index <= pd.to_datetime('2015-12-01'))]
data_2016 = df_p.loc[(df_p.index >= pd.to_datetime('2016-01-01')) & (df_p.index <= pd.to_datetime('2016-12-01'))]
data_2017 = df_p.loc[(df_p.index >= pd.to_datetime('2017-01-01')) & (df_p.index <= pd.to_datetime('2017-12-01'))]
data_2018 = df_p.loc[(df_p.index >= pd.to_datetime('2018-01-01')) & (df_p.index <= pd.to_datetime('2018-12-01'))]
data_2019 = df_p.loc[(df_p.index >= pd.to_datetime('2019-01-01')) & (df_p.index <= pd.to_datetime('2019-12-01'))]

data_list = [data_2015, data_2016, data_2017, data_2018, data_2019]
color_list_prism = ['green', 'limegreen','cornflowerblue', 'blue', 'navy']
months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec', '']
hatching = ['///', '', '', '///', '']
year_list = [2015, 2016, 2017, 2018, 2019]
linestyles = ['solid', 'dotted', 'dashed', 'dashdot', 'solid']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[2, 1], sharex=False)
n = len(data_list)
width = 0.5/n
x = np.arange(13)
for i, (data, color, linestyle, year) in enumerate(zip(data_list, color_list_prism, linestyles, year_list)): 
    label_name = data['year'].iloc[0]     
    ax1.plot(data['month'], data['tmean'], color=color, linewidth=2, marker='.', linestyle=linestyle, label=year)
ax1.set_ylabel('Mean. Temperature (°C)')
ax1.set_xticks(range(len(months)), labels = months)
ax1.legend(loc='upper right')
for i, (data, color, hatch) in enumerate(zip(data_list, color_list_prism, hatching)):
    label_name = data['year'].iloc[0]
    offset = (i - n/2) * width + width/2
    data = data.sort_values('month')
    ax2.bar(data['month'] + offset, data['ppt']*2.54, width=width, color=color, label=str(label_name), hatch = hatch, edgecolor='white')
ax2.set_xticks(range(len(months)), labels = months)
ax2.set_ylabel('Precipitation (cm)')
ax2.legend(title='Year', loc='upper right', fontsize=10)
plt.xlabel('Month', fontsize=fontsize)
plt.tight_layout()
plt.show()

''' Figure S2- Scatter plots with GAM and Linear Regression''' 
GAM_LR_year_grid(all_arrays, years=year_list, label1=label_list, label2='yield', label_names = label_names_list, color_list = color_list)

