'''
Author: Rahel
Date: March 4, 2026
Description:
    QC on remote sensing imagery.
'''

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
from sklearn.linear_model import LinearRegression
from rasterio.plot import show
import math

''' Script organization'''
if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()
base_dir = os.path.abspath(os.path.join(script_dir, ".."))

''' User input '''
# Landsat 8 folder
landsat8_path = os.path.join(base_dir, 'input', 'landsat8')

# Open ET folder
openET_path = os.path.join(base_dir, 'input', 'openET')

# CSV of start/end dates
date_path =  os.path.join(base_dir, 'input', 'start_end_dates.csv')

# Study area shapefile
studyArea_path = os.path.join(base_dir, 'input', "studyarea", "RC_field_bound.shp") 

''' Script '''
date_df = pd.read_csv(date_path)
date_df['startDate'] = pd.to_datetime(date_df['startDate'])
date_df['endDate'] = pd.to_datetime(date_df['endDate'])

studyArea = gpd.read_file(studyArea_path)
geometry = studyArea.geometry.values

year_list = [2015, 2016, 2017, 2018, 2019]

results = []

''' Landsat 8 '''      
for year in year_list:
    print(f"\nProcessing {year}...")
    images = []
    titles = []

    year_df = date_df[date_df['startDate'].dt.year == year]
    max_pixels = None
    for start, end in zip(year_df['startDate'], year_df['endDate']):
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        pattern = os.path.join(landsat8_path, str(year), f"landsat8_{start_str}_{end_str}_*_median_cloudless.tif")
        matches = glob(pattern)
        if not matches:
            print(f"Missing: {pattern}")
            images.append(None)
            titles.append(f"{start_str}\n{end_str}")
            continue
        raster_path = matches[0]
        with rasterio.open(raster_path) as src:
            clipped, transform = mask(src, geometry, crop=True, filled=False)
            data = clipped[0]
            inside = data[~data.mask]
            missing_pixels = np.count_nonzero(np.isnan(inside))
            images.append(data)
            titles.append(f"{start_str}\n{end_str}")
            if max_pixels is None:
                max_pixels = data.size
            valid_pixels = max_pixels - missing_pixels
            percent_coverage = (valid_pixels / max_pixels) * 100
            results.append({
                "year": year,
                "startDate": start,
                "endDate": end,
                "valid_pixels": valid_pixels,
                "percent_coverage": percent_coverage,
                "dataset": "landsat8"
            })

    if len(images) > 0:
        n = len(year_df)
        cols = 5
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        axes = axes.flatten()
        for i in range(n):
            img = images[i]
            if img is None:
                axes[i].text(0.5, 0.5, "Missing", ha='center', va='center', fontsize=9)
                axes[i].axis('off')
            else:
                axes[i].imshow(images[i], cmap="viridis")
                # axes[i].imshow(images[i], cmap="viridis", vmin=vmin, vmax=vmax)
                axes[i].set_title(titles[i], fontsize=8)
                axes[i].axis("off")
        for j in range(n, len(axes)):
            axes[j].axis("off")
        plt.suptitle(f"Landsat 8 Coverage - {year}", fontsize=14)
        plt.tight_layout()
        plt.show()

        
''' OpenET '''

for year in year_list:
    print(f"\nProcessing {year}...")
    images = []
    titles = []
    year_df = date_df[date_df['startDate'].dt.year == year]
    max_pixels = None
    for start, end in zip(year_df['startDate'], year_df['endDate']):
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        pattern = os.path.join(openET_path, str(year), f"GV_et_ensemble_mean_{start_str}_{end_str}_*_growingSeason.tif")
        matches = glob(pattern)
        if not matches:
            print(f"Missing: {pattern}")
            images.append(None)
            titles.append(f"{start_str}\n{end_str}")
            continue
        raster_path = matches[0]
        with rasterio.open(raster_path) as src:
            clipped, transform = mask(src, geometry, crop=True, filled=False)
            data = clipped[0]
            inside = data[~data.mask]
            missing_pixels = np.count_nonzero(np.isnan(inside))
            images.append(data)
            titles.append(f"{start_str}\n{end_str}")
            if max_pixels is None:
                max_pixels = data.size
            valid_pixels = max_pixels - missing_pixels
            percent_coverage = (valid_pixels / max_pixels) * 100
            results.append({
                "year": year,
                "startDate": start,
                "endDate": end,
                "valid_pixels": valid_pixels,
                "percent_coverage": percent_coverage,
                "dataset": "openET"
            })

    if len(images) > 0:
        n = len(year_df)
        cols = 5
        rows = math.ceil(n / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        axes = axes.flatten()
        for i in range(n):
            img = images[i]
            if img is None:
                axes[i].text(0.5, 0.5, "Missing", ha='center', va='center', fontsize=9)
                axes[i].axis('off')
            else:
                axes[i].imshow(images[i], cmap="viridis")
                # axes[i].imshow(images[i], cmap="viridis", vmin=vmin, vmax=vmax)
                axes[i].set_title(titles[i], fontsize=8)
                axes[i].axis("off")
        for j in range(n, len(axes)):
            axes[j].axis("off")
        plt.suptitle(f"OpenET Coverage - {year}", fontsize=14)
        plt.tight_layout()
        plt.show()
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(base_dir, "output", "qc_landsat8_openET.csv"), index=False)
    

    