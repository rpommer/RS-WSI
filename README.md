### **Exploring the Heterogenous Nature of Irrigated Winter Wheat and Potato Fields using an Experimental Water Stress Index and Vegetation Indices**

## 

#### **Abstract**

Growing competition for water has intensified the push for efficient irrigation management. As crop growth and soil properties vary at sub-field scales, irrigation management zones are developed and monitored using in situ data and vegetation indices (VIs). The latter has limited ability to capture plant water stress due to bare soil interference, crop-specific spectral differences, and a time lag between stress onset and phenological response to water stress. Remote sensing evapotranspiration (ET) data can supplement VIs to better characterize water stress. Here, we develop the Water Stress Index (WSI), a satellite-based index that identifies areas of water stress relative to the rest of the field over a selected period using an energy balance approach. The WSI is derived from ET and solar radiation data. An energy use proxy (latent heat flux from OpenET ensemble model) is subtracted from an energy availability proxy (LiDAR-derived and gridMET-modeled incoming solar radiation), producing a gridded map of relative water stress. The downscaled energy availability proxy provides field-scale variability that can be compared with high-resolution ET estimates more effectively than traditional, coarse-resolution products. The WSI is evaluated against in situ matric potential data and multi-season yield data. Results show the WSI consistently outperforms VIs and OpenET in describing end-of-season winter wheat yield. VIs outperform the WSI and OpenET for potatoes, which could be related to potatoes complex and rapid growth patterns. By leveraging energy balance concepts and accessible remote sensing products, the WSI offers an adaptable framework for identifying water-stressed areas in irrigated agricultural fields.


## 
#### Keywords:
Water Stress Index, Evapotranspiration, Solar Radiation, Irrigated Agriculture

## 
#### Publications
Pommerenke, R., Smith, R. G., Niemann, J. D., Hansen, N. C., & Rivera, L. (2026). Exploring the heterogenous nature of irrigated winter wheat and potato fields using an experimental water stress index and vegetation indices. Agricultural Water Management, 337, 110864. https://doi.org/10.1016/j.agwat.2026.110864


## 
#### Organizations
Colorado State University <br>
Brigham Young University <br>
METER Group, Inc. <br>

## 
#### Funding
METER Group, Inc. 

## 
#### Calculating the WSI
This github repository aims to provide instruction on how to access and process data to generate the WSI. Python scripts and instructions on how to organize data and run scripts are presented here. Data availability is described below. 

1) **Download scripts/set-up virtual environment**: To get started, download the scripts and yaml file from github and downloaded needed data from [Hydroshare](https://www.hydroshare.org/resource/0b522780d7464009a882f94c19daecce/)). Set-up a python environment using the yaml file (i.e., authors ran code using Anaconda and a python IDE that supported virtual environments).

2) **Download data**: Geospatial data is downloaded through the Google Earth Engine API or is avaiable through Hydroshare << LINK >> for specific files. To download GEE data, run the 0_gee_download.py script. Users will need to log-in to their Google Earth Engine account and obtain an authentication key which will need to be pasted into the python IDE prompt the first time the script is run. Data will download to Google Drive (text default folder: gem_valley_month_data). Download GEE data. 

3) **Setting-up file structure**: Sort downloaded files into the corresponding folders (see File Structure). A pre-formatted folder structure is included. These folders are empty when downloaded from github. For the script to run, files need to be placed in the correct folders (i.e., OpenET monthly mean rasters for 2015 >> input>  openET > 2015)

4) **Preprocessing Data**: As data is downloaded from different sources and has different coordinate reference systems and spatial resolutions, the raster data must be pre-processed to match. For this, a reference raster is used. This reference raster can be downloaded through Hydrobase and is specific to the study area the WSI is tested over. Run 1_preprocessing_data.py in python. A separate output folder for each timeslice (indicated in the input_info_months.csv file will be written to the Output folder.
   
6) **Calculating the WSI**: Now that the input data has been pre-processed, the WSI and vegetation indices can be calculated. Run 2_calculating_wsi.py. Outputs include rasters of vegetation indices, WSI, latent heat flux, and the WSI-specific solar radiation term. For details on how each of these terms are calculated, reference the publication. Additional outputs include Linear Regression, Spearman Rank Correlation Coefficient, and Pearson Correlation Coefficient results, analyzing the relationship between crop yield and the WSI, its components, and the vegetation indices (results_linear_regression_r2_golfcourseMonthly_LR_optimalwindows.csv). 

7) **Summarizing Results**: Summarizing results happens in two steps. For the first, run the 3_concat_results.py script. This will concatenate raster information (flattened arrays) for each timeslice/variable into an excel sheet. Each tab in the excel sheet represents a year of data (results_full_timeseries_LR_optimalwindows.csv). This information will be used in plots included in the publication. The second step involves manually copying raster information for the phenological windows that have the highest $Linear Regression adjusted R^2$ into a csv file (results_optimized_LR_optimalwindows.csv). 

8) **Generating Plots**: Run 4_plots.py

9) **Remote Sensing QC**: As an additional check, run the 5_rs_qc.py script to see pixel coverage for OpenET and Landsat 8 data over the study area for each monthly timeslice. This script will plot rasters for the study area over the growing season. Data is pulled from the Output folder. 

**List of Scripts**
* 0_gee_download.py
* 1_preprocessing_data.py
* 2_calculating_wsi.py
* 3_concat_results.py
* 4_plots.py
* 5_rs_qc.py

###### File Structure
```text
project\_root/
├── input/
│   ├── gridmet/
│   │   └── 2015 ... 2019
│   ├── landsat8/
│   │   └── 2015 ... 2019
│   ├── openET/
│   │   └── 2015 ... 2019
│   ├── yield/
│   │   └── 2015 ... 2019
│   ├── lidar/
│   ├── matric_pot/
│   ├── ref_rasters/
│   └── studyarea/
│
├── output/
│   └── <version>/
│       ├── indices/
│       ├── landsat8/
│       │   ├── bands/
│       │   └── indices/
│       ├── openet/
│       └── topo/
│
└── scripts/
    ├── 0_gee_download/
    ├── 1_preprocessing_data/
    ├── 2_calculating_wsi/
    ├── 3_concat_results/
    ├── 4_plots/
    └── 5_rs_qc/
```

###### Dependencies
* **Operating system:** Code associated with this github was run through Anaconda virtual environment using a python IDE. 
* **Conda environment**: The conda environment needed to run this script can be replicated using the .yaml file. 

## 
#### Data availability

|**Data**|**Source**|**Note**|
|-|-|-|
|Landsat 8|GEE|GEE API (see above)|
|OpenET|GEE|GEE API (see above)|
|gridMET|GEE|GEE API (see above)|
|LiDAR/Solar Radiation|BSU / Processed with ArcPro|[LiDAR Data](https://isu.maps.arcgis.com/apps/instant/media/index.html?appid=796225ca9c9f4c07a1288225ed1afd30) / [Processed Solar Radiation Data (HydroShare)](https://www.hydroshare.org/resource/0b522780d7464009a882f94c19daecce/)|
|Yield|Grower|Contact Author|
|Matric Potential Timeseries|Grower|Contact Author|
|Reference Raster(s)|GEE/processed with ArcPro or QGIS|[HydroShare](https://www.hydroshare.org/resource/0b522780d7464009a882f94c19daecce/)|



