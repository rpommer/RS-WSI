'''
AUTHOR: rpommer
DATE: August 13, 2025
DESCRIPTION: 
    Pre-processing landsat8, OpenET, gridMET, and LiDAR topographic input data.
    - Takes multi-band raster and extracts band information to calculate vegetation indices (VIs)
    - Writes out all rasters, clipped to study area and in same coordinate reference system.   
'''

import glob
import os
from osgeo import gdal
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling


''' Setting up directories '''
if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

base_dir = os.path.abspath(os.path.join(script_dir, ".."))
base_dir_input = os.path.join(base_dir, "input")
os.makedirs(base_dir_input, exist_ok = True)

''' user input '''
# CSV with start/end date information
input_info_path = os.path.join(base_dir, "input", "input_info_months_optimalwindows.csv")

# Reference raster path (.tif) --> Transformation and Resolution are pulled from this raster.  
reference_raster = os.path.join(base_dir_input, "ref_rasters", "landsat8_2023_gemval_ref_ras_full.tif") 

# Study Area Polygon path (.shp)
field_boundary = os.path.join(base_dir_input, "studyarea", "RC_field_bound.shp") 

''' Calling data '''
df_input_info = pd.read_csv(input_info_path)

# No Data Value
nodatavalue = 255

''' functions '''

def makedirs(directory_list):    
    """    Make directory (if not exists) from a list of directory.    
    :param directory_list: A list of directories to create.    
    :return: None.    """    
    for directory in directory_list:        
        if not os.path.exists(directory):            
            os.makedirs(directory)
            
def get_var_name(var):
    """ Convert the name of a variable to a string. 
    : parm: variable
    : return: string
    """
    import inspect
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def read_raster_arr_object(raster_file, rasterio_obj=False, band=1, get_file=True, change_dtype=True):
    """
    Function provided by Fahim 9/30/2024. 
    
    Get raster array and raster file.

    :param raster_file: Input raster filepath.
    :param rasterio_obj: Set True if raster_file is a rasterio object.
    :param band: Selected band to read. Default set to 1.
    :param get_file: Set to False if raster file is not required.
    :param change_dtype: Set to True if want to change raster data type to float. Default set to True.

    :return: Raster numpy array and rasterio object file (get_file=True, rasterio_obj=False).
    """
    if not os.path.exists(raster_file):
        print(f"Issue with path: {raster_file}")
    
    if not rasterio_obj:
        raster_file = rasterio.open(raster_file)
    else:
        get_file = False
    raster_arr = raster_file.read(band)

    if change_dtype:
        raster_arr = raster_arr.astype(np.float32)
        if raster_file.nodata:
            raster_arr[np.isclose(raster_arr, raster_file.nodata)] = np.nan
    if get_file:
        return raster_arr, raster_file
    else:
        return raster_arr

def clip_resample_reproject_raster(input_raster, input_shape, output_raster_dir,
                                   keyword=' ', raster_name=None,
                                   clip=False, resample=False, clip_and_resample=True,
                                   targetaligned=True, resample_algorithm='near',
                                   resolution=None,
                                   crs='EPSG:4269', output_datatype=gdal.GDT_Float32,
                                   use_ref_width_height=False, ref_raster=reference_raster, no_data_value=nodatavalue):
    """
    Clips, resamples, reprojects a given raster using input shapefile, resolution, and crs.

    ** If resolution is None, must provide a left_zone_ref_raster. One of resolution and left_zone_ref_raster must be available.

    :param input_raster: Input raster filepath.
    :param input_shape: Input shape filepath. Set to None when resample=True.
    :param keyword: 'str' keyword to attach in front of processed raster. Default set to ' '.
                    ** Only works when raster_name = None.
    :param raster_name: Output raster name. Default set to None to set raster name from the input raster.
    :param output_raster_dir: Output directory filepath.
    :param clip: Set to True to clip only. When True, resample and clip_and_resample should be False. resolution can be
                 used as None (Default) in this case.
    :param resample: Set to True to resample only. When True, clip and clip_and_resample should be False. Need to set
                     a resolution.
    :param clip_and_resample: Set to True to both clip and resample. When True, clip and resample should be False. Need
                              to set a resolution. Set input_shape to None.
    :param targetaligned: Set to False if pixels don't need to be aligned in case of clip_and_resample=True. Look into
                          the result when using this to be sure about expected outcome.
    :param resample_algorithm: Resample algorithm to use in resampling. Can take near/bilinear/average/mode/max/min/cubic etc.
                               Default is 'near'.
    :param resolution: Output raster resolution. Default set to None (for clip=True). For other purposes, use specific
                       resolution value (for example, 0.02000000000000000389 deg).
    :param crs: Output raster projection. Default set to 'EPSG:4269' (NAD83).
    :param output_datatype: Output data type. Default set to gdal.GDT_Float32.
    :param use_ref_width_height: Set to True to use reference raster's widht+height for resampling/clipping,
                                 instead of a particular assigned resolution. 'resolution' can be set to None.
    :param ref_raster: Filepath of reference raster to be used for assigning processed raster's width+height.

    :return: Processed raster filepath.
    """
    if raster_name is None:  # if raster_name is None will set raster name from the input raster.
        raster_name = os.path.basename(input_raster)
        output_raster_name = keyword + '_' + raster_name
        if keyword == ' ':
            output_raster_name = raster_name
    else:  # to assign assigned raster_name. If None, will figure out raster_name using the first conditional if.
        if '.tif' not in raster_name:
            raster_name = raster_name + '.tif'
        output_raster_name = raster_name

    # creating output directory
    makedirs([output_raster_dir])
    output_filepath = os.path.join(output_raster_dir, output_raster_name)

    # opening input raster
    raster_file = gdal.Open(input_raster)

    if clip:  # set resample, clip_and_resample = False
        # resolution argument can be None in clip operation
        if use_ref_width_height:
            # have to provide a reference raster
            # resolution can be set to None
            ref_arr = read_raster_arr_object(ref_raster, get_file=False)
            height, width = ref_arr.shape
            processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                       targetAlignedPixels=False, width=width, height=height,
                                       cutlineDSName=input_shape,
                                       cropToCutline=True, dstNodata=no_data_value, outputType=output_datatype)
        else:
            _, xres, _, _, _, yres = raster_file.GetGeoTransform()
            processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                       targetAlignedPixels=targetaligned, xRes=xres, yRes=yres,
                                       cutlineDSName=input_shape, cropToCutline=True, dstNodata=no_data_value,
                                       outputType=output_datatype)
            
        if processed_data is None:
            raise RuntimeError(f"gdal.Warp failed when writing to {output_filepath}")

    elif resample:  # set clip, clip_and_resample = False
        if use_ref_width_height:
            # have to provide a reference raster
            # resolution can be set to None
            # input_shape can be set to None
            ref_arr = read_raster_arr_object(ref_raster, get_file=False)
            height, width = ref_arr.shape
            processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                       targetAlignedPixels=False, width=width, height=height,
                                       dstNodata=no_data_value, resampleAlg=resample_algorithm,
                                       outputType=output_datatype)
        else:
            # have to provide a resolution value in argument
            # input_shape can be set to None
            processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                       targetAlignedPixels=targetaligned, xRes=resolution, yRes=resolution,
                                       dstNodata=no_data_value, resampleAlg=resample_algorithm,
                                       outputType=output_datatype)
            
        if processed_data is None:
            raise RuntimeError(f"gdal.Warp failed when writing to {output_filepath}")

    elif clip_and_resample:  # set clip=False, resample = False
        if use_ref_width_height:
            # have to provide a reference raster
            # resolution can be set to None
            ref_arr = read_raster_arr_object(ref_raster, get_file=False)
            height, width = ref_arr.shape
            processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                       targetAlignedPixels=False, width=width, height=height,
                                       cutlineDSName=input_shape, cropToCutline=True, dstNodata=no_data_value,
                                       resampleAlg=resample_algorithm, outputType=output_datatype)
        else:
            # argument must have input_shape and resolution value
            processed_data = gdal.Warp(destNameOrDestDS=output_filepath, srcDSOrSrcDSTab=raster_file, dstSRS=crs,
                                       targetAlignedPixels=targetaligned, xRes=resolution, yRes=resolution,
                                       cutlineDSName=input_shape, cropToCutline=True, dstNodata=no_data_value,
                                       resampleAlg=resample_algorithm, outputType=output_datatype)
            
            if processed_data is None:
                raise RuntimeError(f"gdal.Warp failed when writing to {output_filepath}")
        
    del processed_data

    return output_filepath

def write_array_to_raster(raster_arr, raster_file, transform, output_path, dtype=None,
                          ref_file=None, nodata=nodatavalue):
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
        with rasterio.open(ref_file) as ref:
            raster_file = ref
            transform = ref.transform
            crs = ref.crs
            count = ref.count
    else:
        crs = raster_file.crs
        count = raster_file.count
        
    if os.path.exists(output_path):
        os.remove(output_path)
        
    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=raster_arr.shape[0],
            width=raster_arr.shape[1],
            dtype=dtype,
            count=count,
            crs=crs,
            transform=transform,
            nodata=nodata
    ) as dst:
        dst.write(raster_arr, count)

    return output_path

def mask_rescale_write2raster(input_file_path, filter_file_path, out_path, tolerance_value=1e-6, no_data_value=nodatavalue, rescale=False, filter_file_path_2 = None):
    '''

    Parameters
    ----------
    input_file_path : string
        Raster that will be masked. Path to raster (.tif)
    filter_file_path : string
        Raster that will be masked. Path to raster (.tif)
    out_path : string
        Path to output folder and file for final raster to be written out to. Has .tif extension.
    tolerance_value : float, optional
        Tolerance (in degrees) for which mask is applied. The default is 1e-6.
    no_data_value : float, optional
        The default is nodatavalue.
    rescale : Flag, optional
        When True, data will be rescaled. When False, data will keep original range. The default is True.

    Returns
    -------
    dst : raster
        Masked (rescaled) raster file.

    '''
    with rasterio.open(filter_file_path) as filt_file:
        filt = filt_file.read(1)
        filt2 = filt
        
        if filter_file_path_2:
            filt_file_2 = rasterio.open(filter_file_path_2)
            filt2 = filt_file_2.read(1)
        
        # Open raster
        with rasterio.open(input_file_path, 'r') as index:
            index_data = index.read(1)
            
            # Reproject index_data to match the crop_mask dimensions and transform
            reprojected_data = np.empty_like(index_data, dtype=index_data.dtype) # empty raster
            
            rasterio.warp.reproject(
                        source=index_data,
                        destination=reprojected_data,
                        src_transform=index.transform,
                        src_crs=index.crs,
                        dst_transform=filt_file.transform,
                        dst_crs=filt_file.crs,
                        dst_resolution= (filt_file.transform[0], abs(filt_file.transform[4])),
                        resampling=Resampling.nearest
                    )
        
        # Create binary mask based on irrigated areas
        tolerance = tolerance_value
        filt_data_condition = np.abs(filt - no_data_value) > tolerance
        filt_data_condition_2 = np.abs(filt2 - no_data_value) > tolerance
                    
        binary_mask = filt_data_condition & filt_data_condition_2
        
        # Apply mask to reprojected data
        masked_data = np.where(binary_mask, reprojected_data, no_data_value)
        masked_data = np.where(masked_data == no_data_value, np.nan, masked_data) # remove nan values from background
        masked_data = np.ma.array(masked_data, mask=np.isnan(masked_data))
        
        if rescale: 
            # Rescale data between [-1, 1]
            masked_data = (masked_data - np.mean(masked_data))/np.std(masked_data)
            # masked_data = 2.0 * (masked_data - np.min(masked_data)) / np.ptp(masked_data) - 1
            
        else: 
            masked_data = masked_data
        
        # Fill masked data for writing to the raster
        masked_data_filled = np.ma.filled(masked_data, fill_value=no_data_value).astype(index_data.dtype)

        # Write the masked and rescaled data to a new raster with the crop mask's transform and CRS
        with rasterio.open(out_path, 'w',
                           driver='GTiff',
                           height=filt_file.height,
                           width=filt_file.width,
                           count=1,
                           dtype=masked_data_filled.dtype,
                           crs=filt_file.crs,
                           transform=filt_file.transform,
                           nodata=no_data_value
                    ) as dst:
                    dst.write(masked_data_filled, 1)
                
                    
        return masked_data_filled, out_path
    
''' end functions '''

for idx, row in df_input_info.iterrows():
    year = row['year']
    output_extension = row['output_extension']
    start_date = row['start_date']
    end_date = row['end_date']
    crop = row['crop']
    vNum = row['vNum']
    mean_days_in_month_for_timeperiod = row['mean_days_in_month_for_timeperiod']
    if pd.isna(row['median']):
        median = row['median'] 
    if pd.isna(row['matPot_start_date']):
        matPot_start_date = row['matPot_start_date']
    if pd.isna(row['matPot_end_date']):
        matPot_start_date = row['matPot_end_date']
        
    print(f'Processing: {start_date} - {end_date}')
    
    start_date = pd.to_datetime(start_date, format='%m/%d/%Y')
    start_date = start_date.strftime('%Y-%m-%d')
    
    end_date = pd.to_datetime(end_date, format='%m/%d/%Y')
    end_date = end_date.strftime('%Y-%m-%d')
        
    base_dir_output = os.path.join(base_dir, "output", f"gv_mz_{year}_golfcourse_{output_extension}")
    os.makedirs(base_dir_output, exist_ok = True)
    
    # The following rasters were generated in ArcPro using the Raster Solar Radiation (Spatial Analyst) toolbox. 
    # Solar Radiation path (.tif)
    solarRadSS_path = os.path.join(base_dir_input, "lidar", f"blackfootportneuf_30m_solarradiationSummerSolstice_{year}.tif") # Summer solstice
    
    # Landsat8 path (.tif)
    # landsat8_path = os.path.join(base_dir_input, "landsat8", f"{year}", f"landsat8_{start_date}_{end_date}_{crop}_median_cloudless.tif") # 2-week timeslice
    landsat8_path = os.path.join(base_dir_input, "landsat8", f"{year}", f"landsat8_{crop}_{start_date}_{end_date}.tif") # monthly
    
    # ET raster path (.tif)
    et_raster_path = os.path.join(base_dir_input, "openET", f"{year}")
    
    # gridMET file path
    # gridmet_raster_path = os.path.join(base_dir_input, "gridmet", f"{year}", f"srad_gridMet_30m_{start_date}_{end_date}_{crop}.tif") # 2-week timeslice
    gridmet_raster_path = os.path.join(base_dir_input, "gridmet", f"{year}", f"gridmet_srad_{crop}_{start_date}_{end_date}.tif") # monthly
    
    crop_name = crop
    
    if not os.path.exists(field_boundary):
        print("Check field boundary file path.")
    if not os.path.exists(reference_raster):
        print("Check field boundary file path.")
    
    # Open reference raster
    ref_raster_live_i = rasterio.open(reference_raster)
    
    # Clip and Resample reference raster
    ref_raster_live_og = clip_resample_reproject_raster(reference_raster, input_shape=field_boundary, output_raster_dir=os.path.dirname(reference_raster), keyword=' ', 
                                                     raster_name=os.path.basename(reference_raster).split(".")[0] + "_reproj.tif", clip=False, resample=False, clip_and_resample=True, 
                                                     targetaligned=True, resample_algorithm='near', resolution=ref_raster_live_i.transform[0], crs='EPSG:4326', 
                                                     output_datatype=gdal.GDT_Float64, use_ref_width_height=False, ref_raster= reference_raster 
                                                     ) 
    
    # Open updated refernce raster
    ref_raster_live = rasterio.open(ref_raster_live_og)
    
    # temporary mask. Added to replace IrrMapper and CDL masking and maintain same functions. If certain areas need to be masked out,
    # the temporary mask can be replaced with some other mask. 
    temp_mask = np.ones_like(ref_raster_live.read(1), dtype=float)
    temp_mask_path = write_array_to_raster(temp_mask, 
                                           ref_raster_live, 
                                           ref_raster_live.transform, 
                                           os.path.join(base_dir_output, 'temp_mask.tif'),
                                           dtype=None,
                                           ref_file=ref_raster_live_og,
                                           nodata=nodatavalue)
       
    ''' Landsat 8 Processing '''
    final_ras_list = []   
    name_list = []   
    
    # Read in Landsat8 data
    l8_output_path = os.path.join(base_dir_output, 'landsat8')
    makedirs([l8_output_path])
    
    if os.path.exists(landsat8_path):
        # Reproject and clip Landsat8 data to study area
        ras_reproj=clip_resample_reproject_raster(landsat8_path, input_shape=field_boundary, output_raster_dir=l8_output_path, keyword=' ', 
                                                         raster_name=landsat8_path.split("\\")[-1].split(".")[0] + '_reproj.tif', clip=False, resample=False, clip_and_resample=True, 
                                                         targetaligned=True, resample_algorithm='near', resolution=ref_raster_live.transform[0], crs='EPSG:4326', 
                                                         output_datatype=gdal.GDT_Float64, use_ref_width_height=False, ref_raster= reference_raster 
                                                         ) 
        # Open updated Landsat8 raster
        ras = rasterio.open(ras_reproj, 'r')
        
        # Read Landsat8 bands
        # Separate bands from multi-band raster
        band_names = ras.descriptions 
        if not band_names:
            # If no descriptions are provided, use default names
            band_names = [f"Band_{i+1}" for i in range(ras.count)]     
        
        # Create raster for each band. Loop through bands in raster.
        for band_idx in range(1, ras.count + 1): 
            # Read the band data
            band_data = ras.read(band_idx)
            # Get the band name (use default if not available)
            band_name = band_names[band_idx - 1]    
            # Copy metadata from the original raster
            band_meta = ras.meta.copy()  
            band_meta.update({
                    'count': 1,                 # Only one band
                    'dtype': band_data.dtype,   # Data type of the band
                    'driver': 'GTiff'           # Output format (GeoTIFF)
                })
            bands_out_path = os.path.join(base_dir_output, 'landsat8', 'bands')
            makedirs([bands_out_path])
            with rasterio.open(os.path.join(bands_out_path, f'L8_{year}_median_{band_name}.tif'), 'w', **band_meta) as dst:
                dst.write(band_data, 1)  # Write the data to the first band
            # print(f"\t\t{band_name}")
            
        # Setting-up output directory
        index_output_path = os.path.join(base_dir_output, "landsat8", "indices")
        os.makedirs(index_output_path, exist_ok = True)
        
        # Assigning bands to variables
        blue = ras.read(2) # blue                - 452 - 512 nm     (SR-B2)
        green = ras.read(3) # green              - 533 - 590 nm     (SR-B3)
        red = ras.read(4) # red                  - 636 - 673 nm     (SR-B4)
        nir = ras.read(5) # nir                  - 851 - 879 nm     (SR-B5)
        swir1 = ras.read(6) # shortwave ir 1     - 1566 - 1651 nm   (SR-B6)
        swir2 = ras.read(7) # shortwave ir 2     - 2107 - 2294 nm   (SR-B7)
        
        # index: NDVI = (NIR-Red)/(NIR + red)
        NDVI = (nir-red)/(nir+red)

        # index: soil adjusted vegetation index (SAVI); L = 0.5
        SAVI = ((nir - red)/(nir + red + 0.5)) * (1.5)
        
        # List of indices (as string)
        index_list = ['NDVI', 'SAVI']
        etmodel_list = ['ensemble']
        
        # Call reference raster for write_array_to_raster function. Do not use "ras" because +1 bands. 
        # Loop through list of indices and write to raster in output folder
        for i in index_list:
            # Write indices to rasters
            write_array_to_raster(raster_arr = eval(i), 
                                  raster_file = ref_raster_live, 
                                  transform = ref_raster_live.transform, 
                                  output_path = os.path.join(index_output_path, f'L8_{year}_median_{i}.tif'), 
                                  ref_file = ref_raster_live_og 
                                  )
        
        # Setting up output directory for merged indices        
        out_dir = os.path.join(base_dir_output, 'indices')
        makedirs([out_dir])
        
        in_index_path = os.path.join(base_dir_output, 'landsat8', 'indices')
        
        # Reprojected merged indices to make them all the same shape 
        for dirpath, dirnames, filenames in os.walk(in_index_path):
            in_path = glob.glob(os.path.join(dirpath, "*median*.tif"))
            for path in in_path:
                out_name = os.path.basename(path)
                # print(f"\t{out_name}")
                out_path = os.path.join(out_dir, out_name)
                file_reproj = clip_resample_reproject_raster(path, input_shape=field_boundary, output_raster_dir=out_dir, keyword=' ', 
                                                             raster_name=out_name, clip=False, resample=False, clip_and_resample=True, 
                                                             targetaligned=True, resample_algorithm='near', resolution=ref_raster_live.transform[0], crs='EPSG:4326', 
                                                             output_datatype=gdal.GDT_Float64, use_ref_width_height=False, ref_raster= reference_raster 
                                                             ) 
              
  
        for dirpath, dirnames, filenames in os.walk(os.path.join(base_dir_output, 'indices')):
            in_path = [p for p in glob.glob(os.path.join(dirpath, "L8_*_median_*.tif"))
                   if not p.endswith("_final.tif")]
            for path in in_path:
                out_name = os.path.basename(path).split(".")[0] + "_final.tif"
                # print(f"\t{out_name}")
                out_path = os.path.join(base_dir_output, 'indices', out_name)
                file_reproj_final_arr, file_reproj_final_path = mask_rescale_write2raster(input_file_path = path, 
                                                              filter_file_path = temp_mask_path, 
                                                              out_path = out_path, 
                                                              tolerance_value=1e-6, 
                                                              no_data_value=nodatavalue, 
                                                              rescale=False, 
                                                              filter_file_path_2 = None)
                name_list.append(out_name.split('.')[0].split('_')[-2])
                final_ras_list.append(file_reproj_final_path)
                print(f'\t{out_name}')
    else:
        print(f"Landsat8 data for {start_date} to {end_date} not found.")
        
    ''' OpenET Data Processing '''
    # resample and clip ET data to target resolution
    et_output_path = os.path.join(base_dir_output, 'openet')
    makedirs([et_output_path])
    
    et_name = f"openet_et_{crop}_{start_date}_{end_date}.tif" # monthly
    

    for et_file in glob.glob(os.path.join(et_raster_path, et_name)):
        et_file_name = et_file.split('\\')[-1].split("_")[2]
            
        et_ras_reproject = clip_resample_reproject_raster(
                et_file,
                input_shape = field_boundary,
                output_raster_dir = et_output_path, 
                keyword = ' ',
                raster_name=f'GV_openet_{et_file_name}_{year}_clip.tif', 
                clip=False, 
                resample=False, 
                clip_and_resample=True, 
                targetaligned=True, 
                resample_algorithm='near', 
                resolution=ref_raster_live.transform[0], 
                crs='EPSG:4326', 
                output_datatype=gdal.GDT_Float32, 
                use_ref_width_height=False, 
                ref_raster= reference_raster 
                 ) 
            
        out_path = os.path.join(et_output_path, "openet_ensemble_final.tif")
            
        file_reproj_final_arr, file_reproj_final_path = mask_rescale_write2raster(input_file_path = et_ras_reproject, 
                                                            filter_file_path = temp_mask_path, 
                                                            out_path = out_path, 
                                                            tolerance_value=1e-6, 
                                                            no_data_value=nodatavalue, 
                                                            rescale=False, 
                                                            filter_file_path_2 = None)
        
        name_list.append("openet")
        final_ras_list.append(file_reproj_final_path)
        print("\tET ensemble")
    
    ''' Topographic Data Processing ''' 
    # List of topographic rasters to process
    topo_raster_list=[solarRadSS_path, gridmet_raster_path]
    
    # Set-up output directory
    topo_output_path = os.path.join(base_dir_output, 'topo')
    makedirs([topo_output_path])
    
    # Reproject raster
    reproj_rast_list = []
    for ras in topo_raster_list:
        name = get_var_name(ras)[0].split('_')[0]
        
        # print(f"\t{name.split('_')[0].title()}")
        raster_output_name = f"{name}_topo.tif"
        raster_reproj = clip_resample_reproject_raster(ras, input_shape=field_boundary, output_raster_dir=topo_output_path, keyword=' ', 
                                                         raster_name=raster_output_name, clip=False, resample=False, clip_and_resample=True, 
                                                         targetaligned=True, resample_algorithm='near', resolution=ref_raster_live.transform[0], crs='EPSG:4326', 
                                                         output_datatype=gdal.GDT_Float32, use_ref_width_height=False, ref_raster= reference_raster 
                                                         ) 
        
    
        # mask raster with crop-irr mask
        raster_reproj_masked_name = f"{name}_topo_final.tif"
        raster_reproj_masked, raster_reproj_masked_path = mask_rescale_write2raster(input_file_path = raster_reproj, 
                                                         filter_file_path = temp_mask_path, 
                                                         out_path = os.path.join(topo_output_path, raster_reproj_masked_name), 
                                                         tolerance_value=1e-6, 
                                                         no_data_value=nodatavalue, 
                                                         rescale=False, 
                                                         filter_file_path_2 = None)
        reproj_rast_list.append(raster_reproj_masked)
        print(f"\t{name.split('_')[0].title()}")
     
print("End of script.")
