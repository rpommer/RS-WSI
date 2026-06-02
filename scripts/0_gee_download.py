# import libraries
import ee
import os
import geemap
import geopandas as gpd
import pandas as pd

# Set up working directory
if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()

base_dir = os.path.abspath(os.path.join(script_dir, ".."))

# Connect to Google Earth Engine
ee.Authenticate()
ee.Initialize()

''' User Input Start '''
# CSV file with startDate, endDate, crop, year,	days_before_harvest,	 days_after_planting, harvestDate, plantingDate
CSV_FILE_PATH = os.path.join(base_dir, 'input', 'input_info_months_optimalwindows.csv')        

# Boundary shapefile EPSG: 4326         
SHAPEFILE_PATH = os.path.join(base_dir, 'input', 'studyarea', 'gem_valley_boundary.shp')

# Outut Google Drive Folder
DRIVE_FOLDER = 'gem_valley_month_data'

''' User Input End '''

# load shapefile
gdf = gpd.read_file(SHAPEFILE_PATH)
studyArea = geemap.geopandas_to_ee(gdf).geometry()

# load csv file
df = pd.read_csv(CSV_FILE_PATH)

''' Functions Start '''
def maskL8sr(image):
    cloudShadowBitMask = 1 << 3
    cloudsBitMask = 1 << 5
    qa = image.select('QA_PIXEL')
    mask = (qa.bitwiseAnd(cloudShadowBitMask).eq(0)
            .And(qa.bitwiseAnd(cloudsBitMask).eq(0)))
    return (image.updateMask(mask)
            .divide(10000)
            .select("SR_B[0-9]*")
            .copyProperties(image, ["system:time_start"]))

def export_to_drive(image, description):
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=DRIVE_FOLDER,
        region=studyArea,
        scale=30,
        maxPixels=1e13
    )
    task.start()
    print(f"Started export: {description}")

def clean(text):
    return (
        str(text)
        .replace('/', '-')
        .replace(' ', '')
        .replace(':', '')
    )

''' Functions End '''

for idx, row in df.iterrows():
    start_date = row['startDate']
    end_date = row['endDate']
    crop = row['crop']
    year = start_date[-4:]
    
    start_date = pd.to_datetime(start_date, format='%m/%d/%Y')
    start_date = start_date.strftime('%Y-%m-%d')
    
    end_date = pd.to_datetime(end_date, format='%m/%d/%Y')
    end_date = end_date.strftime('%Y-%m-%d')

    print(f'\nProcessing {crop} {start_date} - {end_date}')

    collection = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                  .filterDate(start_date, end_date)
                  .filterBounds(studyArea)
                  .map(maskL8sr))
    
    count = collection.size().getInfo()
    if count == 0:
        print(f"No Landsat images found for {crop} {start_date} - {end_date}, skipping...")
    else:
        landsat = collection.mean() # mean
        export_to_drive(
            landsat,
            f'landsat8_{crop}_{start_date}_{end_date}'
        )

    gridmet = (ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
               .select('srad')
               .filterDate(start_date, end_date)
               .filterBounds(studyArea)
               .mean())  # mean

    export_to_drive(
        gridmet,
        f'gridmet_srad_{crop}_{start_date}_{end_date}'
    )

    openet = (ee.ImageCollection('OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0')
              .select('et_ensemble_mad')
              .filterDate(start_date, end_date)
              .filterBounds(studyArea)
              .mean()  # mean
              .toFloat())

    export_to_drive(
        openet,
        f'openet_et_{crop}_{start_date}_{end_date}'
    )
    
print("\n All jobs submitted to Google Earth Engine.")


# # Cancel all tasks    
# tasks = ee.data.listOperations()

# for task in tasks:
#     # Check if the task is running or pending
#     if task['metadata']['state'] in ['RUNNING', 'PENDING']:
#         task_id = task['name'].split('/')[-1]
#         ee.data.cancelOperation(task['name'])
#         print(f"Cancelled task: {task_id}")   