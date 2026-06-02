"""
Author: rpommer
Date: April 2026
Description:
    Compile timeslice reults from each monthly timeslice into one excel sheet.
    Every tab represents one growing season of data.
    
"""

import os
import re
from datetime import datetime
import pandas as pd

''' Setting up directories '''

if "__file__" in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
else:
    script_dir = os.getcwd()
    
base_dir = os.path.abspath(os.path.join(script_dir, ".."))

''' User input ''' 

DATE_CSV = os.path.join(base_dir, "input", "input_info_months.csv") # trimmed to only include time slices (not optimized windows)

DATA_FOLDER = os.path.join(base_dir, "output", "wsi_timeslices") # output folder

OUTPUT_EXCEL = os.path.join(base_dir, "output", "results_full_timeseries_LR_optimalwindows.xlsx") # Name of output excel file

EXTENSION = "_lr_optimalwindows.csv" # extension must be typed as all lowercase

# Regex to capture the two dates from column headers
DATE_PATTERN = re.compile(r'_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})')

''' Functions ''' 

def extract_dates_from_column(col_name):
    match = DATE_PATTERN.search(col_name)
    if match:
        start = datetime.strptime(match.group(1), "%Y-%m-%d")
        end = datetime.strptime(match.group(2), "%Y-%m-%d")
        return start, end
    return None, None

def get_matching_columns(df, start_date, end_date):
    start_date = start_date.date()
    end_date = end_date.date()

    matches = []
    for col in df.columns:
        col_start, col_end = extract_dates_from_column(col)
        if col_start and col_end:
            if col_start.date() == start_date and col_end.date() == end_date:
                matches.append(col)
    return matches

''' Running script ''' 

print("Loading date CSV...")
date_df = pd.read_csv(DATE_CSV, parse_dates=["start_date", "end_date"])
date_df["year"] = date_df["start_date"].dt.year

# Keep DATE_CSV order exactly
date_df = date_df.sort_index()

# Pre-read all csv files once (faster)
csv_data = {}
for file in os.listdir(DATA_FOLDER):
    if file.lower().endswith(EXTENSION):
        path = os.path.join(DATA_FOLDER, file)
        csv_data[file] = pd.read_csv(path)

# Dictionary to accumulate data per year
year_data = {}

for _, row in date_df.iterrows():
    year = int(row["year"])
    start_date = row["start_date"]
    end_date = row["end_date"]

    print(f"Processing {start_date.date()} → {end_date.date()}")

    if year not in year_data:
        year_data[year] = pd.DataFrame()

    for fname, df in csv_data.items():
        matching_cols = get_matching_columns(df, start_date, end_date)

        if matching_cols:
            year_data[year] = pd.concat([year_data[year], df[matching_cols]], axis=1)

with pd.ExcelWriter(OUTPUT_EXCEL, engine="xlsxwriter") as writer:
    for year in sorted(year_data.keys()):
        if not year_data[year].empty:
            year_data[year].to_excel(writer, sheet_name=str(year), index=False)
            print(f"Sheet written for {year}")

print("End of script.")