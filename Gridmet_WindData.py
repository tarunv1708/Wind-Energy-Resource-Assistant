import xarray as xr
import pandas as pd
import numpy as np


# try this code : conda install netcdf4 h5netcdf

def get_gridmet_data(lat: float, lon: float, start: str, end: str, vars=['ws']):
    """
    Fetch GRIDMET data for specified variables and location within a time range.

    Args:
    vars (list): List of variable abbreviations to fetch (default is ['ws'] for wind speed).
    lat (float): Latitude of the location.
    lon (float): Longitude of the location.
    start (str): Start date in YYYY-MM-DD format.
    end (str): End date in YYYY-MM-DD format.

    Returns:
    pandas.DataFrame: DataFrame containing the dates and requested variables.
    """

    # Mapping from short variable names to GRIDMET dataset names
    variables_map = {
        'ws': 'vs',  # Wind speed
        'wd': 'th',  # Wind direction
        'tmax': 'tmmx',  # Maximum temperature
        'tmin': 'tmmn',  # Minimum temperature
        'srad': 'srad',  # Solar radiation
        'prcp': 'pr',  # Precipitation
        'vpd': 'vpd',  # Mean Vapor Pressure Deficit
        'pdsi': 'pdsi'  # Palmer Drought Severity Index
    }

    # Initialize DataFrame
    df = None

    # Process each variable requested
    for variable in vars:
        # Check if the variable is in the map
        if variable not in variables_map:
            raise ValueError(f"Variable {variable} not recognized.")

        # Construct the URL for the dataset
        dataset_url = f"http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_{variables_map[variable]}_1979_CurrentYear_CONUS.nc"
        data = xr.open_dataset(dataset_url)

        # Select the nearest location data and time slice
        data = data.sel(lon=lon, lat=lat, method='nearest')
        data = data.sel(day=slice(start, end))

        # Extract the variable as a DataFrame and round the values
        var_df = data[list(data.data_vars)[0]].to_dataframe().drop(['lat', 'lon'], axis=1)
        var_df.rename(columns={list(data.data_vars)[0]: variable}, inplace=True)

        # Combine the data into one DataFrame
        df = var_df if df is None else df.join(var_df)
    return df.round(2)


if __name__ == '__main__':
    # Example of usage:
    df = get_gridmet_data(vars=['ws', 'wd'], lat=29.594884, lon=-90.728542, start='2000-01-01', end='2023-12-31')
    print(df)

    # Save to CSV
    df.to_csv("gridmet3_output.csv", index=True)
    print("Data saved to gridmet312-31-2023_output.csv")