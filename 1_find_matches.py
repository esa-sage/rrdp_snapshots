import os, sys
import argparse
import numpy as np
from snapshot_functions import *


def read_icechart_to_25km(date, dataset, filenames, hemis):
    import geopandas as gpd
    import pandas as pd
    from netCDF4 import Dataset as NetCDFFile
    
    gdf_list = []
    # account for multiple files per day
    for filename in filenames:
        gdf = gpd.read_file(filename)
        # this reduces the resolution of the shapefiles to 5 km for faster plotting 
        if dataset == 'AARI' and int(date[:4]) < 2020:
            gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.05, preserve_topology=True)
        else:
            gdf["geometry"] = gdf["geometry"].simplify(tolerance=5000, preserve_topology=True)
        gdf_list.append(gdf)
    # put the different files of same days into one list
    gdf = pd.concat(gdf_list, ignore_index=True)
    # remove polygons which are completely invalid (important for CIS data to have nan on lands)
    gdf = gdf[~gdf['CT'].isna()]
    # convert egg codes to SoD codes
    gdf = convert_egg_to_SoD(gdf)
    # combine icetypes A, B and C into one prominent icetype column, use only if at least 90%
    gdf = add_prominent_icetype_to_gdf(gdf)
    # rasterize to 25 km EASE grid
    data_25km = rasterize_icecharts_to_EASE(gdf, hemis)
    if data_25km is None: # this happens e.g. if there is only unknown ice types given in the gdf
        return None

    return data_25km

def DMI_icechart_to_EASE(xc, yc, raster_maps,
        src_crs="EPSG:3413", tgt_crs="EPSG:6931",
        extent=3_000_000, res=25_000,
        pixel_size=1000.0,
        chunk_rows=256,
        min_fraction_valid=0.9):
    """
    Project all DMI ice chart fields (SOD_A/B/C, SIC_A/B/C, SIC_T) to EASE2 grid,
    then combine to get dominant ice type and concentrations.
    
    Concentrations are calculated RELATIVE TO GRID CELL AREA (not relative to total ice).
    
    Parameters
    ----------
    xc, yc : array
        Source grid coordinates (1D arrays)
    raster_maps : dict
        Dictionary with keys: 'SOD_A', 'SOD_B', 'SOD_C', 'SIC_A', 'SIC_B', 'SIC_C', 'SIC_T'
        All values are 2D arrays at 1km resolution
    src_crs, tgt_crs : str
        Source and target CRS
    extent : float
        Grid extent in meters
    res : float
        Target resolution in meters
    pixel_size : float
        Source pixel size in meters (1000m for DMI)
    chunk_rows : int
        Number of rows to process at once
    min_fraction_valid : float
        Minimum fraction of valid pixels required per cell
    
    Returns
    -------
    dict
        Dictionary with:
        - 'dom_icetype': dominant ice type (1-4 or 0 for water, -77 for no data)
        - 'dom_icetype_conc': concentration of dominant ice type relative to cell area (0-1)
        - 'myi_conc': MYI concentration relative to cell area (0-1)
        - 'total_iceconc': total ice concentration (0-1)
    """
    import numpy as np
    from pyproj import Transformer

    # Target grid setup
    x_centers = np.arange(-extent, extent, res) + res/2.0
    y_centers = np.arange(extent, -extent, -res) + (-res/2.0)
    x_min = -extent
    y_max = extent
    nx_tgt = x_centers.size
    ny_tgt = int((2*extent) // res)
    total_tgt_cells = nx_tgt * ny_tgt

    # Initialize accumulation arrays for each ice type (1=YI, 2=FYI, 3=SYI, 4=MYI)
    # We accumulate CONCENTRATION-WEIGHTED counts
    ice_type_weights = {
        1: np.zeros(total_tgt_cells, dtype=np.float64),  # YI
        2: np.zeros(total_tgt_cells, dtype=np.float64),  # FYI
        3: np.zeros(total_tgt_cells, dtype=np.float64),  # SYI
        4: np.zeros(total_tgt_cells, dtype=np.float64),  # MYI
    }
    
    # Track total concentration separately (unweighted average)
    sum_total_conc = np.zeros(total_tgt_cells, dtype=np.float64)
    count_total_conc = np.zeros(total_tgt_cells, dtype=np.int64)
    
    # Track coverage for validation
    pixel_counts = np.zeros(total_tgt_cells, dtype=np.int64)
    valid_pixel_counts = np.zeros(total_tgt_cells, dtype=np.int64) 
    
    transformer = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)

    nx_src = xc.size
    ny_src = yc.size

    # Process in chunks
    for r0 in range(0, ny_src, chunk_rows):
        r1 = min(ny_src, r0 + chunk_rows)

        yc_chunk = yc[r0:r1]
        Xc_chunk = np.broadcast_to(xc, (yc_chunk.size, xc.size))
        Yc_chunk = np.broadcast_to(yc_chunk[:, None], (yc_chunk.size, xc.size))

        bx = Xc_chunk.ravel()
        by = Yc_chunk.ravel()
        
        # Extract all ice type and concentration fields
        sod_a = np.ma.filled(raster_maps['SOD_A'][r0:r1, :], -77).ravel().astype(np.int16)
        sod_b = np.ma.filled(raster_maps['SOD_B'][r0:r1, :], -77).ravel().astype(np.int16)
        sod_c = np.ma.filled(raster_maps['SOD_C'][r0:r1, :], -77).ravel().astype(np.int16)
        
        sic_a = np.ma.filled(raster_maps['SIC_A'][r0:r1, :], 0).ravel().astype(np.float32)
        sic_b = np.ma.filled(raster_maps['SIC_B'][r0:r1, :], 0).ravel().astype(np.float32)
        sic_c = np.ma.filled(raster_maps['SIC_C'][r0:r1, :], 0).ravel().astype(np.float32)
        
        sic_t = np.ma.filled(raster_maps['SIC_T'][r0:r1, :], -77).ravel().astype(np.float32)
        
        # Clip concentrations to valid range
        sic_a = np.clip(sic_a, 0, 100)
        sic_b = np.clip(sic_b, 0, 100)
        sic_c = np.clip(sic_c, 0, 100)
        
        # Valid mask: at least one valid ice type
        valid_mask = (sic_t >= 0) & (sic_t <= 100)
        
        if not np.any(valid_mask):
            continue

        # Transform coordinates (do once for all data)
        bx_tgt, by_tgt = transformer.transform(bx, by)
        
        # Calculate target grid indices
        col_f = (bx_tgt - x_min) / res
        row_f = (y_max - by_tgt) / res
        col_idx = np.floor(col_f).astype(np.int64)
        row_idx = np.floor(row_f).astype(np.int64)
        
        # Mask for pixels within target grid bounds
        mask_in = (
            (col_idx >= 0) & (col_idx < nx_tgt) &
            (row_idx >= 0) & (row_idx < ny_tgt)
        )

        # Combine masks
        mask_combined = valid_mask & mask_in
        
        if not np.any(mask_combined):
            continue

        # Filter all arrays to valid pixels
        col_idx = col_idx[mask_combined]
        row_idx = row_idx[mask_combined]
        sod_a = sod_a[mask_combined]
        sod_b = sod_b[mask_combined]
        sod_c = sod_c[mask_combined]
        sic_a = sic_a[mask_combined]
        sic_b = sic_b[mask_combined]
        sic_c = sic_c[mask_combined]
        sic_t = sic_t[mask_combined]
        
        flat_idx = row_idx * nx_tgt + col_idx
        
        # Track VALID ocean pixels (those that passed valid_mask)
        # valid_mask is still active in mask_combined
        unique_cells, counts = np.unique(flat_idx, return_counts=True)
        valid_pixel_counts[unique_cells] += counts  # These are VALID ocean pixels

        # Accumulate concentration-weighted ice types
        # Process A, B, C fields
        for sod, sic in [(sod_a, sic_a), (sod_b, sic_b), (sod_c, sic_c)]:
            for ice_type in [1, 2, 3, 4]:
                mask_type = (sod == ice_type)
                if np.any(mask_type):
                    # Add concentration weight to this ice type
                    for i in np.where(mask_type)[0]:
                        ice_type_weights[ice_type][flat_idx[i]] += sic[i]
        
        # Accumulate total concentration (unweighted average)
        valid_total = (sic_t >= 0) & (sic_t <= 100)
        if np.any(valid_total):
            for i in np.where(valid_total)[0]:
                sum_total_conc[flat_idx[i]] += sic_t[i]
                count_total_conc[flat_idx[i]] += 1
        
        # Track pixel coverage
        unique_cells, counts = np.unique(flat_idx, return_counts=True)
        pixel_counts[unique_cells] += counts

    # Reshape accumulation arrays to 2D
    ice_type_weights_2d = {
        ice_type: weights.reshape(ny_tgt, nx_tgt)
        for ice_type, weights in ice_type_weights.items()
    }
    
    # Stack concentrations for analysis
    conc_stack = np.stack([
        ice_type_weights_2d[1],  # YI
        ice_type_weights_2d[2],  # FYI
        ice_type_weights_2d[3],  # SYI
        ice_type_weights_2d[4],  # MYI
    ], axis=2)
    
    # Calculate expected pixels and minimum valid threshold
    expected_pixels = int((res / pixel_size) ** 2)
    min_valid_pixels = int(expected_pixels * min_fraction_valid)
    
    # Create validity mask based on VALID OCEAN pixel coverage
    valid_pixel_counts_2d = valid_pixel_counts.reshape(ny_tgt, nx_tgt)  # ✅ Use valid_pixel_counts
    valid_mask = valid_pixel_counts_2d >= min_valid_pixels 
    
    # Initialize with -77 (no data)
    dominant_icetype = np.full((ny_tgt, nx_tgt), -77, dtype=np.int16)
    
    # Only process cells with valid coverage AND some ice
    has_ice = (conc_stack.max(axis=2) > 0) & valid_mask  # ✅ Valid data AND ice present
    open_water = (conc_stack.max(axis=2) == 0) & valid_mask
    
    # Initialize dominant_idx for safe indexing
    dominant_idx = np.zeros((ny_tgt, nx_tgt), dtype=np.int32)
    
    # Only calculate dominant ice type where there's ice
    if np.any(has_ice):
        dominant_idx[has_ice] = np.argmax(conc_stack[has_ice], axis=1)
        ice_type_values = np.array([1, 2, 3, 4], dtype=np.int8)
        dominant_icetype[has_ice] = ice_type_values[dominant_idx[has_ice]]
    
    # Handle cells with valid coverage but NO ice (open water)
    dominant_icetype[open_water] = 0
    
    # Get concentration of dominant type
    rows = np.arange(ny_tgt)[:, None]
    cols = np.arange(nx_tgt)[None, :]
    dom_conc_raw = conc_stack[rows, cols, dominant_idx]
    
    # Normalize by ACTUAL pixel count * 100 (relative to cell area)
    # Each pixel that contributed data can have 0-100% concentration
    # So total possible is pixel_counts_2d * 100
    expected_total_weight = valid_pixel_counts_2d * 100.0
    
    # Calculate concentration as fraction of cell area
    dom_icetype_conc = np.full((ny_tgt, nx_tgt), np.nan, dtype=np.float32)
    mask_has_pixels = valid_pixel_counts_2d > 0
    dom_icetype_conc[mask_has_pixels] = (
        dom_conc_raw[mask_has_pixels] / expected_total_weight[mask_has_pixels]
    ).astype(np.float32)
    dom_icetype_conc = np.clip(dom_icetype_conc, 0.0, 1.0)
    #dom_icetype_conc[open_water] = 0.0
    
    # Calculate MYI concentration (relative to cell area)
    myi_conc = np.full((ny_tgt, nx_tgt), np.nan, dtype=np.float32)
    myi_conc[mask_has_pixels] = (
        ice_type_weights_2d[4][mask_has_pixels] / expected_total_weight[mask_has_pixels]
    ).astype(np.float32)
    myi_conc = np.clip(myi_conc, 0.0, 1.0)
    #myi_conc[open_water] = 0.0
    
    # Calculate total concentration from sum of all ice type concentrations
    total_ice_weight = (
        ice_type_weights_2d[1] +  # YI
        ice_type_weights_2d[2] +  # FYI
        ice_type_weights_2d[3] +  # SYI
        ice_type_weights_2d[4]    # MYI
    )

    total_conc = np.full((ny_tgt, nx_tgt), np.nan, dtype=np.float32)
    total_conc[mask_has_pixels] = (
        total_ice_weight[mask_has_pixels] / expected_total_weight[mask_has_pixels]
    ).astype(np.float32)
    total_conc = np.clip(total_conc, 0.0, 1.0)
    #total_conc[open_water] = 0.0

    # Apply validity mask
    dominant_icetype[~valid_mask] = -77
    dom_icetype_conc[~valid_mask] = np.nan
    myi_conc[~valid_mask] = np.nan
    total_conc[~valid_mask] = np.nan
    
    # Build output dictionary
    result = {
        'dom_icetype': dominant_icetype,
        'dom_icetype_conc': dom_icetype_conc,  
        'myi_conc': myi_conc,                  
        'total_ice_conc': total_conc        
    }

    return result

def read_DMI_icechart_to_25km(filenames, hemis):
    import numpy as np
    from netCDF4 import Dataset as NetCDFFile

    # get lookup tables
    SOD_lookup_numeric = ice_type_lookup()
    SIC_lookup_numeric = ice_concentration_lookup()

    for filename in filenames:
        with NetCDFFile(f'{filename}') as data:
            xc = data['xc'][:]
            yc = data['yc'][:]
            ice_poly_id_grid = data['ice_poly_id_grid'][0,:,:]
            polygon_id = data['polygon_id'][0,:]
            polygon_reference = data['polygon_reference'][:]
            SA = DMI_map_lookup_to_grid(ice_poly_id_grid,polygon_id,data['SA'][0,:])
            SB = DMI_map_lookup_to_grid(ice_poly_id_grid,polygon_id,data['SB'][0,:])
            SC = DMI_map_lookup_to_grid(ice_poly_id_grid,polygon_id,data['SC'][0,:])
            CA = DMI_map_lookup_to_grid(ice_poly_id_grid,polygon_id,data['CA'][0,:])
            CB = DMI_map_lookup_to_grid(ice_poly_id_grid,polygon_id,data['CB'][0,:])
            CC = DMI_map_lookup_to_grid(ice_poly_id_grid,polygon_id,data['CC'][0,:]) 
            CT = DMI_map_lookup_to_grid(ice_poly_id_grid,polygon_id,data['CT'][0,:]) 
            
        # map egg codes to SOD/icetype classes
        # Create raster maps dictionary
        raster_maps = {}
        # Map SOD (stage of development) variables
        raster_maps['SOD_A'] = DMI_map_lookup_to_grid(SA, SOD_lookup_numeric.keys(), SOD_lookup_numeric.values())
        raster_maps['SOD_B'] = DMI_map_lookup_to_grid(SB, SOD_lookup_numeric.keys(), SOD_lookup_numeric.values())
        raster_maps['SOD_C'] = DMI_map_lookup_to_grid(SC, SOD_lookup_numeric.keys(), SOD_lookup_numeric.values())
        # Map SIC (sea ice concentration) variables
        raster_maps['SIC_A'] = DMI_map_lookup_to_grid(CA, SIC_lookup_numeric.keys(), SIC_lookup_numeric.values())
        raster_maps['SIC_B'] = DMI_map_lookup_to_grid(CB, SIC_lookup_numeric.keys(), SIC_lookup_numeric.values())
        raster_maps['SIC_C'] = DMI_map_lookup_to_grid(CC, SIC_lookup_numeric.keys(), SIC_lookup_numeric.values())
        raster_maps['SIC_T'] = DMI_map_lookup_to_grid(CT, SIC_lookup_numeric.keys(), SIC_lookup_numeric.values())
        # combine A, B, C into one prominent icetype - use only if at least 90%
        #SOD_out, SIC_out = combined_icetype_maps_from_icecharts(raster_maps, pure = True)

    return DMI_icechart_to_EASE(xc, yc, raster_maps, pixel_size=1000.0)
    
def get_data_dict(date, datasets, hemis):
    
    data_dict_25km = {}
    
    for dataset, filenames in datasets.items():
        print(f"Reading {dataset} data from files: {filenames}")
        if dataset in ['AARI','CIS','NIC']:
            data_25km = read_icechart_to_25km(date, dataset, filenames, hemis)
        elif dataset == 'DMI':
            data_25km = read_DMI_icechart_to_25km(filenames, hemis)
        elif dataset == 'autoDMI':
            data_25km = read_autoDMI_to_25km(filenames, hemis)    
        elif dataset == 'S1':
            data_25km = read_S1_to_25km(filenames, hemis)
        else:
            print(f"Dataset {dataset} not recognized, skipping...")
        if data_25km is not None:
            data_dict_25km[dataset] = data_25km
    
    return data_dict_25km

def get_match_array(data_dict_25km, threshold=1e-5):
    """
    Computes a consensus ice type array from multiple datasets by identifying pixels where the dominant ice type agrees across datasets.
    For each pixel, the function:
    - Filters out low-confidence or missing data (dominant concentration < 0.8 or value == -77).
    - Stacks the dominant ice type arrays from all datasets.
    - Calculates the standard deviation and mean of the dominant ice type at each pixel, ignoring NaNs.
    - Identifies pixels where at least two datasets agree (standard deviation <= threshold and at least 2 valid values).
    - Assigns the rounded mean ice type value to those pixels; otherwise, assigns -77.
    Parameters
    ----------
    data_dict_25km : dict
        Dictionary where each value is a dict containing:
            - 'dom_icetype': 2D numpy array of dominant ice type codes.
            - 'dom_icetype_conc': 2D numpy array of dominant ice type concentrations.
    threshold : float, optional
        Maximum allowed standard deviation for agreement (default is 1e-5).
    Returns
    -------
    match_value : numpy.ndarray
        2D array of consensus ice type values (int16), with -77 indicating no agreement or insufficient data.
    """
    import numpy as np
    
    # For matching, use concentrations of dominant icetypes to filter for dominant ice types with very high (>=90%) concentrations
    icetype_arrays = []
    for value in data_dict_25km.values():
        dominant_concentration = value['dom_icetype_conc']
        dominant_icetype = value['dom_icetype'].copy().astype(np.float32)
        dominant_icetype[dominant_icetype == -77] = np.nan
        # Use NaN instead of -77 for missing/low confidence data
        dominant_icetype[dominant_concentration < 0.8] = -77
        icetype_arrays.append(dominant_icetype)
    
    # Stack filtered arrays
    icetype_arr = np.stack(icetype_arrays, axis=0)  # Shape: (n_datasets, height, width)
    
    # Calculate std across datasets (ignoring NaN)
    # Where std is ~0, all valid values agree
    with np.errstate(invalid='ignore'):  # Suppress NaN warnings
        std_vals = np.nanstd(icetype_arr, axis=0)
        mean_vals = np.nanmean(icetype_arr, axis=0)
    
    # Count valid (non-NaN) values per pixel
    n_valid = (~np.isnan(icetype_arr)).sum(axis=0)
    
    # Agreement where: std is ~0 and at least 2 valid values
    match_yes = (std_vals <= threshold) & (n_valid >= 2)
    
    # Initialize output array
    match_value = np.full(icetype_arr.shape[1:], -77, dtype=np.int16)
    
    # Store mean value where there's agreement (round to nearest int for ice types)
    match_value[match_yes] = np.round(mean_vals[match_yes]).astype(np.int16)
    
    return match_value

def get_match_array_myi_concentration(data_dict_25km, threshold=0.2):
    """
    Find grid cells where MYI concentration values agree across multiple datasets.
    Vectorized for better performance.
    
    Parameters
    ----------
    data_dict_25km : dict
        Dictionary where keys are dataset names and values are dicts containing
        'myi_conc' arrays with MYI concentration values (0.0 to 1.0 or NaN)
    threshold : float
        Maximum allowed difference between MYI concentrations for agreement (default: 0.1)
        
    Returns
    -------
    match_array : 2D np.ndarray
        Array with mean MYI concentration values where datasets agree, NaN elsewhere
    """
    import numpy as np
    
    # Get all myi_conc arrays
    myi_arrays = []
    
    for value in data_dict_25km.values():
        myi_concentration = value['myi_conc'].copy()
        #myi_concentration[(myi_concentration >= 0.9) | (myi_concentration <= 0.1)] = np.nan
        myi_arrays.append(myi_concentration)
    
    if len(myi_arrays) < 2:
        print(f"Need at least 2 datasets with myi_conc, found {len(myi_arrays)}")
        return np.full_like(myi_arrays[0], np.nan) if len(myi_arrays) > 0 else None
    
    # Stack arrays along new axis
    stacked = np.stack(myi_arrays, axis=0)  # Shape: (n_datasets, height, width)
    
    # Calculate mean across datasets (ignoring NaN)
    mean_myi_conc = np.nanmean(stacked, axis=0)
    
    # For each pixel, check if all finite values are within ±threshold of the mean
    abs_diff = np.abs(stacked - mean_myi_conc)
    # Mask for finite values
    finite_mask = np.isfinite(stacked)
    # For each pixel, check if all finite values are within threshold
    all_within = np.all((abs_diff <= threshold) | ~finite_mask, axis=0)
    # Also require at least 2 finite values
    n_finite = np.sum(finite_mask, axis=0)
    final_mask = (all_within & (n_finite >= 2))
    
    # Create output array
    match_array = np.full(stacked.shape[1:], np.nan, dtype=np.float32)
    match_array[final_mask] = mean_myi_conc[final_mask]
    
    return np.round(match_array, 2)

def save_to_netcdf(data_dict, used_datasets, output_path, hemis='N', res=25000):
    """
    Save matched ice type data to NetCDF file with CF-compliant metadata.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary containing ice type arrays and metadata
    output_path : str or Path
        Path to output NetCDF file
    hemis : str
        Hemisphere ('N' or 'S')
    res : int
        Grid resolution in meters
    """
    import xarray as xr
    import numpy as np
    from datetime import datetime
    from pathlib import Path
    import pandas as pd
    
    # Extract date from data_dict or output_path
    date_str = data_dict.get('date', Path(output_path).stem.split('_')[0])
    
    # Create dimension coordinates
    if hemis == 'N':
        extent = 3000000
        projection = '+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    else:
        extent = 4000000
        projection = '+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    
    size = int(2 * extent / res)
    xc = np.linspace(-extent, extent, size)
    yc = np.linspace(extent, -extent, size)
    
    # Build data_vars dictionary
    data_vars = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            # Expand to (time, yc, xc)
            data_vars[key] = (['time', 'yc', 'xc'], value[np.newaxis, :, :])
    
    # Parse date string and set to 12:00 UTC
    date_dt = pd.to_datetime(date_str, format='%Y%m%d').replace(hour=12, minute=0, second=0)

    # Create Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': pd.DatetimeIndex([date_dt]),  # 
            'xc': (['xc'], xc),
            'yc': (['yc'], yc)
        }
    )

    # Add time encoding for CF compliance
    ds.time.encoding['units'] = 'days since 1970-01-01'
    ds.time.encoding['calendar'] = 'proleptic_gregorian'
    
    # ===== GLOBAL ATTRIBUTES =====
    ds.attrs['title'] = f'Round Robin Data Package for Sea Ice Types'
    ds.attrs['summary'] = (
        'Multi-source sea ice type classification product gridded on 25 km EASE2 grid.'
        'Contains dominant ice type and MYI concentration where datasets agree,'
        'plus dominant ice types, their concentration and the MYI concentration of the single datasets.'
        'AMSR brightness temperatures and ERA5 variables are added.'
    )
    #ds.attrs['source'] = 'Sentinel-1 SAR, NIC ice charts, CIS ice charts, DMI ice charts, AARI ice charts, AMSR2 brightness temperatures, ERA5 reanalysis'
    #ds.attrs['institution'] = 'Your Institution Name'
    #ds.attrs['creator_name'] = 'Your Name'
    #ds.attrs['creator_email'] = 'your.email@domain.com'
    ds.attrs['date_created'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
    ds.attrs['date'] = date_str
    ds.attrs['hemisphere'] = hemis
    ds.attrs['resolution_m'] = res
    ds.attrs['projection'] = projection
    ds.attrs['datasets_included'] = ', '.join(used_datasets)
    #ds.attrs['Conventions'] = 'CF-1.8'
    
    # ===== ICE TYPE CODE TABLE =====
    ds.attrs['ice_type_codes'] = '1=YI (Young Ice), 2=FYI (First-Year Ice), 3=SYI (Second-Year Ice), 4=MYI (Multi-Year Ice), -77=No Data'
    
    # ===== VARIABLE ATTRIBUTES =====
    
    # Coordinate attributes
    ds['time'].attrs = {
        'standard_name': 'time',
        'long_name': 'time',
        'axis': 'T'
    }
    ds['xc'].attrs = {
        'standard_name': 'projection_x_coordinate',
        'long_name': 'x coordinate of projection',
        'units': 'm',
        'axis': 'X'
    }
    ds['yc'].attrs = {
        'standard_name': 'projection_y_coordinate',
        'long_name': 'y coordinate of projection',
        'units': 'm',
        'axis': 'Y'
    }
    
    # Ice type match variables
    if 'icetype_matches' in ds:
        ds['icetype_matches'].attrs = {
            'long_name': 'Dominant ice type where all datasets agree on ice type and its concentration >=80%',
            'units': '1',
            'flag_values': np.array([1, 2, 3, 4], dtype=np.int16),
            'flag_meanings': 'young_ice first_year_ice second_year_ice multi_year_ice',
            'valid_range': np.array([1, 4], dtype=np.int16),
            #'_FillValue': np.int16(-77),
            'grid_mapping': 'crs'
        }
    
    if 'myi_concentration_matches' in ds:
        ds['myi_concentration_matches'].attrs = {
            'long_name': 'Mean multi-year ice concentration if all datasets agree within +-20%',
            'units': '1',
            'valid_min': np.float32(0.0),
            'valid_max': np.float32(1.0),
            #'_FillValue': np.float32(np.nan),
            'grid_mapping': 'crs'
        }
    
    # Dataset-specific variables (loop through known datasets)
    datasets = ['S1', 'NIC', 'CIS', 'DMI', 'AARI', 'autoDMI']
    for dataset in datasets:
        if f'{dataset}_dom_icetype' in ds:
            ds[f'{dataset}_dom_icetype'].attrs = {
                'long_name': f'{dataset} dominant ice type',
                'source': dataset,
                'units': '1',
                'flag_values': np.array([1, 2, 3, 4], dtype=np.int16),
                'flag_meanings': 'young_ice first_year_ice second_year_ice multi_year_ice',
                'valid_range': np.array([1, 4], dtype=np.int16),
                #'_FillValue': np.int16(-77),
                'grid_mapping': 'crs'
            }
        
        if f'{dataset}_dom_icetype_conc' in ds:
            ds[f'{dataset}_dom_icetype_conc'].attrs = {
                'long_name': f'{dataset} concentration of dominant ice type',
                'source': dataset,
                'units': '1',
                'valid_min': np.float32(0.0),
                'valid_max': np.float32(1.0),
                #'_FillValue': np.float32(np.nan),
                'grid_mapping': 'crs'
            }
        
        if f'{dataset}_myi_conc' in ds:
            ds[f'{dataset}_myi_conc'].attrs = {
                'long_name': f'{dataset} multi-year ice concentration',
                'source': dataset,
                'units': '1',
                'valid_min': np.float32(0.0),
                'valid_max': np.float32(1.0),
                #'_FillValue': np.float32(np.nan),
                'grid_mapping': 'crs'
            }
            
        if f'{dataset}_total_ice_conc' in ds:
            ds[f'{dataset}_total_ice_conc'].attrs = {
                'long_name': f'{dataset} total sea ice concentration',
                'source': dataset,
                'units': '1',
                'valid_min': np.float32(0.0),
                'valid_max': np.float32(1.0),
                #'_FillValue': np.float32(np.nan),
                'grid_mapping': 'crs'
            }
    
    # AMSR2 brightness temperatures
    frequencies = ['6.9H', '6.9V', '10.7H', '10.7V', '18.7H', '18.7V', '23.8V', '23.8H', '36.5V', '36.5H', '89V', '89H']
    for freq in frequencies:
        for orbit in ['M', 'E']:
            var_name = f'AMSR2_TB{freq}_{orbit}'
            if var_name in ds:
                orbit_desc = 'morning overflight' if orbit == 'M' else 'evening overflight'
                ds[var_name].attrs = {
                    'long_name': f'AMSR2 brightness temperature at {freq}, {orbit_desc}',
                    'source': 'AMSR2',
                    'units': 'K',
                    #'_FillValue': np.float32(np.nan),
                    'grid_mapping': 'crs'
                }
    
    # ERA5 variables
    era5_vars = ['t2m_min', 't2m_max', 't2m_mean', 
             'skt_min', 'skt_max', 'skt_mean',
             'tcwv_min', 'tcwv_max', 'tcwv_mean',
             'tclw_min', 'tclw_max', 'tclw_mean',
             'tciw_min', 'tciw_max', 'tciw_mean',
             'u10_min', 'u10_max', 'u10_mean',
             'v10_min', 'v10_max', 'v10_mean'
            ]
    era5_longnames = [
        'daily minimum of 2m air temperature', 'daily maximum of 2m air temperature', 'daily mean of 2m air temperature',
        'daily minimum of skin temperature', 'daily maximum of skin temperature', 'daily mean of skin temperature',
        'daily minimum of total column water vapor', 'daily maximum of total column water vapor', 'daily mean of total column water vapor',
        'daily minimum of total column liquid water', 'daily maximum of total column liquid water', 'daily mean of total column liquid water',
        'daily minimum of total column ice water', 'daily maximum of total column ice water', 'daily mean of total column ice water',
        'daily minimum of 10m u-component of wind', 'daily maximum of 10m u-component of wind', 'daily mean of 10m u-component of wind',
        'daily minimum of 10m v-component of wind', 'daily maximum of 10m v-component of wind', 'daily mean of 10m v-component of wind'
    ]
    for var, longname in zip(era5_vars, era5_longnames):
        if f'ERA5_{var}' in ds:
            ds[f'ERA5_{var}'].attrs = {
                'long_name': f'ERA5 {longname}',
                'source': 'ERA5',
                'units': (
                    'K' if ('t2m' in var or 'skt' in var)
                    else 'm/s' if '10' in var
                    else 'kg m-2'
                ),
                #'_FillValue': np.float32(np.nan),
                'grid_mapping': 'crs'
            }

    # ASCAT variables
    ascat_vars = ['ASCAT_S0']
    for var in ascat_vars:
        if var in ds:
            ds[var].attrs = {
                'long_name': 'ASCAT normalized radar cross section',
                'source': 'ASCAT',
                'units': 'dB',
                #'_FillValue': np.float32(np.nan),
                'grid_mapping': 'crs'
            }
    
    # Add CRS variable
    crs = xr.DataArray(
        np.int32(0),
        attrs={
            'grid_mapping_name': 'lambert_azimuthal_equal_area',
            'latitude_of_projection_origin': 90.0 if hemis == 'N' else -90.0,
            'longitude_of_projection_origin': 0.0,
            'false_easting': 0.0,
            'false_northing': 0.0,
            'spatial_ref': projection,
            'crs_wkt': projection
        }
    )
    ds['crs'] = crs
    
    # Save to NetCDF
    encoding = {}
    for var in ds.data_vars:
        var_encoding = {'zlib': True, 'complevel': 4}
        
        # Set _FillValue based on dtype
        if ds[var].dtype in [np.int8, np.int16, np.int32, np.int64]:
            var_encoding['dtype'] = ds[var].dtype
            var_encoding['_FillValue'] = np.int16(-77)  # For integer ice type variables
        else:
            # For float variables (concentrations, TB, ERA5)
            var_encoding['dtype'] = np.float32
            var_encoding['_FillValue'] = np.float32(np.nan)
        
        encoding[var] = var_encoding
    
    ds.to_netcdf(output_path, encoding=encoding, unlimited_dims=['time'])
    print(f"Saved to {output_path}")

def main(start_date, end_date, hemis, outpath):
    
    # Generate list of dates to process
    dates = get_date_range_in_str(start_date, end_date)
    

    for date in dates:

        day_filename = os.path.join(outpath, f"{date}_{hemis}.nc")
        # Check if date is already processed
        if os.path.exists(day_filename):
            print(f"### Date {date} already processed, skipping.")
            continue
        
        # check how many datasets are available for the day
        datasets = check_data_availability(date, hemis)
        if len(datasets) < 2:
            print(f"### Only {list(datasets.keys())} data available for {date}, skipping...")
            continue
        else:
            print(f"### Processing date {date} with datasets: {list(datasets.keys())}")
            
        # read datasets of the day directly into 25 km EASE grid
        data_dict_25km = get_data_dict(date, datasets, hemis)
        
        # create new array with matches of dominant icetypes
        match_array = get_match_array(data_dict_25km)
        # use myi concentration arrays to find matches within +- 10% myi concentration
        match_conc = get_match_array_myi_concentration(data_dict_25km, threshold=0.2)
        # set match_conc to nan if it is valid in the pure match_array
        #match_conc[match_array != -77] = np.nan
        
        # collect AMSR data for the day
        AMSR_dict = read_AMSR_day_to_EASE2(date, hemis)
        
        # collect ERA5 data for the day
        era5_dict = read_ERA5_day_to_EASE2(date, hemis)
        
        # collect ASCAT data for the day
        ASCAT_dict = read_ASCAT_day_to_EASE2(date, hemis)

        # save to netcdf
        #save_to_netcdf(date, hemis, data_dict_25km, match_array, match_conc, AMSR_dict, era5_dict, day_filename)
        # Combine all data into a single dictionary for saving
        combined_dict = {
            'date': date,  # Add date for metadata
            'icetype_matches': match_array,
            'myi_concentration_matches': match_conc
        }
        
        used_datasets = list(data_dict_25km.keys())
        # Add dataset-specific variables (flatten nested dict)
        for dataset_name, dataset_data in data_dict_25km.items():
            combined_dict[f'{dataset_name}_dom_icetype'] = dataset_data['dom_icetype']
            combined_dict[f'{dataset_name}_dom_icetype_conc'] = dataset_data['dom_icetype_conc']
            combined_dict[f'{dataset_name}_myi_conc'] = dataset_data['myi_conc']
            # Only add total_ice_conc if it exists (ice charts only)
            if 'total_ice_conc' in dataset_data:
                combined_dict[f'{dataset_name}_total_ice_conc'] = dataset_data['total_ice_conc']
        
        # Add AMSR2 data
        if AMSR_dict is not None:
            combined_dict.update(AMSR_dict)
        # Add ERA5 data
        if era5_dict is not None:
            combined_dict.update(era5_dict)
        # Add ASCAT data
        if ASCAT_dict is not None:
            combined_dict.update(ASCAT_dict)
        
        # Save to netcdf
        save_to_netcdf(combined_dict, used_datasets, day_filename, hemis)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare icetype datasets and find possible references.")
    parser.add_argument("start_date", help="start date of sarching period in YYYYMMDD format")
    parser.add_argument("end_date", help="end date of sarching period in YYYYMMDD format")
    parser.add_argument("hemis", help="hemisphere, either 'N' or 'S'")
    parser.add_argument("outpath", help="Path to the output netcdf files")
    args = parser.parse_args()
    
    main(args.start_date, args.end_date, args.hemis, args.outpath)