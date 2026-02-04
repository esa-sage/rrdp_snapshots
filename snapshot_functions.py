def get_date_range_in_str(start_str, end_str):

    """
    takes two strings of dates in the format "yyyymmdd" and returns a list of all dates in between in the same format
    """

    from datetime import datetime, timedelta
    
    # Parse input strings into datetime objects
    start_date = datetime.strptime(start_str, "%Y%m%d")
    end_date = datetime.strptime(end_str, "%Y%m%d")
    
    # Compute number of days between dates
    num_days = (end_date - start_date).days + 1  # +1 to include end date
    
    # Generate list of date strings
    date_list = [
        (start_date + timedelta(days=i)).strftime("%Y%m%d")
        for i in range(num_days)
    ]
    return date_list

def check_data_availability(date, hemis):
    import os, sys
    import glob
    if hemis == 'N':
        use_datasets=['NIC','CIS','AARI','DMI','autoDMI','S1']
    else:
        use_datasets=['NIC','AARI']
    DATASETS = {}
    if 'NIC' in use_datasets:
        AA = {"N": "ARCTIC", "S": "ANTARC"}.get(hemis)
        FILE = f"/mnt/spaces/Users/hniehaus/data/SAGE_RRDP/NIC_icecharts/{AA}{date}.shp"
        if os.path.exists(FILE):
            DATASETS['NIC'] = [FILE]
    if 'CIS' in use_datasets and hemis == 'N':
        FILES = glob.glob(f"/mnt/spaces/Users/hniehaus/data/SAGE_RRDP/CIS_icecharts/cis_SGRD*_{date}*pl_a.shp")
        if FILES:
            DATASETS['CIS'] = FILES
    if 'AARI' in use_datasets:
        AA = {"N": "arc", "S": "antice"}.get(hemis)
        FILES = glob.glob(f"/mnt/spaces/Users/hniehaus/data/SAGE_RRDP/AARI_icecharts/aari_{AA}_{date}_pl*.shp")
        if FILES:
            DATASETS['AARI'] = FILES
    if 'DMI' in use_datasets and hemis == 'N':
        FILES = glob.glob(f'/mnt/spaces/Users/hniehaus/data/SAGE_RRDP/DMI_icecharts/ice_conc_overview_greenland_{date}*.nc')
        if FILES:
            DATASETS['DMI'] = FILES
    if 'autoDMI' in use_datasets:
        FILES = glob.glob(f"/mnt/spaces/Users/hniehaus/data/SAGE_RRDP/DMI_auto_icecharts/dmi_asip_seaice_mosaic_arc_l3*{date}.nc") 
        if FILES:
            DATASETS['autoDMI'] = FILES
    if 'S1' in use_datasets and hemis == 'N':
        FILE = f'/mnt/spaces/SAR/Sentinel-1 ice types/s1_icetype_mosaic_{date}0600.nc'
        if os.path.exists(FILE):
            DATASETS['S1'] = [FILE]
    if 'TSX' in use_datasets:
        FILENAMES = glob.glob(f"/mnt/spaces/Users/hniehaus/data/SAGE_RRDP/TSX_Karl/Surface_Type_Classification_{date}*.nc")
        if FILENAMES:
            DATASETS['TSX'] = FILENAMES
    return DATASETS

def rasterize_icecharts_to_EASE(
        gdf,
        hemis,
        res=25000,
        extent_window=None,
        oversample=25,
        min_valid_fraction=0.95):
    """
    Rasterize ice chart polygons to EASE2 and return:
      - dominant_icetype: class with highest concentration
      - dom_icetype_conc: concentration of dominant ice type
      - myi_conc, syi_conc, fyi_conc, yi_conc: concentrations for each ice type

    All outputs share the same validity mask (geometric coverage ≥ min_valid_fraction).
    """
    import numpy as np
    import math
    import pyproj
    from rasterio.features import rasterize
    from affine import Affine

    assert hemis in ["N", "S"]
    if extent_window is None:
        extent_window = 3_000_000 if hemis == "N" else 3_000_000

    # Grid definition (EASE2 Polar)
    EXTENT = 9_000_000
    size_lookup = {3125: 5760, 6250: 2880, 12500: 1440, 25000: 720, 36000: 500}
    if res not in size_lookup:
        raise ValueError("Unsupported resolution")
    full_size = size_lookup[res]
    ease_crs = "EPSG:6931" if hemis == "N" else "EPSG:6932"
    half = res / 2
    origin_x = -EXTENT + half
    origin_y = (+EXTENT - half) if hemis == "N" else (-EXTENT + half)

    xmin, xmax = -extent_window, extent_window
    ymin, ymax = -extent_window, extent_window

    col_start = max(0, math.ceil((xmin - origin_x) / res))
    col_end   = min(full_size - 1, math.floor((xmax - origin_x) / res))
    if hemis == "N":
        row_start = max(0, math.ceil((origin_y - ymax) / res))
        row_end   = min(full_size - 1, math.floor((origin_y - ymin) / res))
    else:
        row_start = max(0, math.ceil((ymin - origin_y) / res))
        row_end   = min(full_size - 1, math.floor((ymax - origin_y) / res))

    width  = col_end - col_start + 1
    height = row_end - row_start + 1

    if hemis == "N":
        transform = Affine(res, 0.0, origin_x + col_start * res,
                           0.0, -res, origin_y - row_start * res)
    else:
        transform = Affine(res, 0.0, origin_x + col_start * res,
                           0.0, +res, origin_y + row_start * res)

    # Reproject input gdf to EASE2 if needed
    if pyproj.CRS(gdf.crs) != pyproj.CRS(ease_crs):
        gdf = gdf.to_crs(ease_crs)

    # Subpixel transform
    sub_h = height * oversample
    sub_w = width * oversample
    sub_res = res / oversample
    if hemis == "N":
        sub_transform = Affine(sub_res, 0, transform.c, 0, -sub_res, transform.f)
    else:
        sub_transform = Affine(sub_res, 0, transform.c, 0, +sub_res, transform.f)

    # ---------------------------------------------------------
    # 1) Calculate valid mask once (all polygons, geometric coverage)
    # ---------------------------------------------------------
    all_geoms = list(gdf.geometry)
    sub_area = rasterize(
        [(geom, 1) for geom in all_geoms],
        out_shape=(sub_h, sub_w),
        transform=sub_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True
    )
    sub_area = sub_area.reshape(height, oversample, width, oversample)
    total_area_fraction = sub_area.mean(axis=(1, 3)).astype(np.float32)
    # flip for South
    if hemis == "S":
        total_area_fraction = np.flipud(total_area_fraction)
    
    # Valid mask: at least min_valid_fraction of cell is covered
    valid = total_area_fraction >= min_valid_fraction

    # ---------------------------------------------------------
    # 2) Rasterize concentrations for each ice type (MYI, SYI, FYI, YI)
    # ---------------------------------------------------------
    ice_types = {
        'myi': 4,
        'syi': 3,
        'fyi': 2,
        'yi': 1
    }
    
    concentrations = {}
    
    for ice_name, ice_value in ice_types.items():
        # Check if concentration column exists
        conc_column = f'{ice_name}_conc'
        if conc_column not in gdf.columns:
            concentrations[ice_name] = np.zeros((height, width), dtype=np.float32)
            continue
        
        # Get geometries and concentration values for this ice type
        conc_vals = gdf[conc_column].to_numpy(dtype=np.float32)
        
        # Sanitize: valid concentrations are 0-1 (as concentrations)
        conc_vals = np.where((conc_vals >= 0) & (conc_vals <= 1), conc_vals, 0.0)
        
        # Rasterize with concentration values
        shapes = [(geom, val) for geom, val in zip(gdf.geometry, conc_vals) if val > 0]
        
        if len(shapes) == 0:
            concentrations[ice_name] = np.zeros((height, width), dtype=np.float32)
            continue
        
        sub_conc = rasterize(
            shapes,
            out_shape=(sub_h, sub_w),
            transform=sub_transform,
            fill=0.0,
            dtype=np.float32,
            all_touched=False
        )
        
        # Average over subpixels to get 25km concentration
        sub_conc = sub_conc.reshape(height, oversample, width, oversample)
        conc_25km = sub_conc.mean(axis=(1, 3)).astype(np.float32)
        # 🔧 Apply same orientation fix as total_area_fraction
        if hemis == "S":
            conc_25km = np.flipud(conc_25km)
        
        concentrations[ice_name] = conc_25km

    # ---------------------------------------------------------
    # 3) Determine dominant ice type and its concentration
    # ---------------------------------------------------------
    # Stack concentrations for comparison (only ice types, not water)
    conc_stack = np.stack([
        concentrations['yi'],   # index 0 -> ice type 1
        concentrations['fyi'],  # index 1 -> ice type 2
        concentrations['syi'],  # index 2 -> ice type 3
        concentrations['myi']   # index 3 -> ice type 4
    ], axis=2)
    
    # sum over ice type concentrations to get total sic
    total_sic = np.nansum(conc_stack, axis = 2)
    total_sic = np.where(total_sic > 1, 1, total_sic)
    
    # Find index of maximum concentration
    dom_idx = np.argmax(conc_stack, axis=2)
    
    # Map index to ice type value (1, 2, 3, 4)
    ice_type_values = np.array([1, 2, 3, 4], dtype=np.int8)
    dominant_icetype = ice_type_values[dom_idx]
    
    # Get concentration of dominant type
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    dom_icetype_conc = conc_stack[rows, cols, dom_idx].astype(np.float32)
    
    # Handle cells with no ice (all concentrations are 0)
    no_ice = conc_stack.max(axis=2) == 0
    dominant_icetype[no_ice] = 0  # Set to water/open ocean
    dom_icetype_conc[no_ice] = 0.0
    
    # Apply validity mask
    dominant_icetype[~valid] = -77
    dom_icetype_conc[~valid] = np.nan
    for ice_name in ice_types.keys():
        concentrations[ice_name][~valid] = np.nan
    total_sic[~valid] = np.nan
    # set to nan if total_sea ice_conc < 1%
    dominant_icetype[total_sic < 0.01] = -77
    dom_icetype_conc[total_sic < 0.01] = np.nan
    concentrations['myi'][total_sic < 0.01] = np.nan

    # ---------------------------------------------------------
    # 4) Build output dictionary
    # ---------------------------------------------------------
    result = {
        'dom_icetype': dominant_icetype.astype(np.int8),
        'dom_icetype_conc': dom_icetype_conc,
        'myi_conc': concentrations['myi'],
        'total_ice_conc': total_sic
    }
    
    return result


def autoDMI_S1_to_EASE(
        xc, yc, ice, src_crs="EPSG:3413", tgt_crs="EPSG:6931",
        extent=3_000_000, res=25_000,
        pixel_size=500.0, n_classes=5,
        chunk_rows=256,
        min_fraction_valid=0.95):

    import numpy as np
    from pyproj import Transformer
    import math
    import rasterio
    from rasterio.transform import from_origin

    # target grid
    x_centers = np.arange(-extent, extent, res) + res/2.0
    y_centers = np.arange(extent, -extent, -res) + (-res/2.0)
    x_min = -extent
    y_max = extent
    nx_tgt = x_centers.size
    ny_tgt = int((2*extent) // res)

    total_tgt_cells = nx_tgt * ny_tgt

    counts_flat = np.zeros((n_classes, total_tgt_cells), dtype=np.int64)    
    myi_counts_flat = np.zeros(total_tgt_cells, dtype=np.int64)  # NEW: Track MYI pixels
    ice_counts_flat = np.zeros(total_tgt_cells, dtype=np.int64)  # NEW: Track all ice pixels
    
    transformer = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)

    nx_src = xc.size
    ny_src = yc.size

    for r0 in range(0, ny_src, chunk_rows):
        r1 = min(ny_src, r0 + chunk_rows)

        yc_chunk = yc[r0:r1]
        Xc_chunk = np.broadcast_to(xc, (yc_chunk.size, xc.size))
        Yc_chunk = np.broadcast_to(yc_chunk[:, None], (yc_chunk.size, xc.size))

        bx = Xc_chunk.ravel()
        by = Yc_chunk.ravel()
        vals = np.nan_to_num(ice[r0:r1, :], nan=-77).ravel().astype(np.int32)

        valid_mask = vals != 0
        if not np.any(valid_mask):
            continue

        bx = bx[valid_mask]
        by = by[valid_mask]
        vals = vals[valid_mask]

        bx_tgt, by_tgt = transformer.transform(bx, by)

        col_f = (bx_tgt - x_min) / res
        row_f = (y_max - by_tgt) / res

        col_idx = np.floor(col_f).astype(np.int64)
        row_idx = np.floor(row_f).astype(np.int64)

        mask_in = (
            (col_idx >= 0) & (col_idx < nx_tgt) &
            (row_idx >= 0) & (row_idx < ny_tgt)
        )
        if not np.any(mask_in):
            continue

        col_idx = col_idx[mask_in]
        row_idx = row_idx[mask_in]
        vals    = vals[mask_in]

        flat_idx = row_idx * nx_tgt + col_idx

        # count per class
        for cls in range(1, n_classes+1):
            sel = (vals == cls)
            if not np.any(sel):
                continue
            blob = np.bincount(flat_idx[sel], minlength=total_tgt_cells)
            counts_flat[cls-1, :] += blob
            
        # Count MYI pixels (ice == 4)
        myi_sel = (vals == 4)
        if np.any(myi_sel):
            myi_blob = np.bincount(flat_idx[myi_sel], minlength=total_tgt_cells)
            myi_counts_flat += myi_blob
        
        # Count all ice pixels (ice types 1, 2, 4 - excluding 0 and invalid)
        ice_sel = np.isin(vals, [1, 2, 4])
        if np.any(ice_sel):
            ice_blob = np.bincount(flat_idx[ice_sel], minlength=total_tgt_cells)
            ice_counts_flat += ice_blob

    # reshape
    counts = counts_flat.T.reshape(ny_tgt, nx_tgt, n_classes)
    total_counts = counts.sum(axis=2)
    myi_counts = myi_counts_flat.reshape(ny_tgt, nx_tgt)  # NEW
    ice_counts = ice_counts_flat.reshape(ny_tgt, nx_tgt)  # NEW

    # expected pixels per cell (25 km / 0.5 km = 50 pixels)
    expected_pixels = int((res / pixel_size) ** 2)
    min_valid_pixels = int(expected_pixels * min_fraction_valid)

    # concentration + dominant class
    denom = np.where(total_counts == 0, 1, total_counts)
    concentration_cover = counts / denom[:, :, None]

    dominant_class = np.full((ny_tgt, nx_tgt), -77, dtype=np.int8)
    mask_nonzero = total_counts > 0
    if np.any(mask_nonzero):
        class_indices = np.argmax(counts[mask_nonzero], axis=1)
        dominant_class[mask_nonzero] = (class_indices + 1).astype(np.int8)

    dominant_concentration = np.full((ny_tgt, nx_tgt), np.nan, dtype=np.float32)
    if np.any(mask_nonzero):
        dominant_concentration[mask_nonzero] = concentration_cover[
            mask_nonzero,
            dominant_class[mask_nonzero] - 1
        ]
    dominant_concentration[~mask_nonzero] = np.nan

    myi_concentration = np.zeros((ny_tgt, nx_tgt), dtype=np.float32)
    total_pixels_mask = total_counts > 0
    if np.any(total_pixels_mask):
        myi_concentration[total_pixels_mask] = myi_counts[total_pixels_mask]/ expected_pixels
    # Clip to [0,1] and set invalid areas to NaN
    myi_concentration = np.clip(myi_concentration, 0.0, 1.0)

    # apply minimum-valid coverage rule
    valid_mask = (total_counts >= min_valid_pixels) #& (dominant_concentration >= 0.9)
    #valid_mask_myi = (total_counts >= min_valid_pixels) & (myi_concentration < 0.9) & (myi_concentration > 0.1)

    # invalid cells become nodata in all outputs
    dominant_class[~valid_mask] = -77
    concentration_cover[~valid_mask, :] = np.nan
    dominant_concentration[~valid_mask] = np.nan
    myi_concentration[~valid_mask] = np.nan  

    return {'dom_icetype':dominant_class, 'dom_icetype_conc':dominant_concentration, 'myi_conc': myi_concentration}

def add_prominent_icetype_to_gdf(gdf):
    """
    For each polygon in the GeoDataFrame, determine the most prominent ice type
    (the one with the highest combined concentration among A, B, C) and add two new columns:
    - 'SOD_out': stage of development of the most prominent ice type
    - 'SIC_out': concentration of the most prominent ice type

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame with columns 'SOD_A', 'SOD_B', 'SOD_C', 'SIC_A', 'SIC_B', 'SIC_C'

    Returns
    -------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame with added columns 'SOD_out' and 'SIC_out'
    """
    import numpy as np

    # if there is total ice concentration but no concentration in A, B, or C, set A to total
    gdf['SIC_A'] = np.where((gdf['SIC_A'] == -77) & (gdf['SOD_A'] != -77) & (gdf['SOD_B'] == -77), gdf['SIC_T'], gdf['SIC_A'])

    sod_cols = ['SOD_A', 'SOD_B', 'SOD_C']
    sic_cols = ['SIC_A', 'SIC_B', 'SIC_C']

    SOD_out = []
    SIC_out = []
    MYI_conc = []
    SYI_conc = []
    FYI_conc = []
    YI_conc = []

    for idx, row in gdf.iterrows():
        sods = np.array([row[col] for col in sod_cols])
        sics = np.array([row[col] for col in sic_cols])

        valid_all = sods != -77
        sods_all_valid = sods[valid_all]
        sics_all_valid = sics[valid_all]

        # Concentration for each type (sum, not fraction)
        myi_conc = sics_all_valid[(sods_all_valid == 3) | (sods_all_valid == 4)].sum()
        syi_conc = sics_all_valid[sods_all_valid == 3].sum()
        fyi_conc = sics_all_valid[sods_all_valid == 2].sum()
        yi_conc  = sics_all_valid[sods_all_valid == 1].sum()

        MYI_conc.append(myi_conc/100.0)
        SYI_conc.append(syi_conc/100.0)
        FYI_conc.append(fyi_conc/100.0)
        YI_conc.append(yi_conc/100.0)

        # Dominant ice type logic 
        valid_ice = (sods != -77) & (sods != 0)
        sods_ice_valid = sods[valid_ice]
        sics_ice_valid = sics[valid_ice]
        if len(sods_ice_valid) == 0:
            SOD_out.append(0)
            SIC_out.append(0)
        else:
            unique_types = np.unique(sods_ice_valid)
            sums = [sics_ice_valid[sods_ice_valid == t].sum() for t in unique_types]
            max_idx = np.argmax(sums)
            SOD_out.append(unique_types[max_idx])
            SIC_out.append(sums[max_idx])

    gdf['SOD_out'] = SOD_out
    gdf['SIC_out'] = SIC_out
    gdf['myi_conc'] = MYI_conc
    gdf['syi_conc'] = SYI_conc
    gdf['fyi_conc'] = FYI_conc
    gdf['yi_conc'] = YI_conc

    return gdf

def convert_egg_to_SoD(gdf):

    import numpy as np
    import pandas as pd
    import geopandas as gpd
    
    # get lookup tables
    SOD_lookup_numeric = ice_type_lookup()
    SIC_lookup_numeric = ice_concentration_lookup()

    # Step 1: Convert SA (stage of development A), CA (sea ice concentration of ice type A) and CT (total sea ice concentration) to numeric 
    gdf['SA'] = pd.to_numeric(gdf['SA'], errors='coerce') # stage of development
    gdf['SB'] = pd.to_numeric(gdf['SB'], errors='coerce')
    gdf['SC'] = pd.to_numeric(gdf['SC'], errors='coerce')
    gdf['CA'] = pd.to_numeric(gdf['CA'], errors='coerce') # concentration of ice type A
    gdf['CB'] = pd.to_numeric(gdf['CB'], errors='coerce')
    gdf['CC'] = pd.to_numeric(gdf['CC'], errors='coerce')
    gdf['CT'] = pd.to_numeric(gdf['CT'], errors='coerce') # total ice concentration

    # Step 2: Reduce to final numeric classes
    gdf['SOD_A'] = gdf['SA'].map(SOD_lookup_numeric)
    gdf['SOD_B'] = gdf['SB'].map(SOD_lookup_numeric)
    gdf['SOD_C'] = gdf['SC'].map(SOD_lookup_numeric)
    gdf['SIC_A'] = gdf['CA'].map(SIC_lookup_numeric)
    gdf['SIC_B'] = gdf['CB'].map(SIC_lookup_numeric)
    gdf['SIC_C'] = gdf['CC'].map(SIC_lookup_numeric)
    gdf['SIC_T'] = gdf['CT'].map(SIC_lookup_numeric)

    return gdf

def ice_type_lookup():

    SOD_lookup_numeric = {
    -9: -77,
    0: 0, # water
    55: 0,
    80: -77,
    81: 1, # YI
    82: 1,
    83: 1,
    84: 1,
    85: 1,
    86: 2, # FYI
    87: 2, # thin 
    88: 2, # thin FYI
    89: 2, # thin FYI
    91: 2, # medium FYI
    93: 2, # thick FYI
    95: 4, # old ice
    96: 3, # SYI
    97: 4, # MYI
    98: -77, #glacier ice
    99: -77, # undetermined/unknown
    }
    
    return SOD_lookup_numeric


def ice_concentration_lookup():

    SIC_lookup_numeric = {
    -9: -77, # unknown
    0: 0, # water
    1: 0, # water
    2: 0, # water
    10: 10, # 10% ice concentration
    12: 15, # 10 to 20 % ice concentration
    13: 20, # 10 to 30 % ice concentration
    20: 20, # 20% ice concentration
    23: 25, # 20 to 30 % ice concentration
    24: 30, # 20 to 40 % ice concentration
    30: 30, # 30% ice concentration
    34: 35, # 30 to 40 % ice concentration
    35: 40, # 30 to 50 % ice concentration
    40: 40, # 40% ice concentration
    45: 45, # 45% ice concentration
    46: 50, # 40 to 60 % ice concentration
    50: 50, # 50% ice concentration
    56: 55, # 50 to 60 % ice concentration
    57: 60, # 50 to 70 % ice concentration
    60: 60, # 60% ice concentration
    67: 65, # 60 to 70 % ice concentration
    68: 70, # 60 to 80 % ice concentration
    70: 70, # 70% ice concentration
    78: 75, # 70 to 90 % ice concentration
    79: 80, # 70 to 90 % ice concentration
    80: 80, # 80% ice concentration
    81: 90, # 80 to 100 % ice concentration
    89: 85, # 80 to 90 % ice concentration
    90: 90, # 90% ice concentration
    91: 95, # 90 to 100 % ice concentration
    92: 100, # 100% ice concentration
    99: -77,  # all unknown
    }
    
    return SIC_lookup_numeric

def DMI_map_lookup_to_grid(grid, keys, values):
    """
    Input:
        grid : 2d array 
        keys :  1d array of ids/values in grid
        values : 1d array of same length as keys with corresponding new values
    Output:
        mapped_grid : 2d array of same shape as input grid assigned with new values
    """
    import numpy as np

    keys = np.array(list(keys))
    values = np.array(list(values))
    
    # Mask of valid IDs
    valid_mask = grid >= 0
    
    # Create lookup table
    lookup = np.full(int(keys.max()) + 1, np.nan, dtype=values.dtype)
    lookup[keys.astype(int)] = values
    
    # Fill output grid
    mapped_grid = np.full_like(grid, np.nan, dtype=values.dtype)
    mapped_grid[valid_mask] = lookup[grid[valid_mask].astype(int)]
    
    return mapped_grid

def combined_icetype_maps_from_icecharts(raster_maps, pure = True):
    """
    Takes the dictionary with rasteritzed ice charts from the reading functions
    if pure is True: 
    they are combined into one map containing only the ice types where this type is 100% of the total SIC
    returns the SOD and corresponding SIC map
    if pure is False:
    they are combined into one map containing the most prominent ice type (the one with the highest SIC) at each grid cell
    returns the SOD and corresponding SIC map
    """
    import numpy as np
    
    # Convert all arrays in raster_maps to int16
    for key in raster_maps.keys():
        raster_maps[key] = raster_maps[key].astype(np.int16)
    
    # CA/SIC_A is set to -9 if there is only one ice tye reported for that area, i.e. there is no ice types B and C
    # thus the partial iceconcentration of type A is equal to the total ice concentration
    raster_maps['SIC_A'] = np.where((raster_maps['SIC_A'] == -77) & (raster_maps['SOD_A'] != -77) & (raster_maps['SOD_B'] == -77), raster_maps['SIC_T'], raster_maps['SIC_A'])
    # if SIC_A-SIC_T <= 0 everywhere, there is no difference between the two (as total concentration can not be smaller than the concentration of any partial concentration)
    # in that case, we only use SIC_A and SOD_A as B and C should be empty
    # it should actually be exactly 0 everywhere except in SIC_T some area is marked as open water while SIC_A has no raster_maps there
    compare = raster_maps['SIC_T']-raster_maps['SIC_A']
    compare = compare[raster_maps['SIC_A'] != -77] # remove no raster_maps values
    if np.all(compare < 1e-5):
        SOD_out = raster_maps['SOD_A']
        SIC_out = raster_maps['SIC_A']
        SOD_out[raster_maps['SIC_T'] < 90] = -77
        SIC_out[raster_maps['SIC_T'] < 90] = -77
        return SOD_out, SIC_out 
    else:
        # prepeare ouput arrays filled woth -77
        SOD_out = np.zeros_like(raster_maps['SIC_A'], dtype=np.int16)
        SOD_out[:] = -77
        SIC_out = np.zeros_like(raster_maps['SIC_A'], dtype=np.int16)
        SIC_out[:] = -77
        SOD2_out = np.zeros_like(raster_maps['SIC_A'], dtype=np.int16)
        SOD2_out[:] = -77
        SIC2_out = np.zeros_like(raster_maps['SIC_A'], dtype=np.int16)
        SIC2_out[:] = -77
        SOD3_out = np.zeros_like(raster_maps['SIC_A'], dtype=np.int16)
        SOD3_out[:] = -77
        SIC3_out = np.zeros_like(raster_maps['SIC_A'], dtype=np.int16)
        SIC3_out[:] = -77
        # Create masks for valid raster_maps (not -77)
        valid_a = raster_maps['SOD_A'] != -77
        valid_b = raster_maps['SOD_B'] != -77
        valid_c = raster_maps['SOD_C'] != -77
        
        # Get unique ice types from all SOD arrays
        all_sod_values = np.concatenate([
            raster_maps['SOD_A'][valid_a],
            raster_maps['SOD_B'][valid_b], 
            raster_maps['SOD_C'][valid_c]
        ])
        unique_ice_types = np.unique(all_sod_values[all_sod_values != -77])
        # Only consider ice types for "dominant" (exclude 0)
        ice_types_only = unique_ice_types[unique_ice_types != 0]
        
        # For each ice type, calculate total SIC across all agreements
        #preapare array to keep track of most prominent SIC found so far
        prominent_sic = np.zeros_like(raster_maps['SIC_A'], dtype=np.float32)
        prominent_sic[:] = -77  # Fill after creation
        secondary_sic = np.zeros_like(raster_maps['SIC_A'], dtype=np.float32)
        secondary_sic[:] = -77
        tertiary_sic = np.zeros_like(raster_maps['SIC_A'], dtype=np.float32)
        tertiary_sic[:] = -77
        
        for ice_type in unique_ice_types:
            # Create masks where each SOD matches this ice type
            mask_a = (raster_maps['SOD_A'] == ice_type) & valid_a
            mask_b = (raster_maps['SOD_B'] == ice_type) & valid_b
            mask_c = (raster_maps['SOD_C'] == ice_type) & valid_c
            
            # Sum SIC values for this ice type
            total_sic = np.zeros_like(raster_maps['SIC_A'], dtype=np.float32)
            total_sic += np.where(mask_a, raster_maps['SIC_A'], 0)
            total_sic += np.where(mask_b, raster_maps['SIC_B'], 0)
            total_sic += np.where(mask_c, raster_maps['SIC_C'], 0)

            # Update output where this ice type has higher SIC
            mask_better_prominent = (mask_a | mask_b | mask_c) & (total_sic > prominent_sic)
            mask_better_secondary = (mask_a | mask_b | mask_c) & (total_sic < prominent_sic) & (total_sic > secondary_sic)
            mask_better_tertiary = (mask_a | mask_b | mask_c) & (total_sic < secondary_sic) & (total_sic > tertiary_sic)
            SOD_out = np.where(mask_better_prominent, ice_type, SOD_out)
            SOD2_out = np.where(mask_better_secondary, ice_type, SOD2_out)
            SOD3_out = np.where(mask_better_tertiary, ice_type, SOD3_out)
            prominent_sic = np.where(mask_better_prominent, total_sic, prominent_sic)
            secondary_sic = np.where(mask_better_secondary, total_sic, secondary_sic)
            tertiary_sic = np.where(mask_better_tertiary, total_sic, tertiary_sic)
        
        # If all SOD_A/B/C == 0 (and not -77), set SOD_out=0, SIC_out=0
        only_water = (
            (raster_maps['SOD_A'] == 0) & (raster_maps['SOD_B'] == 0) & (raster_maps['SOD_C'] == 0)
        )
        SOD_out = np.where(only_water, 0, SOD_out)
        SIC_out = np.where(only_water, 0, SIC_out)
        
        SIC_out = np.where(prominent_sic > -77, prominent_sic, -77).astype(raster_maps['SIC_A'].dtype)
        SIC2_out = np.where(secondary_sic > -77, secondary_sic, -77).astype(raster_maps['SIC_A'].dtype)
        SIC3_out = np.where(tertiary_sic > -77, tertiary_sic, -77).astype(raster_maps['SIC_A'].dtype)
        
        
    return SOD_out, SIC_out


def read_autoDMI_to_25km(filenames, hemis):
    
    import os, glob
    import numpy as np
    from netCDF4 import Dataset as NetCDFFile
    import xarray as xr

    ds = xr.open_dataset(filenames[0], engine='h5netcdf')  # or engine='netcdf4'
    xc = ds['xc'][2000:13500].values
    yc = ds['yc'][7000:18500].values
    sod = ds["sod"][0,7000:18500,2000:13500].values
    sod[sod == 4] = np.nan # 4 is glacier ice in autoDMI
    sod[sod == 3] = 4 # 3 is MYI in autoDMI but SYI in the other products
    sod[sod == 1] = 2 # 1 and 2 are thin and thick FYI in autoDMI
    sod[sod == 0] = 1 # 0 is new and young ice in autoDMI
    
    return autoDMI_S1_to_EASE(xc, yc, sod, src_crs="EPSG:3413", tgt_crs="EPSG:6931", pixel_size = 500.0, min_fraction_valid=0.9)

def read_S1_to_25km(filenames, hemis):
    
    import os
    import numpy as np
    from netCDF4 import Dataset as NetCDFFile
    import xarray as xr

    ds = xr.open_dataset(filenames[0], engine='h5netcdf')  # or engine='netcdf4'
    xc = ds['xc'].values
    yc = ds['yc'].values
    icetype = ds['ice_type'][0,:,:]
    confidence = ds['confidence'][0,:,:].values
    # fill invalid / missing values
    fill = icetype.attrs.get('fill_value', None)
    if fill is not None:
        icetype = icetype.where(icetype != fill, np.nan)
    # mask values not in the defined flag_values (valid ice types)
    valid_flags = icetype.attrs.get('flag_values', [-1,0,1,2,3])
    icetype = icetype.where(np.isin(icetype, valid_flags), np.nan)
    # convert to NumPy array
    icetype = icetype.values
    icetype[icetype == 3] = 4 # 3 is SYI in the other products
    icetype[icetype == -1] = np.nan
    icetype[icetype == -127.0] = np.nan
        
    return autoDMI_S1_to_EASE(xc, yc, icetype, src_crs="EPSG:6931", tgt_crs="EPSG:6931", pixel_size = 1000.0)

def read_AMSR_day_to_EASE2(date, hemis):
    import glob
    import numpy as np
    from netCDF4 import Dataset as NetCDFFile
    
    AMSR_dict = {}
    read_frequencies = ['6.9H','6.9V','10.7H','10.7V','18H','18V','23V', '23H', '36V', '36H', '89V', '89H']
    name_frequencies = ['6.9H','6.9V','10.7H','10.7V','18.7H','18.7V','23.8V', '23.8H', '36.5V', '36.5H', '89V', '89H']
    for read_frequency, name_frequency in zip(read_frequencies,name_frequencies):
        for ME in ['M','E']:
            filenames = glob.glob(f'/mnt/spaces/Radiometers/AMSR2/EASE2_25km/{date[:4]}/'
                                f'NSIDC0630_*EASE2_{hemis}25km_GCOMW1_AMSR2_{ME}_{read_frequency}_{date}*v2.0.nc')
            if len(filenames) == 0:
                filenames = glob.glob(f'/mnt/spaces/Radiometers/AMSR2/EASE2_25km/{date[:4]}/'
                                f'NSIDC0630_*EASE2_{hemis}25km_GCOMW1_AMSR2_{read_frequency}_{date}*v2.0.nc')
                AMSR_dict[f'AMSR2_TB{name_frequency}'] = np.full((240, 240), np.nan)
                continue
            with NetCDFFile(filenames[0]) as data:
                TB = data.variables['TB'][0,240:480,240:480]
                TB = np.ma.filled(TB, np.nan)  
            
            AMSR_dict[f'AMSR2_TB{name_frequency}_{ME}'] = TB
            
    return AMSR_dict

def read_ERA5_day_to_EASE2(DATE, hemis, target_res=25000):
    import glob
    import natsort
    from netCDF4 import Dataset as NetCDFFile
    from pathlib import Path
    import pyresample
    from pyresample import geometry, kd_tree
    import numpy as np
    
    # get filename
    path_base = Path("/mnt/spaces/Models/ERA5/incoming/") / f"{DATE[:4]}/"
    if hemis == 'N':
        pattern = f"era5_{DATE}_250_1hr_arc.nc"
    elif hemis == 'S':
        pattern = f"era5_{DATE}_250_1hr_ant.nc"
    # load data
    file = natsort.natsorted(glob.glob(str(path_base / pattern)))[0]
    with NetCDFFile(file) as data:
        eratemp = data.variables['t2m'][:,:,:]
        eraskintemp = data.variables['skt'][:,:,:]
        eratcwv = data.variables['tcwv'][:,:,:]
        eratclw = data.variables['tclw'][:,:,:]
        eratciw = data.variables['tciw'][:,:,:]
        erau10wind = data.variables['u10'][:,:,:]
        erav10wind = data.variables['v10'][:,:,:]

    # Define EASE-Grid 2.0 North projection
    if hemis == 'N':
        ease2_proj = '+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
    else:
        ease2_proj = '+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'

    # Create coordinate arrays for EASE-Grid 2.0
    # 3000000m extent means -3000000 to +3000000 in both x and y
    # 25km resolution means 240 pixels (6000000 / 25000)
    extent = 3000000
    resolution = 25000
    size = int(2 * extent / resolution)
    x = np.linspace(-extent, extent, size)
    y = np.linspace(extent, -extent, size)  # Note: y decreases (typical for images)
    xx, yy = np.meshgrid(x, y)
    # Create EASE-Grid 2.0 geometry
    ease2_area = geometry.AreaDefinition('ease2_north', 'EASE-Grid 2.0 North', 'ease2_north',
                                        ease2_proj, xx.shape[1], xx.shape[0],
                                        [-extent, -extent, extent, extent])
    # Get latitude and longitude from NetCDF file (assuming they're in the file)
    with NetCDFFile(file) as nc_data:
        lons = nc_data.variables['longitude'][:]
        lats = nc_data.variables['latitude'][:]
    # Create meshgrid for lat/lon
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    # Define source geometry (ERA5 regular lat/lon grid)
    era5_area = geometry.SwathDefinition(lon_grid, lat_grid)
    
    # Get min and max for all variables within given day
    variables = {
        't2m': eratemp,
        'skt': eraskintemp, 
        'tcwv': eratcwv,
        'tclw': eratclw,
        'tciw': eratciw,
        'u10': erau10wind,
        'v10': erav10wind
    }

    # Calculate min and max for each variable
    era_vars = {}
    for var_name, var_data in variables.items():
        era_vars[f'{var_name}_min'] = np.min(var_data, axis=0)
        era_vars[f'{var_name}_max'] = np.max(var_data, axis=0)
        era_vars[f'{var_name}_mean'] = np.nanmean(var_data, axis=0)

    # Perform reprojection for all variables
    ease2_data = {}
    for var_key, var_array in era_vars.items():
        ease2_data[f'ERA5_{var_key}'] = kd_tree.resample_nearest(
            era5_area, var_array, ease2_area,
            radius_of_influence=50000,
            fill_value=np.nan
        )
    
    return ease2_data

def read_ASCAT_day_to_EASE2(DATE, hemis, target_res=25000):
    import glob
    import natsort
    from netCDF4 import Dataset as NetCDFFile
    from pathlib import Path
    from pyresample import geometry, kd_tree
    import numpy as np
    from pyproj import Proj, CRS
    
    # Get filename
    path_base = Path("/mnt/raid01/Scatterometers/ASCAT/nrt_products") / f"{DATE[:4]}/"
    if hemis == 'N':
        pattern = f"ASCAT_NRT_{DATE}_N*.nc"
    elif hemis == 'S':
        pattern = f"ASCAT_NRT_{DATE}_S*.nc"
    
    files = natsort.natsorted(glob.glob(str(path_base / pattern)))
    if not files:
        print(f"No ASCAT data available for {DATE}-{hemis}")
        # Return empty dict with NaN arrays for EASE2 grid
        extent = 3000000
        size = int(2 * extent / target_res)
        ease2_shape = (size, size)
        return {'ascat_S0': np.full(ease2_shape, np.nan)}
    
    # Load ASCAT data
    with NetCDFFile(files[0]) as data:
        S0 = data.variables['S0'][:]
        S0 = np.ma.filled(S0, np.nan)
    
    # Define NSIDC grid parameters for ASCAT source data
    res = 12.5  # ASCAT is on 12.5km NSIDC grid
    
    if hemis == "N":
        Projection = Proj(CRS.from_epsg(3413))
        ymin, ymax = -5350, 5850
        xmin, xmax = -3850, 3750
    else:
        Projection = Proj(CRS.from_epsg(3976))
        ymin, ymax = -3950, 4350
        xmin, xmax = -3950, 3950
    
    # Create NSIDC coordinate grid
    y, x = np.mgrid[ymin + res/2:ymax:res, xmin + res/2:xmax:res] * 1000
    lon, lat = Projection(x, y, inverse=True)
    
    # Define source geometry (ASCAT NSIDC grid)
    ascat_area = geometry.SwathDefinition(lon, lat)
    
    # Define EASE-Grid 2.0 projection
    if hemis == 'N':
        ease2_proj = '+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
    else:
        ease2_proj = '+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'
    
    # Create EASE-Grid 2.0 geometry
    extent = 3000000
    if hemis=="N":
        Projection=Proj(CRS.from_epsg(6931)) 
        xx,yy=np.mgrid[-extent:extent:target_res,
                    -extent:extent:target_res]
    else:
        Projection=Proj(CRS.from_epsg(6932)) 
        xx,yy=np.mgrid[-extent:extent:target_res,
                    -extent:extent:target_res]
    
    ease2_area = geometry.AreaDefinition(
        f'ease2_{hemis.lower()}', 
        f'EASE-Grid 2.0 {hemis}', 
        f'ease2_{hemis.lower()}',
        ease2_proj, 
        xx.shape[1], xx.shape[0],
        [-extent, -extent, extent, extent]
    )
    
    # Perform reprojection
    ease2_data = kd_tree.resample_nearest(
            ascat_area, S0, ease2_area,
            radius_of_influence=50000,
            fill_value=np.nan
        )
    
    return {'ASCAT_S0': ease2_data}
