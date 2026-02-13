import os, sys
import argparse
import numpy as np
from glob import glob
import xarray as xr
from datetime import datetime
import pyresample as pr
from tqdm import tqdm
from snapshot_functions import *

func_path = os.path.abspath(os.path.join('..','rrdp_algos/rrdp-ity-algos'))
print(func_path)
if func_path not in sys.path:
    sys.path.append(func_path)
import osi_sic 

def main():
    """
    Main function that parses command-line arguments, performs sanity checks, and applies the RRDP algorithms.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Apply the algoirithms on the CCI SAGE RRDP.")

    # Add arguments
    parser.add_argument("input_dir", type=str, help="Path to the input directory (must exist).")
    parser.add_argument("output_dir", type=str, help="Path to the output directory (will be created if it doesn't exist).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")

    # Parse arguments
    args = parser.parse_args()

    # Check if the input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)  # Exit with an error status

    # find all the interesting RRDP files
    rrdp_patt = os.path.join(args.input_dir,'*.nc')
    rrdp_files = sorted(glob(rrdp_patt))
    if len(rrdp_files) == 0:
        print(f"Error: Found no RRDP files in '{args.input_dir}'.")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        if args.verbose:
            print(f"Output directory '{args.output_dir}' does not exist. Creating it...")
        os.makedirs(args.output_dir)
    elif args.verbose:
        print(f"Output directory '{args.output_dir}' already exists.")

    if args.verbose:
        print("All checks passed. Script is ready to proceed.")

    # Go through each input RRDP file, load it, apply the algos, write the modified version in output_dir
    for rrdp_file_in in tqdm(rrdp_files, desc='Adding OSISAF_sic to RRDP'):

        if args.verbose:
            print(f"Process '{rrdp_file_in}'.")

        # Load the file with xarray
        ds = xr.open_dataset(rrdp_file_in)
        
        # Get the date from the filename (can later take it from the 'time' coordinate variable)
        #dtstr, area = os.path.basename(rrdp_file_in).replace('.nc','').split('_')
        dtstr = ds.date
        hemis = ds.hemisphere
        dt = datetime.strptime(dtstr,'%Y%m%d').date()
        # Load grid area definition
        trg_adef, _ = pr.utils.load_cf_area(rrdp_file_in)
        
        # Access OSI SAF SIC, remap it to the area of the RRDP snapshot
        v_n = 'OSISAF_sic'
        if v_n not in ds.variables:
            if args.verbose:
                print(f'##### adding OSISAF_sic to file')
            sic, sic_url, sic_source = osi_sic.remap_sic(dt, hemis.lower()+'h', trg_adef)
            v_da = xr.DataArray(sic[None,:].astype(np.float32), dims=('time', 'xc', 'yc'), name=v_n,
                    attrs={
                    'standard_name': 'sea_ice_area_fraction', 'cell_methods': 'area:mean where sea',
                    'long_name': 'Total sea ice concentration from EUMETSAT OSI SAF',
                    'source': sic_source,
                    'osisaf_file': sic_url,
                    'grid_mapping': 'crs'})
            ds[v_n] = v_da
        elif args.verbose:
            print(f"Variable '{v_n}' already exists in file '{rrdp_file_in}', skipping addition.")
            continue

        # Determine output file path
        out_path = os.path.join(args.output_dir, os.path.basename(rrdp_file_in))
        if args.output_dir == args.input_dir:
            if args.verbose:
                print(f"Warning: Overwriting input file '{out_path}'.")
            # Write to a temporary file first, then move to avoid corruption
            tmp_path = out_path + '.tmp'
            ds.to_netcdf(tmp_path)
            os.replace(tmp_path, out_path)
        else:
            ds.to_netcdf(out_path)


if __name__ == "__main__":
    main()