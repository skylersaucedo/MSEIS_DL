"""
Use this to open .xyz files
"""
import os
import re
import numpy as np
import pandas as pd

from c import ROOT_DIR

# change directory to data from Bryan dev

os.chdir(ROOT_DIR + "\\xyz_files")

file_path = 'OVV_Z_GridN.xyz'

def main():
    #
    with open(file_path) as file:
        data_array = file.readlines()

    conditioned_array = []
    skip_rows = 1
    for i in range(skip_rows,len(data_array)):
        
        row_vals = re.sub('\s+', ',', data_array[i])
        row_vals = row_vals.split(',')
        
        # remove first and last column, it's a blank
        row_vals = row_vals[1:]
        row_vals = row_vals[:-1]
        
        conditioned_array.append(row_vals)
        
    conditioned_array = np.array(conditioned_array)
    df = pd.DataFrame(conditioned_array, columns=['MD','Easting','Northing','TVD_msl','Channel', 'Field Channel'] )

    print(df)

if __name__ == "__main__":
    main()