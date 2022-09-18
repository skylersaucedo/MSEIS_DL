
"""
Use this to attach MD info to channels
"""
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from obspy import read

from c import ROOT_DIR

os.chdir(ROOT_DIR + "\\src") ## go back to source 

from mseisML_utils import make_conditioned_sgy_array


def main():

    # change directory to data from Bryan dev

    os.chdir(ROOT_DIR + "\\xyz_files")

    file_path = 'OVV_Z_GridN.xyz'

    # convert entire file into array, parse from here

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

    os.chdir("C:\\Users\\sauce\\OneDrive\\Desktop\\mseisML")

    file = r"C:\Users\sauce\OneDrive\Desktop\mseisML\segy\ABstg01\06_03_2022_MDT_00014evt.sgy"

    data_c = make_conditioned_sgy_array(file)
        
    vMine = np.percentile(data_c,10.0)
    vMaxe = np.percentile(data_c,90.0)

    # Add MD to y-axis values

    channel_vals = ['1', '50', '100', '150', '200', '250', '300']

    # find vals 

    df_md = df.loc[df['Channel'].isin(channel_vals)]
    df_md

    md_list = df_md['MD'].tolist()
    print(md_list)

    # attach list vals to image

    channel_ints = [int(i) for i in channel_vals]

    y_pos = channel_ints

    plt.figure(figsize=(8,6)) 
    plt.imshow(data_c, aspect='auto', vmin=-abs(vMine),vmax=abs(vMaxe), cmap="gray")
    plt.xlabel('Time sample\n',fontsize=16)
    plt.ylabel('MD\n',fontsize=16)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.title(file)
    plt.tight_layout()
    plt.yticks(y_pos, md_list, color='black', rotation=0, fontweight='normal', fontsize='16', horizontalalignment='right')

    plt.show()




if __name__ == "__main__":
    main()