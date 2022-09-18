"""Use this to visualize and overlay pick data from Geophys"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from c import ROOT_DIR

os.chdir(ROOT_DIR + "\\src") ## go back to source 

from mseisML_utils import make_image_from_numpy, make_conditioned_sgy_array

def main():

    # get current dir

    print(os.getcwd())

    # change directory to segy data folder

    os.chdir(ROOT_DIR)

    segy_file_path = ROOT_DIR + '\\segy\\AAstg01\\06_04_2022_MDT_00670evt.sgy'
    pick_file_path = ROOT_DIR +  '\\pk_files\\file_0.csv'

    # look at example pick set

    df = pd.read_csv(pick_file_path)
    
    # set zero values to nans

    df.loc[df['pat'] == 0] = np.nan
    df.loc[df['sat'] == 0] = np.nan

    # overlay p and s arrival time lines on SGY

    channels = df['ich'].to_numpy()

    ot_offset = 0.270 # may need to multiply by 1000!
    # some picks seem to not consider origin time offset...

    p_at = df['pat'].to_numpy() * 1000 + ot_offset
    s_at = df['sat'].to_numpy() * 1000 + ot_offset

    conditioned_segy = make_conditioned_sgy_array(segy_file_path)
    segy_image = make_image_from_numpy(conditioned_segy)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6,6), dpi=256) 
    ax1.set_title(segy_file_path)
    ax1.imshow(segy_image, cmap="magma", vmin=0,vmax=255,aspect='auto')

    ax2.imshow(segy_image, cmap="magma", vmin=0,vmax=255,aspect='auto')
    ax2.scatter(p_at, channels, label='P arrival', color='blue', alpha=0.15)
    ax2.scatter(s_at, channels, label='S arrival', color='red', alpha=0.15)

    ax2.set_xlabel('Elapsed time, ms\n',fontsize=10)
    ax2.set_ylabel('Channel\n',fontsize=10)
    ax2.set_title('pick overlay: ', fontsize=10)
    plt.show()

if __name__ == "__main__":
    main() 

