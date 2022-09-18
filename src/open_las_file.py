"""
Use this to open LAS
"""
from welly import Well
from welly import Curve
import os
import matplotlib.pyplot as plt
import numpy as np

from c import ROOT_DIR

def main():
    #get current dir

    print(os.getcwd())

    # change directory to segy data folder

    os.chdir(ROOT_DIR + "\\sonic_logs\\")

    print(os.getcwd())
    file_path = '100162707209W600.las'

    well = Well.from_las(file_path, index='ft')
    print(well.header)
    print(well._get_curve_mnemonics())

    tracks = ['MD', 'DT_EXPORT_REV', 'DTS_EXPORT_REV']
    well.plot(tracks=tracks)


if __name__ == "__main__":
    main()