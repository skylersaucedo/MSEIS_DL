import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from c import ROOT_DIR
os.chdir(ROOT_DIR + "\\rdv_files_xyz_of_events")

file_path = 'AAstg01.rdv'

def main():

    with open(file_path) as my_file:
        for line in my_file:        
            if line[0:3] == "D:\\":
                
                l = line.split('.')
                ext_long = l[1].split(' ')
                ext = ext_long[0]
                
                
                if ext == 'dat':
                    
                    s = line.split(' ')
                    dat_file = s[0]
                    print(dat_file)
                    
                elif ext == 'sgy':
                    
                    s = line.split(' ')
                    sgy_file = s[0]
                    print(sgy_file)
                    
                elif ext == 'pk':
                    s = line.split(' ')
                    pk_file = s[0]
                    print(pk_file)
                    
                else:
                    print('not sure...')


    # convert entire file into array, parse from here

    with open(file_path) as file:
        data_array = file.readlines()
        
    # reformat values before we can fit into a dataframe
    skip_rows = 14

    conditioned_array = []

    for i in range(skip_rows,len(data_array)):
        
        row_vals = re.sub('\s+', ',', data_array[i])
        row_vals = row_vals.split(',')
        
        # remove last column, it's a blank
        
        row_vals = row_vals[:-1]
        conditioned_array.append(row_vals)
        
    conditioned_array = np.array(conditioned_array)
    df = pd.DataFrame(conditioned_array, columns=['JobTime','MSec','ID','TYPE','NARR','SNR','LOC_X','LOC_Y','LOC_Z','DIST','MAG','RADIUS','MOMENT','APEX','APEX MD'] )

    print(df)


if __name__ == "__main__":
    main()