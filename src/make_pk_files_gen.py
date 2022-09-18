
"""
Read pk files into dataframes
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from c import ROOT_DIR

os.chdir(ROOT_DIR + "\\pk_files") # change directory to data from Bryan dev


print(os.getcwd())

file_path = 'AAstg01.pk'
columns = ['ich','ipflg','isflg','pat','sat','pmod','smod','p_amp','s_amp']

def main():

    # convert entire file into array, parse from here

    with open(file_path) as file:
        data_array = file.readlines()
        
    for i in range(0,len(data_array)):
        
        # look for sgy file
        peek = data_array[i][0:3]
        
        if peek == 'D:\\':
            
            s = data_array[i].split(' ')
            sgy_file = s[0]
            
            # data below
            
            data_below_file = data_array[i+1][:] # grab OT from here
            
            pk_vals = []
            a = np.zeros(len(columns))
            a = np.reshape(a, (len(columns),1))
            # scrape vals for dataframe
            for j in range(i+3,i+3+300):
                
                row_vals = re.sub('\s+', ',', data_array[j][:])
                row_vals = row_vals.split(',')
                row_vals = row_vals[1:] # remove first col, blanks
                row_vals = row_vals[:-1] # remove last col, blanks
                
                pk_vals.append(np.asarray(row_vals))
                
            conditioned_array = np.array(pk_vals)

            df = pd.DataFrame(conditioned_array, columns=columns)
            df.to_csv('file_' + str(i) + '.csv')

    # look at example pick set

    df = pd.read_csv('file_0.csv')
    print(df)
    df.plot(x="ich", y=["pat", "sat", "pmod", "smod"])
    plt.show()
    

if __name__ == "__main__":
    main()