"""
Use this to collect relevant methods for MSEIS ML.
"""
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio
#import datetime
import cv2
from scipy.signal import find_peaks
from obspy import read

# --- open rdv file

def open_rdv_file(file_path):
    """return a dataframe"""

    # determine filenames 

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
    return pd.DataFrame(conditioned_array, columns=['JobTime','MSec','ID','TYPE','NARR','SNR','LOC_X','LOC_Y','LOC_Z','DIST','MAG','RADIUS','MOMENT','APEX','APEX MD'] )

# ---- open xyz

def open_xyz_file(file_path):
    """return a dataframe"""
    cols = ['MD','Easting','Northing','TVD_msl','Channel', 'Field Channel']

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
    return pd.DataFrame(conditioned_array, columns=cols)

# ---- segy to image singletons  ----------------

def parse_trace_headers(segyfile, n_traces):
    '''
    Parse the segy file trace headers into a pandas dataframe.
    Column names are defined from segyio internal tracefield
    One row per trace
    '''
    # Get all header keys
    headers = segyio.tracefield.keys
    #print(headers)
    # Initialize dataframe with trace id as index and headers as columns
    df = pd.DataFrame(index=range(1, n_traces + 1),
                      columns=headers.keys())
    # Fill dataframe with all header values
    for k, v in headers.items():
        df[k] = segyfile.attributes(v)[:]
    return df

def parse_text_header(segyfile):
    '''
    Format segy text header into a readable, clean dict
    '''
    raw_header = segyio.tools.wrap(segyfile.text[0])
    # Cut on C*int pattern
    cut_header = re.split(r'C ', raw_header)[1::]
    # Remove end of line return
    text_header = [x.replace('\n', ' ') for x in cut_header]
    text_header[-1] = text_header[-1][:-2]
    # Format in dict
    clean_header = {}
    i = 1
    for item in text_header:
        key = "C" + str(i).rjust(2, '0')
        i += 1
        clean_header[key] = item
    return clean_header


def make_sgy_array(file_path):
    """returns numpy array of values"""

    with segyio.open(file_path, ignore_geometry=True) as f:
        # Get basic attributes
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000
        n_samples = f.samples.size
        twt = f.samples
        data = f.trace.raw[:]  # Get all data into memory (could crash on big files)

        # Load headers
        bin_headers = f.bin
        #text_headers = parse_text_header(f)
        trace_headers = parse_trace_headers(f, n_traces)
        
        f.close()
        
    f'N Traces: {n_traces}, N Samples: {n_samples}, Sample rate: {sample_rate}ms'
    
    # remove major noise band, maybe seismic 

    clip_percentile = 99 # most robust is 99 

    vm = np.percentile(data, clip_percentile)
    f'The {clip_percentile}th percentile is {vm:.0f}; the max amplitude is {data.max():.0f}'
    extent = [1, n_traces, twt[-1], twt[0]]  # good extent
    
    return data

def condition_segy(data):
    """Input should be numpy array"""
    
    # remove major noise band, maybe seismic 
    data[0:25,:] = 0
    #data[0:150,:] = 0 # remove early channels where lines of noise persist
    #data[:,0:150] = 0 # remove precursing noise to P and S wave arrival

    x = np.abs(np.mean(data, axis=1))
    #summy = np.abs(np.sum(data, axis=1))
    std_factor = 1.5*np.std(x) # 1.2 before, 3.75 is good
    peaks, _ = find_peaks(x, height=std_factor)
    
    # remove lines 
    
    data_new = np.asarray(data, dtype=np.float64)
    w = 3
    for pk in peaks:
        if pk < 200:
            # set amp to zero, it's noise
            data_new[pk-w:pk+w,:] = np.zeros(np.shape(data_new[pk-w:pk+w,:]))
            
    return data_new

def make_image_from_numpy(d_t):
    """take 2D numpy array, turn into formatted image"""
    image_mat = np.array(d_t, dtype = np.float64)
    image_mat *= (255.0/image_mat.max()) # normalize 
    image_mat = np.abs(image_mat) # use abs
    return image_mat.astype(dtype=np.uint8) # return as ints

def show_conditioned_sgy_image(file_path):
    """
    Use this routine to visualize how we filter 
    channels with too much noise and renormalize the signal.
    """
    
    with segyio.open(file_path, ignore_geometry=True) as f:
        # Get basic attributes
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000
        n_samples = f.samples.size
        twt = f.samples
        data = f.trace.raw[:]  # Get all data into memory (could cause on big files)

        # Load headers
        bin_headers = f.bin
        #text_headers = parse_text_header(f)
        trace_headers = parse_trace_headers(f, n_traces)
        
        f.close()
        
    f'N Traces: {n_traces}, N Samples: {n_samples}, Sample rate: {sample_rate}ms'
    
    # remove major noise band, maybe seismic 
    data[0:10,:] = 0

    clip_percentile = 99 # 99 gives most clear. 

    vm = np.percentile(data, clip_percentile)
    f'The {clip_percentile}th percentile is {vm:.0f}; the max amplitude is {data.max():.0f}'
    extent = [1, n_traces, twt[-1], twt[0]]  # good extent
        
    x = np.abs(np.mean(data, axis=1))
    summy = np.abs(np.sum(data, axis=1))

    std_factor = 1.2*np.std(x) # 3.75 is good
    peaks, _ = find_peaks(x, height=std_factor)
    
    # remove lines 
    
    data_new = np.asarray(data, dtype=np.float64)
    w = 3
    for pk in peaks:
        if pk < 200:
            #print('peak at: ', pk)
            # set amp to zero, it's noise
            data_new[pk-w:pk+w,:] = np.zeros(np.shape(data_new[pk-w:pk+w,:]))
            
    i_1 = make_image_from_numpy(data)
    i_2 = make_image_from_numpy(data_new)
    
    fig, ax = plt.subplots(nrows=7, ncols=1,figsize=(10,20), dpi=100)

    c = ax[0].imshow(i_1, cmap="RdBu", vmin=0,vmax=255,aspect='auto')
    ax[2].imshow(i_2, cmap="RdBu", vmin=0,vmax=255,aspect='auto')
    
    ax[1].plot(peaks, x[peaks], "x")
    ax[1].plot(x, label='unfiltered')
    ax[1].axhline(y = std_factor, color ="green", linestyle =":")
    
    x_n = np.abs(np.mean(data_new, axis=1))
    ax[1].plot(x_n, label='filtered')
    ax[1].legend()

    #fig.colorbar(c, ax = ax)
    
    ax[0].set_title('before')
    ax[1].set_title('cols removed ')
    ax[2].set_title('after filtering')
    
    ax[3].imshow(i_1, cmap="RdBu", vmin=0,vmax=255,aspect='auto')
    ax[4].imshow(i_2, cmap="RdBu", vmin=0,vmax=255,aspect='auto')
    
    ax[3].set_ylim([200,300])
    ax[4].set_ylim([200,300])
    
    y_min = 250
    y_max = 700
    
    x_min = 100
    x_max = 300
    
    window = data_new[x_min:x_max,y_min:y_max]
    sobelxy = cv2.Sobel(src=window, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

    heatmapshow = None
    heatmapshow = cv2.normalize(sobelxy, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
    
    ax[5].imshow(sobelxy, cmap="RdBu", vmin=0,vmax=255,aspect='auto')
    ax[6].imshow(heatmapshow, aspect='auto')

    
    plt.show()

def make_raw_sgy_array(file_path):
    
    with segyio.open(file_path, ignore_geometry=True) as f:
        # Get basic attributes
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000
        n_samples = f.samples.size
        twt = f.samples
        data = f.trace.raw[:]  # Get all data into memory (could cause on big files)

        # Load headers
        bin_headers = f.bin
        #text_headers = parse_text_header(f)
        trace_headers = parse_trace_headers(f, n_traces)
        
        f.close()
        
    f'N Traces: {n_traces}, N Samples: {n_samples}, Sample rate: {sample_rate}ms'
    
    # remove major noise band, maybe seismic 

    clip_percentile = 99 # 99 gives most clear. 

    vm = np.percentile(data, clip_percentile)
    f'The {clip_percentile}th percentile is {vm:.0f}; the max amplitude is {data.max():.0f}'
    extent = [1, n_traces, twt[-1], twt[0]]  # good extent

    return data
    
def make_raw_sgy_image(file_path):
    
    data = make_raw_sgy_array(file_path)

    return make_image_from_numpy(data)

def make_conditioned_sgy_array(file_path):
    st =read(file_path,format='SEGY')
    st.detrend("linear")
    st.taper(max_percentage=0.01, type="hann")
    """old, works but we're using routine provided by Bryan"""
    
    data, fs = stream_to_numpy(st, normalise=False)  
    
    # remove major noise band, maybe seismic 
    data[0:10,:] = 0

    clip_percentile = 99 # 99 gives most clear. 

    vm = np.percentile(data, clip_percentile)
    f'The {clip_percentile}th percentile is {vm:.0f}; the max amplitude is {data.max():.0f}'
    #extent = [1, n_traces, twt[-1], twt[0]]  # good extent
        
    x = np.abs(np.mean(data, axis=1))
    summy = np.abs(np.sum(data, axis=1))

    std_factor = 1.2*np.std(x) # 3.75 is good
    peaks, _ = find_peaks(x, height=std_factor)
    
    # remove lines 
    
    data_new = np.asarray(data, dtype=np.float64)
    w = 3
    for pk in peaks:
        if pk < 200:
            # set amp to zero, it's noise
            data_new[pk-w:pk+w,:] = np.zeros(np.shape(data_new[pk-w:pk+w,:]))
            
    return data_new

    
def make_conditioned_sgy_image(file_path):
    
    with segyio.open(file_path, ignore_geometry=True) as f:
        # Get basic attributes
        n_traces = f.tracecount
        sample_rate = segyio.tools.dt(f) / 1000
        n_samples = f.samples.size
        twt = f.samples
        data = f.trace.raw[:]  # Get all data into memory (could cause on big files)

        # Load headers
        bin_headers = f.bin
        #text_headers = parse_text_header(f)
        trace_headers = parse_trace_headers(f, n_traces)
        
        f.close()
        
    f'N Traces: {n_traces}, N Samples: {n_samples}, Sample rate: {sample_rate}ms'
    
    # remove major noise band, maybe seismic 
    data[0:10,:] = 0

    clip_percentile = 99 # 99 gives most clear. 

    vm = np.percentile(data, clip_percentile)
    f'The {clip_percentile}th percentile is {vm:.0f}; the max amplitude is {data.max():.0f}'
    extent = [1, n_traces, twt[-1], twt[0]]  # good extent
        
    x = np.abs(np.mean(data, axis=1))
    summy = np.abs(np.sum(data, axis=1))

    std_factor = 1.2*np.std(x) # 3.75 is good
    peaks, _ = find_peaks(x, height=std_factor)
    
    # remove lines 
    
    data_new = np.asarray(data, dtype=np.float64)
    w = 3
    for pk in peaks:
        if pk < 200:
            # set amp to zero, it's noise
            data_new[pk-w:pk+w,:] = np.zeros(np.shape(data_new[pk-w:pk+w,:]))
            
    return make_image_from_numpy(data_new)

def stream_to_numpy(st,normalise=False):
    """routine provided by Bryan"""
    # Creates a numpy array for stream
    fs=st[0].stats.sampling_rate
    data_length=st[0].stats.npts
    out_data=np.zeros((len(st),data_length))
    
    # Integrate to convert to displacement from velocity
    #     st=st.integrate()
    
    for i in range(len(st)):
        in_data=st[i].data
        if normalise==True:
            # Normalise the data
            tmp=in_data/np.max(in_data)
            out_data[i]=tmp-np.mean(tmp)
        elif normalise==False:
            out_data[i]=in_data-np.mean(in_data)
        else:
            print("Would you like to normalise the data? True of False?")
            break
    return out_data, fs


# ----- grab information from pk files ----------------

def convert_pk_file_to_dataframe(file_path):
    """general routine. """

    # convert entire file into array, parse from here

    with open(file_path) as file:
        data_array = file.readlines()

    for i in range(0,len(data_array)):

        # look for sgy file
        peek = data_array[i][0:3]

        if peek == 'D:\\':

            s = data_array[i].split(' ') 
            sgy_file_path = s[0]
            print('sgy file: ', sgy_file_path)

            origin_time_line = data_array[i+1].split(' ')
            
            # line is not consistently formatted 

            if len(origin_time_line) == 6:

                event_number = origin_time_line[2]
                origin_time = origin_time_line[4]
                time_stamp = origin_time_line[5][:-1] # last portion has '\n'

            else:

                event_number = origin_time_line[2]
                origin_time = origin_time_line[5]
                time_stamp = origin_time_line[6][:-1] # last portion has '\n'

            print('event number: ', event_number)
            print('origin time: , ', origin_time)
            print('time stamp: ', time_stamp)

            # bulk of data
            
            columns = ['ich','ipflg','isflg','pat','sat','pmod','smod','p_amp','s_amp']

            data_below_file = data_array[i+1][:]

            pk_vals = []
            a = np.zeros(len(columns))
            a = np.reshape(a, (len(columns),1))
            #300 channels, repeated with 3 lines of header
            for j in range(i+3,i+3+300):

                row_vals = re.sub('\s+', ',', data_array[j][:])
                row_vals = row_vals.split(',')
                row_vals = row_vals[1:] # remove first col, blanks
                row_vals = row_vals[:-1] # remove last col, blanks
                pk_vals.append(np.asarray(row_vals))

            conditioned_array = np.array(pk_vals)
            df = pd.DataFrame(conditioned_array, columns=columns)
            # df.to_csv('file_' + str(i) + '.csv')
            print(df)
            
            return df, sgy_file_path, event_number, origin_time, time_stamp
        
# extras

def mask_to_pixel(mask, channels, arrival_time, category):
    """this is not the ideal method, need to map two arrays to coordinates in matrix"""
    for i in range(len(channels)):
        for j in range(len(arrival_time)):
            
            mask[j,i] = int(category)
    
    return mask