"""
Use this to generate dataset for DL study
"""
from c import ROOT_DIR
from mseisML_utils import *

def main():

    os.chdir(ROOT_DIR + "\\pk_files") # change directory to data from Bryan dev

    sgy_data_folder_path = ROOT_DIR + "\\segy"
    ml_data_path = ROOT_DIR + "\\MLdata"

    pk_files = os.listdir(os.getcwd())
    # Loop to print each filename separately
    for filename in pk_files:
        print('opening file: ', filename)
        
        # convert entire file into array, parse from here
        #with open(file_path) as file:

        with open(filename) as file:
            data_array = file.readlines()

        for i in range(0,len(data_array)):

            # look for sgy file
            peek = data_array[i][0:3]

            if peek == 'D:\\':

                s = data_array[i].split(' ') 
                old_sgy_file_path = s[0]

                # remove old path
                s_n = os.path.splitext(old_sgy_file_path)[0]

                s_n = s_n.split('\\')

                sgy_file_path = s_n[-1]+'.sgy'
                print('just filename', sgy_file_path)

                origin_time_line = data_array[i+1].split(' ')
                print(origin_time_line)

                # line is not consistently formatted 

                if len(origin_time_line) == 6:

                    event_number = origin_time_line[2]
                    origin_time = origin_time_line[4]
                    time_stamp = origin_time_line[5][:-1] # last portion has '\n'

                else:

                    event_number = origin_time_line[2]
                    origin_time = origin_time_line[5]
                    time_stamp = origin_time_line[6][:-1] # last portion has '\n'

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

                # view segy image
                
                folder = os.path.splitext(filename)[0]            
                final_sgy_pth = sgy_data_folder_path + '\\' + folder + '\\' + sgy_file_path
                
                # some sgy are not available
                
                try:
                    #handle the file
                    
                    print('sgy path: ', final_sgy_pth)
                    segy_array  = make_sgy_array(final_sgy_pth)
                    conditioned_segy = condition_segy(segy_array)
                    #segy_image = make_image_from_numpy(segy_array)

                    segy_image = make_image_from_numpy(conditioned_segy)

                    print('shape of image, ', np.shape(segy_image))

                    # prepare pk data overlay

                    df = df.astype({'pat':'float'})
                    df = df.astype({'sat':'float'})
                    df = df.astype({'ich':'int'})
                    
                    # only show certain channels in mask.
                    
                    df = df[df.ich > 200] # only plot for channels 200 - 300
                    df = df[df.ich < 298] # last channels offset everything

                    # set zero values to nans

                    df.loc[df['pat'] == 0] = np.nan
                    df.loc[df['sat'] == 0] = np.nan

                    ot_offset = float(origin_time) 
                    channels = df['ich'].to_numpy()
                    p_at = df['pat'].to_numpy() 
                    s_at = df['sat'].to_numpy()

                    #p_at = 1000*(p_at) - ot_offset
                    #s_at = 1000*(s_at) - ot_offset
                    
                    p_at = 1000*(p_at)
                    s_at = 1000*(s_at)
                    channels = channels + np.abs(ot_offset)

                    print('ot_offset', ot_offset)
                    print('shape of channels', np.shape(channels))
                    print('shape of p_at', np.shape(p_at))
                    print('shape of s_at', np.shape(s_at))

                    #-----mask dev -------
                    
                    #mask = np.zeros_like(segy_image) # anything else is 0               
                    #mask = mask_to_pixel(mask, channels, p_at, 1) # assign 1 for primary
                    #mask = mask_to_pixel(mask, channels, s_at, 2) # assign 2 for secondary
                    
                    # ----------- OUTPUT MASK 

                    #plt.figure(figsize=(8,6), dpi=256) 
                    #plt.imshow(segy_image, cmap="magma", vmin=0,vmax=255,aspect='auto')

                    #plt.xlabel('Elapsed time, ms\n',fontsize=16)
                    #plt.ylabel('Channel\n',fontsize=16)
                    #plt.yticks(fontsize=14)
                    #plt.xticks(fontsize=14)
                    
                    marker_size = 25

                    #plt.scatter(p_at, channels, label='P arrival', color='blue', alpha=0.15, s=marker_size)
                    #plt.scatter(s_at, channels, label='S arrival', color='red', alpha=0.15, s=marker_size)

                    #ax = plt.gca()
                    #ax.axes.xaxis.set_visible(False)
                    #ax.axes.yaxis.set_visible(False)
                    #plt.grid(False)

                    #plt.savefig(ml_data_path+ '\\' + sgy_file_path[:-3] + '.png')
                    #plt.show()
                    
                    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6,6), dpi=256) 
                    ax1.imshow(segy_image, cmap="magma", vmin=0,vmax=255,aspect='auto')
                    #ax2.imshow(mask, cmap="magma", vmin=0,vmax=2,aspect='auto')
                    
                    ax2.set_facecolor('black')
                    ax2.scatter(p_at, channels, label='P arrival', color='blue', alpha=1, s=marker_size)
                    ax2.scatter(s_at, channels, label='S arrival', color='red', alpha=1, s=marker_size)
                    ax2.set_xlim([0,799])
                    ax2.set_ylim([299,0])
                    
                    ax1.set_title("MSEIS ML data: " + final_sgy_pth, fontsize=6)
                    #ax2.set_title("mask for: " + final_sgy_pth)
                    
                    ax3.imshow(segy_image, cmap="magma", vmin=0,vmax=255,aspect='auto')
                    ax3.scatter(p_at, channels, label='P arrival', color='blue', alpha=0.15, s=marker_size)
                    ax3.scatter(s_at, channels, label='S arrival', color='red', alpha=0.15, s=marker_size)
                    
                    plt.show()


                except OSError:
                    print('file not found - ', final_sgy_pth)

if __name__ == "__main__":
    main()