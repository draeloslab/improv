import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import pandas as pd
import sys

import warnings
warnings.filterwarnings('ignore')

def actor_timestamps(timestamps_path):
    time_df = pd.read_csv(timestamps_path, header=None, delimiter=' ', names=['date', 'time', 'frame'])
    time_df['datetime'] = time_df['date'] + ' ' + time_df['time']
    time_df = time_df.drop(columns=['date', 'time'])
    time_df_cleaned = time_df.drop_duplicates(subset='frame', keep='last')

    time_df_cleaned['datetime'] = pd.to_datetime(time_df_cleaned['datetime'], format='%Y-%m-%d %H:%M:%S.%f')
    time_df_cleaned['delta_time']= time_df_cleaned['datetime'].diff().dt.total_seconds()

    if 'pstim' in timestamps_path:
        time_df_cleaned['delta_frame']= time_df_cleaned['frame'].diff()
        time_df_final = time_df_cleaned[time_df_cleaned['delta_frame'] == 1][['frame', 'datetime', 'delta_time']]
        return time_df_final
    else:
        return time_df_cleaned
    

def mem_from_log(log_path):
    times = []
    actors = []
    mems = []
    with open(log_path, 'r') as log_file:
        for line in log_file:
            if "mem: " in line:
                parts = line.split("mem: ")
                if len(parts) > 1:
                    time = parts[0].split(' ')[1]
                    actor, mem = parts[1].strip().split()[0][2:-2], float(parts[1].strip().split()[1][:-1])
                    times.append(time)
                    actors.append(actor)
                    mems.append(mem)

    if len(mems) == 0:
        print('No memory was tracked during this experiement!')

    else:
        mem_usage = pd.DataFrame({'Time': times, 'Actors': actors, 'Memory':mems})
        mem_usage = mem_usage.drop_duplicates().reset_index(drop=True)

        acquirer_mem = mem_usage.loc[mem_usage['Actors'] == 'Acquirer']
        analysis_mem = mem_usage.loc[mem_usage['Actors'] == 'Analysis']
        processor_mem = mem_usage.loc[mem_usage['Actors'] == 'Processor']
        microscope_mem = mem_usage.loc[mem_usage['Actors'] == 'Microscope'] 


        plt.figure(figsize=(15, 5))
        plt.plot(acquirer_mem['Time'], acquirer_mem['Memory'], label='Acquirer', marker='o')
        plt.plot(analysis_mem['Time'], analysis_mem['Memory'], label='Analysis', marker='o')
        plt.plot(processor_mem['Time'], processor_mem['Memory'], label='Processor', marker='o')
        plt.plot(microscope_mem['Time'], microscope_mem['Memory'], label='Microscope', marker='o')
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (%)')
        plt.title('Memory Usage (%) over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()

    
def plot_timestamps(acquire_frame, acquire_pstim, analysis_frame, process_frame, visual_frame):

    plt.figure(figsize=(25, 10))
    plt.plot(range(len(acquire_frame['delta_time'])), acquire_frame['delta_time'], label='Acquire (frames)')
    plt.scatter(acquire_pstim['frame'].iloc[:-1], acquire_pstim['delta_time'].iloc[:-1], color='purple', s=15)
    plt.plot(acquire_pstim['frame'].iloc[:-1], acquire_pstim['delta_time'].iloc[:-1], linestyle='--', color='purple',label='Acquire (pstim)')
    plt.plot(range(len(analysis_frame)), analysis_frame, label = 'Analysis')
    plt.plot(range(len(process_frame)), process_frame, label='Processor')
    plt.plot(range(len(visual_frame)), visual_frame, label='Visual')
    plt.title(u'${\Delta}$T')
    plt.xlabel('Frames')
    plt.ylabel(u'${\Delta}$T (s)')
    plt.legend()
    plt.show()


if len(sys.argv) > 1:
    dataset = sys.argv[1]
else:
    dataset = input('Enter dataset: ')

path = './improv/demos/live/output_' + dataset 

process_frame = np.loadtxt(path + '/timing/process_frame_time.txt')
analysis_frame = np.loadtxt(path + '/timing/analysis_frame_time.txt')
visual_frame = np.loadtxt(path + '/timing/visual_frame_time.txt')
if len(visual_frame) !=0:
    visual_frame = visual_frame[:,1]

acquire_frame = actor_timestamps(path+'/timing/acquire_frame_timestamp.txt')
acquire_pstim = actor_timestamps(path+'/timing/acquire_pstim_timestamp.txt')

plot_timestamps(acquire_frame, acquire_pstim, analysis_frame, process_frame, visual_frame)

mem_from_log(path+'/global.log')



