import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_benchmark_data(root_directory):
    benchmarks_path = os.path.join(root_directory, 'benchmarks')
    benchmark_files = [f for f in os.listdir(benchmarks_path) if f.endswith('.csv')]
    data = {}
    
    for file in benchmark_files:
        filepath = os.path.join(benchmarks_path, file)
        df = pd.read_csv(filepath)
        df = df.iloc[1:].reset_index(drop=True)
        df['cumulative_time'] = df['timestamp'].cumsum() / 60  # Convert to minutes
        camera_name = df['camera_name'].iloc[0] if not df.empty else file
        data[camera_name] = df
    
    return data

def plot_benchmark_data(data):
    axis_fps = [59, 61]
    axis_delay = [0, 0.03]
    sns.set(style="whitegrid")
    
    n_cameras = len(data)
    fig, axs = plt.subplots(n_cameras, 2, figsize=(15, 5 * n_cameras), squeeze=False)
    
    # Colors
    fps_color = 'blue'
    delay_color = 'red'
    
    for idx, (camera, df) in enumerate(data.items()):
        # Subplot 1: FPS over time
        axs[idx, 0].plot(df['cumulative_time'], df['fps'], color=fps_color, label='FPS')
        
        axs[idx, 0].set_ylabel('FPS')
        axs[idx, 0].set_ylim(axis_fps)
        axs[idx, 0].set_xlim([df['cumulative_time'].min(), df['cumulative_time'].max()])
        axs[idx, 0].legend()

        if idx == 0:
            axs[idx, 0].set_title(f'FPS Over Time')
        elif idx == n_cameras - 1:
            axs[idx, 0].set_xlabel('Time (minutes)')

        # Subplot 2: Avg Delay over time with min and max delay shaded
        axs[idx, 1].plot(df['cumulative_time'], df['avg_delay'], color=delay_color, label='Avg Delay')
        axs[idx, 1].fill_between(df['cumulative_time'], df['min_delay'], df['max_delay'], color=delay_color, alpha=0.3, label='Min-Max Delay')
        axs[idx, 1].set_ylabel('Delay (s)')
        axs[idx, 1].set_ylim(axis_delay)
        axs[idx, 1].set_xlim([df['cumulative_time'].min(), df['cumulative_time'].max()])
        axs[idx, 1].legend()

        if idx == 0:
            axs[idx, 1].set_title(f'Average Delay Over Time')
        elif idx == n_cameras - 1:
            axs[idx, 1].set_xlabel('Time (minutes)')
    
    plt.tight_layout()
    plt.show()
    
    # Second Plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    
    # Prepare data for violin plots
    fps_data = []
    delay_data = []
    cameras = []
    
    for camera, df in data.items():
        fps_data.extend(df['fps'])
        delay_data.extend(df['avg_delay'])
        cameras.extend([camera] * len(df))
    
    violin_df = pd.DataFrame({
        'Camera': cameras,
        'FPS': fps_data,
        'Avg Delay': delay_data
    })
    
    # Subplot 1: FPS Violin Plot
    sns.violinplot(x='Camera', y='FPS', data=violin_df, ax=axs[0], palette=['blue'])
    axs[0].set_title('FPS Distribution per Camera')
    axs[0].set_ylim(axis_fps)
    
    # Subplot 2: Avg Delay Violin Plot
    sns.violinplot(x='Camera', y='Avg Delay', data=violin_df, ax=axs[1], palette=['red'])
    axs[1].set_title('Average Delay Distribution per Camera')
    axs[1].set_ylim(axis_delay)
    
    plt.tight_layout()
    plt.show()

def main():
    root_directory = Path(__file__).resolve().parent.parent.parent
    data = load_benchmark_data(root_directory)
    
    if data:
        plot_benchmark_data(data)
    else:
        print("No benchmark files found.")

if __name__ == "__main__":
    main()