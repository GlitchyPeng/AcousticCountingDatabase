import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def m4a_to_waveform_png(m4a_file, png_file, size=(128, 128)):
    """
    Takes a .m4a file, plots the waveform, and saves it as a .png file.

    Args:
        m4a_file (str): Path to the .m4a file.
        png_file (str): Path to save the .png file.
        size (tuple): Size of the output .png file (width, height).
    """

    # 1. Load the audio file
    y, sr = librosa.load(m4a_file)

    # 2. Create a figure and axes for plotting
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=1300)  # Adjust figsize for desired output size

    # 3. Plot the waveform
    librosa.display.waveshow(y, sr=sr, ax=ax)

    # 4. Remove axes and labels for a cleaner look
    ax.axis('off')

    # 5. Save the figure as a .png file
    fig.savefig(png_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def process_m4a_files_in_folder(input_folder, output_folder, size=(128, 128)):
    """
    Processes all .m4a files in a folder and saves their waveform plots as .png files.

    Args:
        input_folder (str): Path to the folder containing .m4a files.
        output_folder (str): Path to save the .png files.
        size (tuple): Size of the output .png files (width, height).
    """

    # 1. Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # 2. Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".m4a"):  # Adjust to handle .m4a files
            input_file_path = os.path.join(input_folder, filename)
            output_file_name = os.path.splitext(filename)[0] + ".png"  # Replace .m4a with .png
            output_file_path = os.path.join(output_folder, output_file_name)

            # 3. Process the .m4a file and save the waveform
            m4a_to_waveform_png(input_file_path, output_file_path, size)
            print(f"Processed: {input_file_path} -> {output_file_path}")

# Usage
input_folder = "data-all"
output_folder = "waveforms"
process_m4a_files_in_folder(input_folder, output_folder)
