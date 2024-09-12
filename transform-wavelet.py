import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pywt  # Importing PyWavelets for the Continuous Wavelet Transform

def m4a_to_wavelet_png(m4a_file, png_file, size=(128, 128), wavelet_type='morl'):
    """
    Takes a .m4a file, applies the Continuous Wavelet Transform (CWT), and saves the scalogram as a .png file.

    Args:
        m4a_file (str): Path to the .m4a file.
        png_file (str): Path to save the .png file.
        size (tuple): Size of the output .png file (width, height).
        wavelet_type (str): Type of wavelet to use (e.g., 'morl', 'cmor', etc.).
    """

    # 1. Load the audio file (supports .m4a if FFmpeg is installed)
    y, sr = librosa.load(m4a_file)

    # 2. Perform the Continuous Wavelet Transform (CWT)
    scales = np.arange(1, 128)  # Define scales for the wavelet transform
    coefficients, frequencies = pywt.cwt(y, scales, wavelet_type, sampling_period=1/sr)

    # 3. Convert the coefficients to power (similar to amplitude for better visualization)
    coefficients = np.abs(coefficients)

    # 4. Create a figure and axes for plotting
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=1300)  # Adjust figsize for desired output size

    # 5. Display the scalogram (CWT coefficients) as an image
    img = ax.imshow(coefficients, extent=[0, len(y)/sr, 1, 128], cmap='jet', aspect='auto',
                    vmax=np.max(coefficients), vmin=np.min(coefficients))

    # 6. Remove axes and labels for a cleaner look
    ax.axis('off')

    # 7. Save the figure as a .png file
    fig.savefig(png_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def process_m4a_files_in_folder(input_folder, output_folder, size=(128, 128), wavelet_type='morl'):
    """
    Processes all .m4a files in a folder and saves their scalograms as .png files.

    Args:
        input_folder (str): Path to the folder containing .m4a files.
        output_folder (str): Path to save the .png files.
        size (tuple): Size of the output .png files (width, height).
        wavelet_type (str): Type of wavelet to use for the CWT (e.g., 'morl', 'cmor', etc.).
    """

    # 1. Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # 2. Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".m4a"):
            input_file_path = os.path.join(input_folder, filename)
            output_file_name = os.path.splitext(filename)[0] + ".png"  # Replace .m4a with .png
            output_file_path = os.path.join(output_folder, output_file_name)

            # 3. Process the .m4a file and save the scalogram
            m4a_to_wavelet_png(input_file_path, output_file_path, size, wavelet_type)
            print(f"Processed: {input_file_path} -> {output_file_path}")

# Usage
input_folder = "data-all"
output_folder = "wavelet-scalograms"
process_m4a_files_in_folder(input_folder, output_folder)
