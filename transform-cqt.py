import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def audio_to_cqt_png(audio_file, png_file, size=(128, 128)):
    """
    Takes an audio file (e.g., .m4a, .wav), computes the Constant-Q Transform (CQT), and saves the spectrogram as a .png file.

    Args:
        audio_file (str): Path to the audio file.
        png_file (str): Path to save the .png file.
        size (tuple): Size of the output .png file (width, height).
    """

    # 1. Load the audio file (librosa supports multiple formats, including .m4a)
    y, sr = librosa.load(audio_file)

    # 2. Compute the Constant-Q Transform (CQT)
    cqt = np.abs(librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('C1'), n_bins=84))

    # 3. Convert the CQT to dB scale for better visualization
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)

    # 4. Create a figure and axes for plotting
    fig, ax = plt.subplots(figsize=(size[0]/100, size[1]/100), dpi=1300)  # Adjust figsize for desired output size

    # 5. Display the CQT spectrogram
    img = librosa.display.specshow(cqt_db, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax, cmap='jet')

    # 6. Remove axes and labels for a cleaner look
    ax.axis('off')

    # 7. Save the figure as a .png file
    fig.savefig(png_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def process_audio_files_in_folder(input_folder, output_folder, size=(128, 128)):
    """
    Processes all .m4a or .wav files in a folder and saves their CQT spectrograms as .png files.

    Args:
        input_folder (str): Path to the folder containing audio files (.m4a or .wav).
        output_folder (str): Path to save the .png files.
        size (tuple): Size of the output .png files (width, height).
    """

    # 1. Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # 2. Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".m4a") or filename.endswith(".wav"):
            input_file_path = os.path.join(input_folder, filename)
            output_file_name = os.path.splitext(filename)[0] + ".png"  # Replace .m4a/.wav with .png
            output_file_path = os.path.join(output_folder, output_file_name)

            # 3. Process the audio file and save the CQT spectrogram
            audio_to_cqt_png(input_file_path, output_file_path, size)
            print(f"Processed: {input_file_path} -> {output_file_path}")

# Usage
input_folder = "data-all"
output_folder = "cqt-spectrograms"
process_audio_files_in_folder(input_folder, output_folder)
