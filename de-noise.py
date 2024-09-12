import os
from pydub import AudioSegment
from pydub.utils import which
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import stft, istft

# 设置 ffmpeg 和 ffprobe 路径
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

def compute_noise_spectrum(noise_samples, sample_rate):
    # 计算噪声的STFT
    _, _, Zxx = stft(noise_samples, fs=sample_rate)
    # 计算噪声幅度谱的均值
    noise_spectrum = np.mean(np.abs(Zxx), axis=1, keepdims=True)
    return noise_spectrum

def compute_average_noise_spectrum(noise_folder):
    noise_spectrums = []
    
    for filename in os.listdir(noise_folder):
        if filename.startswith("0-") and filename.endswith(".m4a"):
            noise_path = os.path.join(noise_folder, filename)
            print(f"处理噪声文件: {noise_path}")
            
            # 读取噪声文件
            noise_sound = AudioSegment.from_file(noise_path, format="m4a")
            noise_samples = np.array(noise_sound.get_array_of_samples())
            sample_rate = noise_sound.frame_rate
            
            # 计算并存储每个噪声文件的频谱
            noise_spectrum = compute_noise_spectrum(noise_samples, sample_rate)
            noise_spectrums.append(noise_spectrum)

    # 计算所有噪声频谱的平均值
    average_noise_spectrum = np.mean(noise_spectrums, axis=0)
    return average_noise_spectrum

def spectral_subtraction_with_noise(audio, sample_rate, noise_spectrum, noise_reduction=0.5):
    # 将音频数据转换为时域信号
    _, _, Zxx = stft(audio, fs=sample_rate)
    # 计算音频的幅度谱
    magnitude = np.abs(Zxx)
    # 用给定的噪声谱减去
    magnitude_clean = np.maximum(magnitude - noise_reduction * noise_spectrum, 0)
    # 重建信号
    _, audio_clean = istft(magnitude_clean * np.exp(1j * np.angle(Zxx)), fs=sample_rate)
    return audio_clean.astype(np.int16)

def process_m4a_files_with_noise_spectrum(input_folder, average_noise_spectrum):
    for filename in os.listdir(input_folder):
        if not filename.startswith("0-") and filename.endswith(".m4a"):
            audio_path = os.path.join(input_folder, filename)
            print(f"正在处理: {audio_path}")
            
            # 读取音频文件并进行谱减法降噪
            sound = AudioSegment.from_file(audio_path, format="m4a")
            samples = np.array(sound.get_array_of_samples())
            sample_rate = sound.frame_rate

            # 使用平均噪声频谱进行降噪
            clean_samples = spectral_subtraction_with_noise(samples, sample_rate, average_noise_spectrum)
            
            # 保存降噪后的音频
            clean_audio_path = os.path.splitext(audio_path)[0] + "_clean.wav"
            write(clean_audio_path, sample_rate, clean_samples)

            print(f"处理完成: {clean_audio_path}")

# 调用示例
noise_folder = "data-all"  # 替换为包含噪声文件的文件夹路径
input_folder = "data-all-clean"  # 替换为需要降噪的音频文件夹路径

# 计算平均噪声频谱
average_noise_spectrum = compute_average_noise_spectrum(noise_folder)

# 使用平均噪声频谱处理其他音频文件
process_m4a_files_with_noise_spectrum(input_folder, average_noise_spectrum)
