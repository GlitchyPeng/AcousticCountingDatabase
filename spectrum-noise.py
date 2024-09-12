import numpy as np
import librosa
import os
import matplotlib.pyplot as plt

# 文件路径模板，使用索引 1 到 20
noisy_audio_template = 'data-all/0-{}.m4a'  # 带噪音的目标音频文件路径
indices = range(1, 21)  # 索引从 1 到 20
noise_folder = 'data-all'  # 包含 20 个噪音文件的文件夹

# 初始化噪声谱
noise_spectrum_sum = None

# 设置帧大小和步长
frame_size = 1024
hop_length = 512

# Find the minimum number of frames among all noise files for padding/truncating
min_frames = np.inf

# 遍历每个索引文件并累加噪声频谱
for index in indices:
    noisy_audio_path = noisy_audio_template.format(index)  # 根据索引生成路径
    
    if not os.path.exists(noisy_audio_path):
        print(f"文件不存在: {noisy_audio_path}")
        continue
    
    # 加载噪音文件
    noisy_signal, sr = librosa.load(noisy_audio_path, sr=None)
    
    # 计算噪声的 STFT
    noise_stft = librosa.stft(noisy_signal, n_fft=frame_size, hop_length=hop_length)
    noise_magnitude = np.abs(noise_stft)
    
    # 更新最小帧数
    min_frames = min(min_frames, noise_magnitude.shape[1])

# 计算所有噪音文件的平均频谱
for index in indices:
    noisy_audio_path = noisy_audio_template.format(index)
    
    if not os.path.exists(noisy_audio_path):
        continue
    
    noisy_signal, sr = librosa.load(noisy_audio_path, sr=None)
    
    # 计算噪声的 STFT
    noise_stft = librosa.stft(noisy_signal, n_fft=frame_size, hop_length=hop_length)
    noise_magnitude = np.abs(noise_stft)
    
    # Truncate/pad the STFT to match the minimum number of frames
    noise_magnitude = noise_magnitude[:, :min_frames]
    
    # 累加噪声谱
    if noise_spectrum_sum is None:
        noise_spectrum_sum = noise_magnitude
    else:
        noise_spectrum_sum += noise_magnitude

# 计算噪声频谱的平均值
num_noise_files = len(indices)  # 噪音文件数
average_noise_spectrum = noise_spectrum_sum / num_noise_files

# 可视化平均噪声频谱
plt.figure(figsize=(10, 6))
plt.imshow(10 * np.log10(average_noise_spectrum + 1e-6), origin='lower', aspect='auto', cmap='inferno')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Time Frames',fontsize=24)
plt.ylabel('Frequency Bins',fontsize=24)
plt.show()
