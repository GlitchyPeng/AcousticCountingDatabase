import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pydub import AudioSegment
import glob
import os

# 存储每个类别的最大包络面积
max_envelope_areas = []
categories_with_data = []

# 遍历 i 的范围，从 1 到 10
for i in range(1, 11):
    waveform_list = []
    fft_list = []
    
    # 遍历 j 的范围，从 1 到 20
    for j in range(1, 21):
        # 查找同一类文件
        file_pattern = f'data-all/{i}-{j}.m4a'
        files = glob.glob(file_pattern)

        # 确保文件存在
        if len(files) == 0:
            print(f"文件 {i}-{j}.m4a 未找到，检查文件路径。")
            continue

        # 读取每个文件
        for file in files:
            # 将 m4a 文件转换为 wav 文件
            wav_filename = file.replace('.m4a', '.wav')
            if not os.path.exists(wav_filename):  # 仅在没有 wav 文件时进行转换
                audio = AudioSegment.from_file(file, format='m4a')
                audio.export(wav_filename, format='wav')

            # 读取wav文件
            sample_rate, data = wavfile.read(wav_filename)

            # 如果音频是立体声(双通道)，将其转换为单声道
            if len(data.shape) > 1:
                data = data.mean(axis=1)

            # 保存波形数据
            waveform_list.append(data)

            # 计算傅里叶变换并保存
            fft_result = np.fft.fft(data)
            fft_list.append(np.abs(fft_result))

    # 确保所有 20 个文件被读取
    if len(waveform_list) > 0 and len(fft_list) > 0:
        # 对每个类别的 20 个文件的波形和频谱进行平均
        min_length = min(len(w) for w in waveform_list)  # 找到最短的波形长度，以便对齐数据
        avg_waveform = np.mean([w[:min_length] for w in waveform_list], axis=0)
        avg_fft = np.mean([f[:min_length] for f in fft_list], axis=0)
        fft_freq = np.fft.fftfreq(min_length, 1 / sample_rate)

        # 计算最大包络面积
        envelope_area = np.trapz(avg_fft[:min_length // 2], fft_freq[:min_length // 2])
        max_envelope_areas.append(envelope_area)
        categories_with_data.append(i)
    else:
        print(f"类别 {i} 的文件数量不足 20 个，实际读取了 {len(waveform_list)} 个文件。请检查文件路径和文件存在情况。")

# 计算误差棒（标准误差）
if max_envelope_areas:
    mean_area = np.mean(max_envelope_areas)
    std_area = np.std(max_envelope_areas)

    # 绘制最大包络面积和 i 的关系
    plt.figure(figsize=(10, 6))
    plt.errorbar(categories_with_data, max_envelope_areas, yerr=std_area, fmt='o', capsize=5, capthick=2, label='Max Envelope Area', markersize=8)
    plt.title('Maximum Envelope Area vs. Category', fontsize=24)
    plt.xlabel('Category', fontsize=24)
    plt.ylabel('Maximum Envelope Area', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=20)
    plt.show()
else:
    print("没有足够的数据来绘制图表。请检查文件路径和数据是否存在。")
