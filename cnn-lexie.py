import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random

# 定义图片文件夹路径
image_folder = 'mfcc-diagram'
image_size = (128, 128)

# 加载图片和标签
images = []
labels = []

for filename in os.listdir(image_folder):
    if filename.endswith('.png'):
        # 提取文件名中的标签，假设标签是以文件名的第一个数字表示
        label = int(filename.split('-')[0])
        labels.append(label)
        
        # 加载图片并调整大小
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).resize(image_size).convert('RGB')
        img_array = np.array(img) / 255.0  # 归一化处理
        images.append(img_array)

images = np.array(images)
labels = np.array(labels)

# 输出总共有多少组标签
num_classes = len(set(labels))
print(f"Total number of classes: {num_classes}")

# 切分训练集和测试集
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=40,  # 增大旋转范围
    width_shift_range=0.4,  # 增大水平位移范围
    height_shift_range=0.4,  # 增大垂直位移范围
    shear_range=0.4,  # 增大剪切范围
    zoom_range=0.4,  # 增大缩放范围
    horizontal_flip=True,
    fill_mode='nearest'
)

# 使用ResNet50预训练模型并解冻部分层
base_model = tf.keras.applications.ResNet50(input_shape=(128, 128, 3),
                                            include_top=False,
                                            weights='imagenet')

base_model.trainable = True
for layer in base_model.layers[:-10]:  # 解冻ResNet50的最后10层
    layer.trainable = False

# 构建模型，减少池化层并调整卷积核大小
model = models.Sequential([
    base_model,
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),  # 增加padding保持尺寸
    layers.MaxPooling2D(pool_size=(2, 2)),  # 保留原来的池化层
    layers.Conv2D(256, (2, 2), activation='relu', padding='same'),  # 使用 'same' padding 避免尺寸问题
    layers.GlobalAveragePooling2D(),  # 使用全局池化代替最后的MaxPooling和Flatten
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  # 增加一个更大的全连接层
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')  # 使用上面计算的类数
])

# 定义余弦退火学习率调度器
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=5e-3,  # 提高初始学习率
    first_decay_steps=1000, 
    t_mul=2.0, 
    m_mul=0.9, 
    alpha=1e-6
)

# 编译模型，使用学习率调度器和标签平滑
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              metrics=['accuracy'])

# 训练模型
history = model.fit(datagen.flow(train_images, train_labels, batch_size=256),  # 调整 batch_size 至 256
                    validation_data=(test_images, test_labels),
                    epochs=300)  # 增加训练轮次到 300

# 绘制准确率
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# 测试集评估
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")

# 预测单张图片并进行比较
def load_and_preprocess_image(image_path):
    """加载并预处理单张图片."""
    img = Image.open(image_path).resize(image_size).convert('RGB')
    img_array = np.array(img) / 255.0  # 归一化
    img_array = np.expand_dims(img_array, axis=0)  # 增加一个维度用于批处理
    return img_array

def predict_and_evaluate(image_path, true_label):
    """预测图片的类别并计算 accuracy."""
    img_array = load_and_preprocess_image(image_path)
    prediction = model.predict(img_array)
    
    predicted_label = np.argmax(prediction, axis=1)[0]  # 获取预测的类别
    
    # 输出预测的结果和真实的标签
    print(f"Predicted Label: {predicted_label}, True Label: {true_label}")
    
    # 使用公式 |m - n| / n
    accuracy = abs(predicted_label - true_label) / true_label
    
    print(f"Accuracy for this image: {accuracy}")
    
    return accuracy

def evaluate_random_images(num_images):
    """随机选择图片并执行评估."""
    random_indices = random.sample(range(len(test_images)), num_images)
    for idx in random_indices:
        # 找到与此索引匹配的文件名
        filename = os.listdir(image_folder)[idx]
        image_path = os.path.join(image_folder, filename)  # 使用实际文件名拼接路径
        true_label = test_labels[idx]
        predict_and_evaluate(image_path, true_label)

# 执行评估
evaluate_random_images(10)  # 测试10张图片
