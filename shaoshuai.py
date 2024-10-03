import cv2
import numpy as np
import os
from tqdm import tqdm  # 用于显示进度条

def load_images(image_folder, block_size):
    images = []
    for filename in os.listdir(image_folder):
        img = cv2.imread(os.path.join(image_folder, filename))
        if img is not None:
            img_resized = cv2.resize(img, block_size)
            images.append(img_resized)
    return images

def average_color(image):
    return np.mean(image, axis=(0, 1))  # 计算图像的平均颜色

def find_best_match(block, images):
    block_avg_color = average_color(block)
    best_match = None
    min_distance = float('inf')
    
    for img in images:
        img_avg_color = average_color(img)
        distance = np.linalg.norm(block_avg_color - img_avg_color)
        
        if distance < min_distance:
            min_distance = distance
            best_match = img
    
    return best_match

def process_video_frame(frame, images, block_size):
    (h, w, _) = frame.shape
    output_frame = np.zeros_like(frame)
    
    for y in range(0, h, block_size[1]):
        for x in range(0, w, block_size[0]):
            block = frame[y:y+block_size[1], x:x+block_size[0]]
            if block.shape[:2] != block_size:
                continue  # 跳过不完整的块
            
            best_match = find_best_match(block, images)
            output_frame[y:y+block_size[1], x:x+block_size[0]] = best_match
    
    return output_frame

def process_video(video_file, image_folder, output_file, block_size=(25, 25)):  # 修改了块大小
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    images = load_images(image_folder, block_size)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    for i in tqdm(range(frame_count), desc="Processing frames"):  # 加入进度条
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_video_frame(frame, images, block_size)
        out.write(processed_frame)
    
    cap.release()
    out.release()

if __name__ == "__main__":
    video_file = 'originalVideo.mp4'  # 输入的视频
    image_folder = 'shuaishuai_images'  # 随机图片的文件夹
    output_file = 'output_video(25).mp4'  # 输出的视频
    block_size = (25, 25)  # 每个像素块的大小, 你可以根据需要调整大小
    
    process_video(video_file, image_folder, output_file, block_size)
