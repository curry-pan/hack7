from flask import Flask, request, render_template, send_file, redirect, url_for
import cv2
import numpy as np
from PIL import Image
import os
import shutil
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization
import subprocess

 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
 
# CycleGANモデルの読み込み
model = tf.keras.models.load_model(".\\saved_modelsDrop\\cyclegan_model", custom_objects={'InstanceNormalization': LayerNormalization})
 
def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.image.random_crop(image, size=[256, 256, 3])
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return tf.expand_dims(image, 0)  # バッチ次元を追加
 
def generate_image(model, input_image):
    prediction = model(input_image)
    return (prediction[0] * 0.5 + 0.5).numpy()
 
@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/submit_form', methods=['POST'])
def submit_form():
    if 'file' not in request.files:
        return "ファイルがアップロードされていません", 400
 
    target_dir = app.config['UPLOAD_FOLDER']
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)
   
    processed_dir = app.config['PROCESSED_FOLDER']
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.mkdir(processed_dir)
 
    file = request.files['file']
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)


        # Change directory to the pytorch-CycleGAN-and-pix2pix directory
        os.chdir('C:\\hakk\\hakk\\pytorch-CycleGAN-and-pix2pix')

        # Install dominate
        subprocess.run(['pip', 'install', 'dominate'], check=True)

        # Run the test.py script with different models
        subprocess.run(['python', 'test.py', '--dataroot', 'C:\\hakk\\hakk\\datasets\\testA', '--name', 'style_ukiyoe_pretrained', '--model', 'test', '--no_dropout', '--gpu_ids', '-1'], check=True)
        subprocess.run(['python', 'test.py', '--dataroot', 'C:\\hakk\\hakk\\datasets\\testA', '--name', 'style_monet_pretrained', '--model', 'test', '--no_dropout', '--gpu_ids', '-1'], check=True)
        subprocess.run(['python', 'test.py', '--dataroot', 'C:\\hakk\\hakk\\datasets\\testA', '--name', 'style_cezanne_pretrained', '--model', 'test', '--no_dropout', '--gpu_ids', '-1'], check=True)
        subprocess.run(['python', 'test.py', '--dataroot', 'C:\\hakk\\hakk\\datasets\\testA', '--name', 'style_vangogh_pretrained', '--model', 'test', '--no_dropout', '--gpu_ids', '-1'], check=True)

        #画像の処理
        input_image = preprocess_image(file_path)
        processed_image_np = generate_image(model, input_image)
 
        processed_image_pil = Image.fromarray((processed_image_np * 255).astype(np.uint8))
        processed_image_path = os.path.join(processed_dir, filename)
        processed_image_pil.save(processed_image_path)
 
        return render_template('index.html', uploaded_img=file_path, processed_img=processed_image_path)
 
@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)
 
if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['PROCESSED_FOLDER']):
        os.makedirs(app.config['PROCESSED_FOLDER'])
   
    app.run(debug=True)
 