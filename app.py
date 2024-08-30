from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import cv2
import numpy as np
import requests
import imghdr
from tensorflow.keras.models import load_model


def extract_signature_with_roboflow(source_image_path, save_path):
    """Roboflow API kullanarak imza tespiti ve çıkarma işlemi."""
    image = open(source_image_path, "rb").read()
    response = requests.post(
        MODEL_ENDPOINT,
        params={
            "api_key": API_KEY,
            "confidence": 40,
            "overlap": 30
        },
        files={
            "file": image
        }
    )
    
    detections = response.json().get('predictions', [])

    # Resmi yükleyin
    img = cv2.imread(source_image_path)
    if detections:
        # İlk tespiti kullanarak imza bölgesini kes
        detection = detections[0]
        x1 = max(int(detection['x'] - detection['width'] / 2), 0)
        y1 = max(int(detection['y'] - detection['height'] / 2), 0)
        x2 = min(int(detection['x'] + detection['width'] / 2), img.shape[1])
        y2 = min(int(detection['y'] + detection['height'] / 2), img.shape[0])

        signature = img[y1:y2, x1:x2]
        cv2.imwrite(save_path, signature)
        return save_path
    else:
        flash('İmza tespit edilemedi.', 'error')
        return None


def preprocess_signature(image_path):
    """Resize and normalize the signature image for classification."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, target_size)
        image = image / 255.0
        return image
    return None


def classify_signature(signature_image):
    """Classify the signature using the signature model and return the percentage."""
    signature_image = np.expand_dims(signature_image, axis=0)  # Add batch dimension
    prediction = signature_model.predict(signature_image)[0][0]
    percentage_prediction = prediction * 100  # Persentaj cinsinden döndür
    return percentage_prediction


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('verify_document'))
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/verify_document', methods=['GET', 'POST'])
def verify_document():
    uploaded_image = None
    extracted_image = None
    prediction = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Dosya yüklenmedi', 'error')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('Dosya seçilmedi', 'error')
            return redirect(request.url)

        if file and imghdr.what(file.stream) not in ['jpeg', 'png', 'gif', 'bmp']:
            flash('Geçersiz dosya formatı. Lütfen bir görüntü dosyası yükleyin.', 'error')
            return redirect(request.url)

        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            extracted_signature_path = os.path.join(app.config['OUTPUT_FOLDER'], 'extracted_signature.png')
            extracted_signature = extract_signature_with_roboflow(filepath, extracted_signature_path)

            if extracted_signature:
                extracted_image = url_for('output_file', filename='extracted_signature.png')
                uploaded_image = url_for('uploaded_file', filename=filename)

                processed_signature = preprocess_signature(extracted_signature_path)
                if processed_signature is not None:
                    prediction = classify_signature(processed_signature)
                else:
                    flash('İmza işlenirken hata oluştu.', 'error')
            else:
                flash('İmza tespit edilemedi.', 'error')

    return render_template('verify_document.html', uploaded_image=uploaded_image, extracted_image=extracted_image, prediction=prediction)


@app.route('/payment', methods=['GET', 'POST'])
def payment():
    if request.method == 'POST':
        flash('Ödeme başarıyla işlendi!', 'success')
        return redirect(url_for('verify_document'))
    return render_template('payment.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
