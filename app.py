from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage import measure, morphology
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import imghdr

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads/'
app.config['OUTPUT_FOLDER'] = './outputs/'
app.secret_key = 'supersecretkey'

# Sabitler
constant_parameter_1 = 84
constant_parameter_2 = 250
constant_parameter_3 = 100
constant_parameter_4 = 18
target_size = (224, 224)

# Önceden eğitilmiş modelin yüklenmesi
model = load_model('resnet50_signature_model.h5')

def extract_signature(source_image_path, save_path):
    """Extract signature from an input image."""
    img = cv2.imread(source_image_path, 0)  # Load image in grayscale
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # Ensure binary

    # Connected component analysis
    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)

    the_biggest_component = 0
    total_area = 0
    counter = 0
    average = 0.0

    for region in regionprops(blobs_labels):
        if region.area > 10:
            total_area += region.area
            counter += 1
        if region.area >= 250:
            if region.area > the_biggest_component:
                the_biggest_component = region.area

    average = total_area / counter if counter > 0 else 0

    # Experimental-based ratio calculation
    a4_small_size_outliar_constant = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3
    a4_big_size_outliar_constant = a4_small_size_outliar_constant * constant_parameter_4

    # Remove the connected pixels that are smaller than a4_small_size_outliar_constant
    pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)
    
    # Remove the connected pixels that are bigger than a4_big_size_outliar_constant
    component_sizes = np.bincount(pre_version.ravel())
    too_small = component_sizes > a4_big_size_outliar_constant
    too_small_mask = too_small[pre_version]
    pre_version[too_small_mask] = 0

    # Save the pre-version image
    plt.imsave('pre_version.png', pre_version)

    # Read the pre-version image
    img = cv2.imread('pre_version.png', 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Save the result
    cv2.imwrite(save_path, img)
    return save_path

def preprocess_signature(image_path):
    """Resize and normalize the signature image for classification."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is not None:
        image = cv2.resize(image, target_size)
        image = image / 255.0
        image = np.stack((image,) * 3, axis=-1)  # Convert to 3-channel image
    return image

def classify_signature(signature_image):
    """Classify the signature using the model and return the percentage."""
    signature_image = np.expand_dims(signature_image, axis=0)
    predictions = []

    # Original image prediction
    original_prediction = model.predict(signature_image)[0][0]
    predictions.append(original_prediction)

    # Rotated image predictions
    for k in range(1, 4):
        rotated_image = np.rot90(signature_image, k=k, axes=(1, 2))
        rotated_prediction = model.predict(rotated_image)[0][0]
        predictions.append(rotated_prediction)

    # Scaled image prediction
    scaled_image = cv2.resize(signature_image[0], (112, 112))
    scaled_image = np.expand_dims(scaled_image, axis=0)
    scaled_prediction = model.predict(scaled_image)[0][0]
    predictions.append(scaled_prediction)

    # Average prediction
    average_prediction = np.mean(predictions)
    percentage_prediction = average_prediction * 100  # Convert to percentage
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
            extracted_signature = extract_signature(filepath, extracted_signature_path)

            if extracted_signature:
                extracted_image = url_for('output_file', filename='extracted_signature.png')
                uploaded_image = url_for('uploaded_file', filename=filename)

                processed_signature = preprocess_signature(extracted_signature_path)
                prediction = classify_signature(processed_signature)
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
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['OUTPUT_FOLDER']):
        os.makedirs(app.config['OUTPUT_FOLDER'])

    app.run(debug=True)
