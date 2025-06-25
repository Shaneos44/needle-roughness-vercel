# api/predict.py - Vercel-compatible FastAPI handler
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import joblib
from skimage.filters import gabor
from scipy.fftpack import fft2, fftshift
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import rank
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from io import BytesIO
from PIL import Image
import os

app = FastAPI()

# Load model from /model path
model_path = os.path.join(os.path.dirname(__file__), "../model/roughness_rf_model.pkl")
model = joblib.load(model_path)

def extract_features(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    img_np = np.array(img)

    h = img_np.shape[0]
    crop_top = int(0.2 * h)
    img_cropped = img_np[crop_top:, :]

    gray = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    needle_mask = cv2.bitwise_not(mask)
    needle_gray = cv2.bitwise_and(gray, gray, mask=needle_mask)

    roi = cv2.resize(needle_gray, (128, 128))
    roi_ubyte = img_as_ubyte(roi / 255.0)
    entropy_img = rank.entropy(roi_ubyte, disk(5))
    entropy_mean = np.mean(entropy_img)
    local_std = np.std(roi)

    glcm = graycomatrix(roi_ubyte, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    gabor_response, _ = gabor(roi, frequency=0.6)
    gabor_mean = np.mean(gabor_response)
    gabor_std = np.std(gabor_response)

    f_transform = np.abs(fftshift(fft2(roi)))
    fft_energy = np.sum(f_transform ** 2)

    return [[local_std, entropy_mean, contrast, homogeneity, gabor_mean, gabor_std, fft_energy]]

@app.post("/predict")
async def predict_roughness(file: UploadFile = File(...)):
    image_bytes = await file.read()
    features = extract_features(image_bytes)
    prediction = model.predict(features)[0]
    return JSONResponse(content={"roughness_score": round(float(prediction), 2)})
