from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io


app = FastAPI(title="MNIST Two-Digit Sum API")


model = None
try:
    model = tf.keras.models.load_model('number_recognition_model.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise Exception("Failed to load model")


def preprocess_image(image):
    image = image.astype('float32') / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

# Function to predict digit
def predict_digit(image):
    image = preprocess_image(image)
    prediction = model.predict(image, verbose=0)
    return np.argmax(prediction[0])


@app.post("/predict")
async def predict_two_digits(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('L')  
        img = img.resize((56, 28))
        img_array = np.array(img)

        if img_array.max() > img_array.min():
            img_array = 255 - img_array

        left_digit = img_array[:, :28].reshape(28, 28, 1)
        right_digit = img_array[:, 28:].reshape(28, 28, 1)

        predicted_digit1 = predict_digit(left_digit)
        predicted_digit2 = predict_digit(right_digit)

        digit_sum = predicted_digit1 + predicted_digit2


        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img_array.reshape(28, 56), cmap='gray')
        plt.title("Input Image")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(left_digit.reshape(28, 28), cmap='gray')
        plt.title(f"Left: Predicted {predicted_digit1}")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(right_digit.reshape(28, 28), cmap='gray')
        plt.title(f"Right: Predicted {predicted_digit2}")
        plt.axis('off')
        plt.suptitle(f"Sum: {predicted_digit1} + {predicted_digit2} = {digit_sum}")
        plt.savefig('two_digit_sum.png')
        plt.close()


        return JSONResponse(content={
            "left_digit": int(predicted_digit1),
            "right_digit": int(predicted_digit2),
            "sum": int(digit_sum)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}