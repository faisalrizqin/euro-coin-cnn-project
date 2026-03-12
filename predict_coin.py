import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model/coin_model.h5")

print("Model berhasil dimuat")

class_labels = ['10c', '1c', '1e', '20c', '2c', '2e', '50c', '5c']

coin_values = {
    '1c': 0.01,
    '2c': 0.02,
    '5c': 0.05,
    '10c': 0.10,
    '20c': 0.20,
    '50c': 0.50,
    '1e': 1.00,
    '2e': 2.00
}

image = cv2.imread("test_images/coins.jpg")
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (9, 9), 2)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp = 1.2,
    minDist = 120,
    param1 = 100,
    param2 = 40,
    minRadius = 40,
    maxRadius = 120
)

total_value = 0

if circles is not None:
    circles = np.round(circles[0, :].astype("int"))

    detected_coins = []

    for (x, y, r) in circles:

        padding = 10

        y1 = max(0, y - r - padding)
        y2 = min(image.shape[0], y + r + padding)
        x1 = max(0, x - r - padding)
        x2 = min(image.shape[1], x + r + padding)
        
        coin = image[y1 : y2, x1 : x2]

        if coin.size == 0:
            continue

        coin = cv2.cvtColor(coin, cv2.COLOR_BGR2RGB)
        coin = cv2.resize(coin, (128, 128))
        coin = coin / 255.0
        coin = np.expand_dims(coin, axis = 0)

        prediction = model.predict(coin)
        class_id = np.argmax(prediction)

        confidence = np.max(prediction)

        label = class_labels[class_id]
        detected_coins.append((label, confidence))

        value = coin_values[label]
        total_value += value

        cv2.circle(output, (x, y), r, (0, 255, 0), 2)

        cv2.putText(
            output,
            f"{label} {confidence:.2f}",
            (x-20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

print("Detected coins:")

for coin, conf in detected_coins:
    print(coin, "confidence: ", round(conf, 2))

print("Total Euro: ", total_value)

cv2.putText(
    output,
    f"Total: {total_value:.2f} Euro",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)