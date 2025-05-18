import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

TRAIN_DIR = "fer2013/training"
TEST_DIR = "fer2013/test"
MODEL_PATH = "models/emotion_model.h5"

img_size = (48, 48)       # حجم الصورة المطلوب
batch_size = 64           # عدد الصور في كل دفعة (Batch)
epochs = 100              # عدد التكرارات للتدريب (Epochs)

train_datagen = ImageDataGenerator(
    rescale=1./255,                  # تحويل القيم من 0-255 إلى 0-1
    rotation_range=20,              # تدوير عشوائي للصورة
    zoom_range=0.2,                 # تكبير عشوائي
    width_shift_range=0.2,          # إزاحة أفقية
    height_shift_range=0.2,         # إزاحة رأسية
    shear_range=0.2,                # تشويه بسيط
    horizontal_flip=True            # قلب أفقي
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=img_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=1,
    shuffle=False
)

# =======================
# تحميل أو تدريب النموذج
# =======================
if os.path.exists(MODEL_PATH):
    print(f" Found existing model at {MODEL_PATH}")
    model = load_model(MODEL_PATH)

else:
    print("🚧 Model not found, training a new one...")

    #  بناء نموذج CNN مع BatchNormalization و Dropout
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dropout(0.5),                   # تقليل الـ Overfitting
        Dense(128, activation='relu'),
        Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(train_data, epochs=epochs, validation_data=test_data, callbacks=[early_stop])

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")

# =========================
#  تقييم النموذج النهائي
# =========================
print("\n🔍 Generating classification report...")

pred_probs = model.predict(test_data)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_data.classes

labels = list(test_data.class_indices.keys())

report = classification_report(y_true, y_pred, target_names=labels, digits=4)
print(report)