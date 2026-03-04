
import numpy as np
from models import build_cnn_model, build_cnn_texture_model

# Dummy normalized grayscale image (96x96x1)
dummy_image = np.random.rand(1, 96, 96, 1).astype(np.float32)

# Dummy texture features (355,)
dummy_texture = np.random.rand(1, 355).astype(np.float32)


# ==========================================
# Example: Single-Input CNN Model 
# ==========================================

print("--- Loading CNN Model ---")
cnn_model = build_cnn_model()

# Simulating loading weights of LOOCV fold 1
cnn_weight_path = 'weights/cnn/fold_1.weights.h5' 
try:
    cnn_model.load_weights(cnn_weight_path, by_name=False, skip_mismatch=True)
    print(f"Successfully loaded weights from {cnn_weight_path}")
except FileNotFoundError:
    print(f"Weight file not found at {cnn_weight_path}. Model initialized randomly.")

# Predict
cnn_prediction = cnn_model.predict(dummy_image)
print(f"CNN Prediction Output: {cnn_prediction}\n")


# ==========================================
# Example: Dual-Input Model (CNN + Texture)
# ==========================================

print("--- Loading CNN + Texture Model ---")
cnn_texture_model = build_cnn_texture_model()

# Simulating loading weights of LOOCV fold 1
cnn_texture_weight_path = 'weights/cnn_texture/fold_1.weights.h5'
try:
    cnn_texture_model.load_weights(cnn_texture_weight_path, by_name=False, skip_mismatch=True)
    print(f"Successfully loaded weights from {cnn_texture_weight_path}")
except FileNotFoundError:
    print(f"Weight file not found at {cnn_texture_weight_path}. Model initialized randomly.")

# Predict
cnn_texture_prediction = cnn_texture_model.predict([dummy_image, dummy_texture])
print(f"CNN + Texture Model Prediction: {cnn_texture_prediction}")

