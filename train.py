
import pickle

threshold = 0.6

model_data = {
    "threshold": threshold,
    "model_name": "DeepFace"
}

with open("models/face_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("Model saved successfully!")
