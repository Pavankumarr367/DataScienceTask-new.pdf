
from fastapi import FastAPI, UploadFile, File
from deepface import DeepFace
import pickle
import tempfile

app = FastAPI()

# Load model config
with open("models/face_model.pkl", "rb") as f:
    model_data = pickle.load(f)

threshold = model_data["threshold"]

def save_temp_image(upload_file):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")

    temp.write(upload_file.file.read())

    temp.close()

    return temp.name

@app.post("/verify")
async def verify(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):

    img1_path = save_temp_image(image1)
    img2_path = save_temp_image(image2)

    try:

        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            enforce_detection=True
        )

        face1 = result["facial_areas"]["img1"]
        face2 = result["facial_areas"]["img2"]

        verification_result = (
            "same person"
            if result["verified"]
            else "different person"
        )

        similarity_score = 1 - float(result["distance"])

        return {
            "verification_result": verification_result,
            "similarity_score": similarity_score,
            "bounding_box_image1": face1,
            "bounding_box_image2": face2
        }

    except Exception as e:
        return {
            "error": str(e)
        }
