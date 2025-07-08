from io import BytesIO
import os
import face_recognition
import numpy as np
import faiss 
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pymongo
from dotenv import load_dotenv
# from test_automated import encode_missing_faces

load_dotenv()


app = FastAPI()


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



MONGO_URL = os.getenv("MONGO_URL")  # This reads from .env

client = pymongo.MongoClient(MONGO_URL)  # Replace with your MongoDB connection string


databases = client.list_database_names()
db_name = databases[0]  # Select the first database
db = client[db_name]

collections = db.list_collection_names()
# print(f"Collections in '{db_name}':", collections)

collection_name = collections[1]
collection = db[collection_name]


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    # 1. تحميل الصورة
    image = face_recognition.load_image_file(file.file)
    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        return JSONResponse(content={
            "found_faces": 0,
            "message": "No faces detected"
        }, status_code=404)

    # 2. استخراج encoding من الصورة الجديدة
    encoding = face_recognition.face_encodings(image, face_locations)[0]

    # 3. تحميل الـ encodings المسجلة في MongoDB
    people = list(collection.find({"encoding2": {"$exists": True}}))

    if not people:
        return JSONResponse(content={"message": "No encoded faces in DB"}, status_code=500)

    known_names = [p["name"] for p in people]
    known_encodings = [p["encoding2"] for p in people]

    # 4. بناء FAISS index في الوقت الفعلي
    index = faiss.IndexFlatL2(128)
    index.add(np.array(known_encodings, dtype='float32'))

    # 5. مطابقة الوجه المرفوع مع البيانات
    D, I = index.search(np.array([encoding], dtype='float32'), 1)
    closest_match_index = I[0][0]
    distance = D[0][0]

    if distance < 0.35:
        matched_name = known_names[closest_match_index]
        return JSONResponse(content={
            "found_faces": 1,
            "matched_name": matched_name,
            "distance": float(distance)
        })

    else:
        return JSONResponse(content={
            "found_faces": 0,
            "matched_name": "No Face Matched"
        })



# ------