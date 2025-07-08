
from io import BytesIO
import face_recognition
import numpy as np
import pickle
import faiss 
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pymongo
import time
from PIL import Image
import requests 
import schedule 
import time
import schedule



client = pymongo.MongoClient("mongodb+srv://abdoadimy:Bx47VYqDJXpc6XZ2@cluster0.me26o.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")


databases = client.list_database_names()
db_name = databases[0]  # Select the first database
db = client[db_name]

collections = db.list_collection_names()
# print(f"Collections in '{db_name}':", collections)

collection_name = collections[1]
collection = db[collection_name]



def encode_missing_faces():
    print("🔁 Checking for new unencoded faces...")
    people = collection.find({"encoding2": {"$exists": False}})

    count = 0
    for person in people:
        name = person.get("name")
        photo_url = person.get("photo")

        if not photo_url:
            print(f"⚠️ Skipping {name}: No photo URL")
            continue

        try:
            # تحميل الصورة من URL
            response = requests.get(photo_url)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content)).convert("RGB")
            image_np = np.array(image)

            face_locations = face_recognition.face_locations(image_np)
            encodings = face_recognition.face_encodings(image_np, face_locations)

            if encodings:
                encoding = encodings[0].tolist()
                collection.update_one(
                    {"_id": person["_id"]},
                    {"$set": {"encoding2": encoding}}
                )
                print(f"✅ Encoded & saved: {name}")
                count += 1
            else:
                print(f"❌ No face found in photo for {name}")

        except Exception as e:
            print(f"❌ Error with {name}: {e}")

    if count == 0:
        print("✅ No new faces found.\n")

# جدولة الوظيفة لتعمل كل دقيقة
schedule.every(0.5).minutes.do(encode_missing_faces)

print("⏱️ Face encoding service started. Checking every 30 second...\n")

# حلقة التنفيذ المستمرة
while True:
    schedule.run_pending()
    time.sleep(1)
