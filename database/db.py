import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()  # take environment variables from .env

mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    print("WARNING: MONGO_URI environment variable is missing!", file=sys.stderr)

client = MongoClient(mongo_uri) if mongo_uri else None
db = client["face_recognition_db"] if client else None
users_collection = db["users"] if db is not None else None
