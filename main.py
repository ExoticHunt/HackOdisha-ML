from importlib.resources import path
from fastapi import FastAPI, File, UploadFile
from face_recog import result,result2
from take_face import create_faces
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Callable
import shutil
import random
import cv2
app = FastAPI()

names = {3: 'jimmy fermin', 6: 'Michael Dam', 51: 'shouvik',1: 'dashmat'} # add a name into this list

def create_img(name):
    count = 0
    face_cascade = cv2.CascadeClassifier('./yml/haarcascade_frontalface_default.xml')
    face_id = random.randint(1, 100)

    personFileName = name

    while(True):
        img = cv2.imread('./find_me/%s' %personFileName)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.imwrite("./images/Users." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
            count += 1
        # Save the captured image into the images directory
        # cv2.imshow('image', img)
    # # Press Escape to end the program.
            k = cv2.waitKey(100) & 0xff
            if k < 30:
                break
            elif count >= 10:
                break


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict/{name}")
async def read_faces(name):
    id = 0
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read('./yml/train.yml')
    face_cascade_Path = "./yml/haarcascade_frontalface_default.xml"

    faceCascade = cv2.CascadeClassifier(face_cascade_Path)

    trainPersonFileName = name

    img = cv2.imread('./find_me/%s.jpg' %trainPersonFileName)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        if (confidence < 100):
        # print(id)
            id = names[id]
            confidence = "{0}%".format(round(100 - confidence))
        else:
        # Unknown Face
            id = "Not able to Match ?"
            confidence = "{0}%".format(round(100 - confidence))

    result = id
    result2 = confidence
    response_object = {"name":result,"match": result2}
    return response_object



@app.post("/upload")
def save_upload_file(file: UploadFile, destination: Path)-> None:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            save_upload_file_tmp(file)
            handle_upload_file(file)
        
        
    except Exception:
        return {"message": "There was an error uploading the file {Exception}"}
    finally:
        file.file.close()
        return {"message": f"Successfully uploaded {file.filename} and {destination}"}




def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    try:
        suffix = Path(upload_file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            tmp_path = Path(tmp.name)
    finally:
        upload_file.file.close()
    return tmp_path


def handle_upload_file(
    upload_file: UploadFile, handler: Callable[[Path], None]
) -> None:
    tmp_path = save_upload_file_tmp(upload_file)
    try:
        handler(tmp_path)  # Do something with the saved temp file
    finally:
        tmp_path.unlink()  # Delete the temp file


