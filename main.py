from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch
import cv2
import base64
import easyocr
import string
import aiofiles
import asyncio
import os
import uuid
from datetime import datetime
# performance test code
# from pyinstrument import Profiler


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
reader = easyocr.Reader(['en'], gpu=True)
if torch.cuda.is_available():
    vehicle_model = YOLO('yolov8n.pt').to('cuda')
    plate_model = YOLO('plate_model.pt').to('cuda')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
else:
    vehicle_model = YOLO('yolov8n.pt')
    plate_model = YOLO('plate_model.pt')
    
modelclass = [2, 3, 5, 7] #COCO model classes for car, truck, bus, motorbike

dict_ch_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'} # Mapping for characters to integers
dict_int_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'} # Mapping for integers to characters

#Process the video frame by frame and returns a dictionary of unique licence plates with their scores and base64 of image.
async def process_video(video_path):
    # performance test code
    # profiler = Profiler()
    # profiler.start()
    # start_time = time.time()
    # /performance test code
    
    cap = cv2.VideoCapture(video_path) # Open the video file
    unique_licence_plates = {} # Dictionary to store unique licence plates
    frame_nmr = -1
    ret = True # Flag to check if the video frame is read successfully
    while ret: # Loop through the video frames
        frame_nmr += 1 # Increment the frame number
        ret, frame = cap.read()
        if ret:
            cars = vehicle_model(frame, verbose=False)[0] # Detect vehicles in the video frame
            cars_ = [] # List to store vehicle coordinates
            for car in cars.boxes.data.tolist(): # Loop through the detected vehicles in frame
                x1, y1, x2, y2, score, class_id = car # Extract vehicle coordinates
                if int(class_id) in modelclass: # Check if the detected vehicle is a car, truck, bus or motorbike
                    cars_.append([x1, y1, x2, y2, score]) # Append the vehicle coordinates to the list
            await process_frame(frame, frame_nmr, unique_licence_plates, cars_) # Start license plate detection and recognition
    cap.release()
    
    #performance test code
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # avg_time = elapsed_time / frame_nmr if frame_nmr > 0 else 0
    # print(f"Elapsed time to process video: {elapsed_time} seconds")
    # print(f"Average time per frame: {avg_time} seconds")
    # profiler.stop()
    # profiler.print()
    # profiler.write_html("./profiler.html")
    # /performance test code
    
    return unique_licence_plates

async def process_frame(frame, frame_nmr, unique_licence_plates, cars_):
    licence_plates = plate_model(frame, verbose= False)[0] # Detect licence plates in the video frame
    await handle_licence_plates(frame, licence_plates, cars_, frame_nmr, unique_licence_plates)

async def handle_licence_plates(frame, licence_plates, cars_, frame_nmr, unique_licence_plates):
    tasks = [] 
    for licence_plate in licence_plates.boxes.data.tolist(): # Loop through the detected licence plates in the frame
        licence_plate = licence_plate[:5] # Extract the licence plate coordinates
        task = asyncio.create_task(prepare_for_ocr(licence_plate, frame, cars_, frame_nmr)) # Associate the licence plate with a vehicle
        tasks.append((task, licence_plate)) # Append the task and licence plate to the list
    for task, licence_plate in tasks: 
        licence_plate_text, car_image, licence_plate_text_score = await task # Get the licence plate text, car image and score
        if licence_plate_text is not None and licence_plate_text_score is not None and car_image is not None:
            if licence_plate_text not in unique_licence_plates:
                unique_licence_plates[licence_plate_text] = {'car_image': car_image, 'score': licence_plate_text_score}
            else:
                # Update the score if the new score is higher than the existing one
                existing_score = unique_licence_plates[licence_plate_text]['score']
                if licence_plate_text_score is not None and existing_score is not None and licence_plate_text_score > existing_score: # Update the score if the new score is higher than the existing one
                    unique_licence_plates[licence_plate_text]['score'] = licence_plate_text_score

async def prepare_for_ocr(licence_plate, frame, cars_, frame_nmr):
    x1, y1, x2, y2, score = licence_plate
    car_id = await get_car(licence_plate, cars_)
    licence_plate_text, licence_plate_text_score, car_image = None, None, None
    if car_id != -1:
        licence_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :] # Crop the licence plate from the frame
        licence_plate_crop_gray = cv2.cvtColor(licence_plate_crop, cv2.COLOR_BGR2GRAY) # Convert the cropped licence plate to grayscale
        _, licence_plate_crop_thresh = cv2.threshold(licence_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV) # Apply thresholding to the cropped licence plate
        licence_plate_text, licence_plate_text_score = await read_text(licence_plate_crop_thresh) # Read the text from the cropped licence plate
        if licence_plate_text is not None:
            if car_id != (-1, -1, -1, -1, -1): # Check if the vehicle is found
                xcar1, ycar1, xcar2, ycar2, car_id = car_id # Extract the vehicle coordinates
                car_crop = frame[int(ycar1):int(ycar2), int(xcar1): int(xcar2), :] # Crop the vehicle from the frame
                car_image = await convertBase64(car_crop) # Convert the vehicle to base64
    return licence_plate_text, car_image, licence_plate_text_score

async def licence_complies_format(text):
    if len(text) != 7:
        return False
    if (text[0] in string.ascii_uppercase or text[0] in dict_int_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_ch_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_ch_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_char.keys()):
        return True
    else:
        return False
    
async def format_licence(text):
    # Convert the licence plate to the correct format
    mapping = [
        dict_int_char, dict_int_char, dict_ch_int, dict_ch_int, 
        dict_int_char, dict_int_char, dict_int_char
    ]
    return ''.join(mapping[i].get(char, char) for i, char in enumerate(text))

async def read_text(licence_plate_crop):
    plate_text = reader.readtext(licence_plate_crop) # Read the text from the cropped licence plate
    for _, text, score in plate_text:
        text = text.upper().replace(' ', '') # Convert the text to uppercase and remove spaces
        if await licence_complies_format(text): # Check if the licence plate complies with the correct format
            return await format_licence(text), score # Return the formatted licence plate and score
    return None, None

async def get_car(licence_plate, vehicle_track_ids):
    x1, y1, x2, y2, score = licence_plate
    for vehicle_id in vehicle_track_ids: # Loop through the vehicle track IDs
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_id
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2: # Check if the licence plate is associated with a vehicle
            return vehicle_id
    return -1, -1, -1, -1, -1

async def convertBase64(image):
    is_success, buffer = cv2.imencode('.jpg', image)
    if is_success:
        return base64.b64encode(buffer).decode('utf-8')
    else:
        return None

@app.post("/upload")
async def upload_video(file: UploadFile = File(...), start_latitude: str = Form(None), start_longitude: str = Form(None), end_latitude: str = Form(None), end_longitude: str = Form(None)):
    print("File Uploaded", file.filename, file.content_type)
    if file.content_type != 'video/mp4':
        return JSONResponse(content={"error": "Unsupported file format. Please upload an MP4 video."}, status_code=400)
    
    os.makedirs('uploads', exist_ok=True)
    
    timestamp = datetime.now().strftime('%d-%m-%H%M%S')
    id = str(uuid.uuid4())
    filename = f'uploads/{timestamp}_{id}.mp4'
    async with aiofiles.open(filename, 'wb') as f:
        await f.write(await file.read())
    print("processing video", filename)

    unique_licence_plates = await process_video(filename)

    response_data = [
        {
            "licence_plate": plate, "score": round(score['score'] * 100, 2), 
            "start_latitude": start_latitude, "start_longitude": start_longitude,
            "end_latitude": end_latitude, "end_longitude": end_longitude,
            "car_image": "data:image/jpeg;base64," + score.get('car_image', '')
        }
        for plate, score in unique_licence_plates.items()
    ]
    return JSONResponse(content=response_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)