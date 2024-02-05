"""
This script defines a FastAPI app that processes a video to detect vehicles and their license plates.
The app has an endpoint '/upload/' that accepts an MP4 video file and returns a JSON response containing unique license plates and their scores.
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import util
app = FastAPI()

# Define the CORS settings
origins = ["*"]  # You can configure this to allow requests only from specific origins.

# Use the CORS middleware with the defined origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def process_video(video_path):
    """
    Processes a video to detect vehicles and their license plates.

    Args:
        video_path (str): The path to the video file.

    Returns:
        dict: A dictionary containing unique license plates and their scores.
    """
    results = {}
    # load models
    coco_model = YOLO('yolov8n.pt')
    coco_model.to('cuda')
    license_plate_detector = YOLO('license_plate_detector.pt')
    license_plate_detector.to('cuda')
    # load video
    cap = cv2.VideoCapture(video_path)

    vehicles = [2, 3, 5, 7]
    unique_license_plates = {}  # Use a dictionary to store license plates and their scores

    # read frames
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        if ret:
            results[frame_nmr] = {}
            # detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # assign license plate to car
                car_id = util.get_car(license_plate, detections_)

                if car_id != -1:
                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number
                    license_plate_text, license_plate_text_score = util.read_license_plate(license_plate_crop_thresh)

                    if license_plate_text is not None:
                        # Store unique license plates and their scores in the dictionary
                        if license_plate_text not in unique_license_plates:
                            unique_license_plates[license_plate_text] = license_plate_text_score

    return unique_license_plates

@app.post("/upload/")

async def upload_video(file: UploadFile = File(...), start_geolocation: str = Form(None), end_geolocation: str = Form(None)):
    """
    Uploads a video file and processes it to extract unique license plates and their scores.

    Args:
        file (UploadFile): The video file to be uploaded.
        start_geolocation (str, optional): The geolocation where the video recording started. Defaults to None.
        end_geolocation (str, optional): The geolocation where the video recording ended. Defaults to None.

    Returns:
        JSONResponse: A JSON response containing the unique license plates and their scores.
    """
    if file.content_type != 'video/mp4':
        return JSONResponse(content={"error": "Unsupported file format. Please upload an MP4 video."}, status_code=400)

    with open('uploaded_video.mp4', 'wb') as f:
        f.write(file.file.read())

    unique_license_plates = process_video('uploaded_video.mp4')

    # Print the unique license plates and their scores
    response_data = [{"License Plate": plate, "Score": score, "Start Geolocation": start_geolocation, "End Geolocation": end_geolocation} for plate, score in unique_license_plates.items()]

    return JSONResponse(content=response_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    