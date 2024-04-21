# Citizen's Eye - FYP

Backend of the Citizen's Eye Final Year Project, processes videos of cars to detect licence plates and returns Licence Plate Data, Images of the car, and confidence results

## Installation 
### Clone Repo
```bash
git clone https://github.com/dylan-mp4/Citizen-eye.git
```
### Create and activate environment
```bash
python3 -m venv env
```
Then activate using:
#### On windows
```bash
.\env\Scripts\activate
```
#### On Linux
```bash
source env/bin/activate
```
### Pre Requisites 
#### CUDA
This project is intended to run using CUDA architecture, along with a matching PyTorch version
CUDA drivers can be obtained if not already installed from:
https://developer.nvidia.com/cuda-downloads - please note that PyTorch currently only supports CUDA versions 12.1 and 11.8
#### PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
To test if PyTorch and CUDA was installed successfully:
```bash
python testgpu.py
```
### Reqs
```bash
pip install -r requirements.txt
```
## Run 1 instance
```bash
python main.py
``` 
## Run multiple instances
### Windows
```bash
uvicorn main:app --workers 3
```
### Linux
You may have to install the gunicorn package first as it is not in reqs
```bash
gunicorn -w 3 -k uvicorn.workers.UvicornWorker main:app
```
## Tech Stack

**Client:** Dart/Flutter

**Server:** Uvicorn, FastAPI
## API Reference

#### Upload File

```http
  POST /upload
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `file` | `video` | **Required**. video to be processed |
| `start_latitude` | `string` | **Optional**. starting GPS lat |
| `start_longitude` | `string` | **Optional**. starting GPS lon |
| `end_latitude` | `string` | **Optional**. ending GPS lat |
| `end_longitude` | `string` | **Optional**. ending GPS lon |

### Returns
| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `licence_plate` | `string` | Licence plate text |
| `score` | `number(float)` | decimal score / confidence of plate being correct |
| `car_image` | `string` | Base64 represented image includes starting header |
| `start_latitude` | `string` | **Optional**. starting GPS lat |
| `start_longitude` | `string` | **Optional**. starting GPS lon |
| `end_latitude` | `string` | **Optional**. ending GPS lat |
| `end_longitude` | `string` | **Optional**. ending GPS lon |
## Author

- [@dylan-mp4](https://github.com/dylan-mp4)
