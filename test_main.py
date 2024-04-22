import pytest
from fastapi.testclient import TestClient
from main import app, upload_video, process_video, process_frame, handle_licence_plates, prepare_for_ocr, licence_complies_format, format_licence, read_text, get_car, convertBase64
from starlette import status
import httpx
from unittest.mock import AsyncMock, patch
import cv2
import numpy as np
import base64
from fastapi import UploadFile

client = TestClient(app)

@pytest.fixture
def create_test_video(tmp_path):
    test_video_path = tmp_path / "test_video.mp4"
    test_video_path.write_text("Fake video content", encoding='utf-8')
    return str(test_video_path)

    
class AsyncContextManagerMock(AsyncMock):
    async def __aenter__(self):
        return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    pass

def test_process_video():
    # Add your test code here
    pass
def test_process_frame():
    # Add your test code here
    pass

@pytest.mark.asyncio
@patch('main.get_car', new_callable=AsyncMock)
@patch('main.read_text', new_callable=AsyncMock)
@patch('main.convertBase64', new_callable=AsyncMock)
@patch.object(cv2, 'cvtColor')
@patch.object(cv2, 'threshold')
async def test_prepare_for_ocr(mock_threshold, mock_cvtColor, mock_convertBase64, mock_read_text, mock_get_car):
    mock_get_car.return_value = (10, 20, 30, 40, 0.8)
    mock_read_text.return_value = ('AB12ABC', 0.9)
    mock_convertBase64.return_value = 'base64string'
    mock_cvtColor.return_value = 'gray_image'
    mock_threshold.return_value = (0, 'thresh_image')

    licence_plate = (10, 20, 30, 40, 0.8)
    frame = np.zeros((100, 100, 3))
    cars_ = 'cars'
    frame_nmr = 1

    licence_plate_text, car_image, licence_plate_text_score = await prepare_for_ocr(licence_plate, frame, cars_, frame_nmr)

    assert licence_plate_text == 'AB12ABC'
    assert car_image == 'base64string'
    assert licence_plate_text_score == 0.9

    mock_get_car.assert_called_once()
    mock_read_text.assert_called_once()
    mock_convertBase64.assert_called_once()
    mock_cvtColor.assert_called_once()
    mock_threshold.assert_called_once()
    
@pytest.mark.asyncio
async def test_licence_complies_format():
    assert await licence_complies_format('AB123CD') == True
    assert await licence_complies_format('A123BCD') == True
    assert await licence_complies_format('AB12CDE') == True
    assert await licence_complies_format('1234567') == False
    assert await licence_complies_format('') == False
    assert await licence_complies_format('AB123C') == False
    assert await licence_complies_format('AB123CDE') == False
    assert await licence_complies_format('AB123CD1') == False

@pytest.mark.asyncio
@pytest.mark.parametrize("text,expected_output", [
    ('11AA666', 'II44GGG'),
    ('55SS555', 'SS55SSS'),
    ('01JA65Y', 'OI34GSY'),
])
async def test_format_licence(text, expected_output):
    with patch('main.dict_int_char', {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}), \
         patch('main.dict_ch_int', {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}):
        output = await format_licence(text)
        assert output == expected_output
        
def test_read_text():
    # Add your test code here
    pass
def test_get_car():
    # Add your test code here
    pass
@pytest.mark.asyncio
@patch.object(cv2, 'imencode')
@patch.object(base64, 'b64encode')
async def test_convertBase64(mock_b64encode, mock_imencode):
    mock_imencode.return_value = (True, b'image_buffer')
    mock_b64encode.return_value = b'base64string'

    image = 'image'
    output = await convertBase64(image)

    assert output == 'base64string'

    mock_imencode.assert_called_once_with('.jpg', image)
    mock_b64encode.assert_called_once_with(b'image_buffer')
    
@pytest.mark.asyncio
@patch('main.process_video', new_callable=AsyncMock)
@patch('main.aiofiles.open', new_callable=AsyncContextManagerMock)
async def test_upload_video(mock_aiofiles_open, mock_process_video):
    mock_process_video.return_value = {
        'AB12CDE': {'score': 0.9, 'car_image': 'image_data'}
    }

    file = AsyncMock(spec=UploadFile)
    file.filename = 'test.mp4'
    file.content_type = 'video/mp4'

    start_latitude = '51.5074'
    start_longitude = '0.1278'
    end_latitude = '52.3555'
    end_longitude = '1.1743'

    response = await upload_video(file, start_latitude, start_longitude, end_latitude, end_longitude)

    assert response.status_code == 200
    assert response.content == [
        {
            "licence_plate": 'AB12CDE', "score": 90.0, 
            "start_latitude": start_latitude, "start_longitude": start_longitude,
            "end_latitude": end_latitude, "end_longitude": end_longitude,
            "car_image": "data:image/jpeg;base64,image_data"
        }
    ]

    mock_process_video.assert_called_once()
    mock_aiofiles_open.assert_called_once()
