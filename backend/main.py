from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from io import BytesIO
from PIL import Image
import contextlib
import numpy as np
import os
import torch
from torchvision import transforms
import pystiche
from pystiche.image.io import import_from_pil, export_to_pil
import cv2
import mediapipe as mp

app = FastAPI()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def get_state_dict():
    try:
        return torch.load('example_transformer.pth')

    except FileNotFoundError:
        @contextlib.contextmanager
        def suppress_output():
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    yield

        url = "https://download.pystiche.org/models/example_transformer.pth"

        with suppress_output():
            return torch.hub.load_state_dict_from_url(url)


transformer = pystiche.demo.transformer().to(device)
transformer.load_state_dict(get_state_dict())
transformer.eval()


def perform_nst(input_image: Image.Image) -> torch.Tensor:
    image_tensor = transforms.ToTensor()(input_image)
    print(f'Image tensor size: {image_tensor.size()}')
    if len(list(image_tensor.size())) == 4:
        image_tensor = image_tensor.squeeze(0)
    image_tensor.to(device)
    print(f'Image tensor size: {image_tensor.size()}')

    with torch.no_grad():
        output_image_tensor = transformer(image_tensor)
    return output_image_tensor


@app.post('/nst/')
async def neural_style_transfer(file: UploadFile = File(...)):
    if file.content_type not in ['image/jpeg']:
        raise HTTPException(400, detail='Invalid file type')

    original_image = Image.open(file.file)
    original_image_tensor = import_from_pil(original_image, device=device, make_batched=True)
    # nst_image = transform_to_pillow_image(perform_nst(original_image))
    # nst_image = perform_nst(original_image)
    nst_image = transformer(original_image_tensor)
    pillow_nst_image = export_to_pil(nst_image)

    response_image = BytesIO()
    pillow_nst_image.save(response_image, 'JPEG')
    response_image.seek(0)

    return StreamingResponse(response_image, media_type='image/jpeg')


mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

BG_COLOR = (255, 255, 255)


@app.post('/nst2/')
async def test(file: UploadFile = File(...)):
    if file.content_type not in ['image/jpeg']:
        raise HTTPException(400, detail='Invalid file type')

    original_image_pillow = Image.open(file.file)
    original_image_numpy = np.array(original_image_pillow)  # RGB

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        image_height, image_width, _ = original_image_numpy.shape

        bg_image = np.zeros(original_image_numpy.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        # Selfie Segmentation
        results = selfie_segmentation.process(original_image_numpy)
        condition = np.stack((cv2.blur(results.segmentation_mask, (5, 5)),) * 3, axis=-1) > 0.6
        segmented_image_numpy = np.where(condition, original_image_numpy, bg_image)
        segmented_image_pillow = Image.fromarray(segmented_image_numpy)

        # NST
        segmented_image_tensor = import_from_pil(segmented_image_pillow, device=device, make_batched=True)
        nst_image_tensor = transformer(segmented_image_tensor)
        nst_image_pillow = export_to_pil(nst_image_tensor)

        # Combine NST image with original image
        nst_image_numpy = np.array(nst_image_pillow)
        output_image_numpy = np.where(condition, nst_image_numpy, original_image_numpy)
        output_image_pillow = Image.fromarray(output_image_numpy)

        # Output
        response_image = BytesIO()
        output_image_pillow.save(response_image, 'JPEG')
        response_image.seek(0)

        return StreamingResponse(response_image, media_type='image/jpeg')


@app.get('/')
async def hello_world():
    return {'Message': 'Hello World'}
