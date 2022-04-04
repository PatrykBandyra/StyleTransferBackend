from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from io import BytesIO
from PIL import Image
import contextlib
import os
import torch
from torchvision import transforms, utils
import pystiche

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
    if len(list(image_tensor.size())) == 4:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor.to(device)

    with torch.no_grad():
        output_image_tensor = transformer(image_tensor)
    return output_image_tensor


def transform_to_pillow_image(tensor_image) -> Image.Image:
    grid = utils.make_grid(tensor_image)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return Image.fromarray(ndarr)


@app.post('/nst')
async def neural_style_transfer(file: UploadFile = File(...)):
    if file.content_type not in ['image/jpeg']:
        raise HTTPException(400, detail='Invalid file type')

    original_image = Image.open(file.file)
    nst_image = transform_to_pillow_image(perform_nst(original_image))

    response_image = BytesIO()
    nst_image.save(response_image, 'JPEG')
    response_image.seek(0)

    return StreamingResponse(response_image, media_type='image/jpeg')
