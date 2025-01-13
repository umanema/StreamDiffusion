import time
import asyncio
import cv2
import time
import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.wrapper import StreamDiffusionWrapper

import torchvision.transforms as transforms

import numpy as np

_prompt = "photography, dynamic view, modernism mid-scale store facade design, Canon EOS 5D Mark IV, EF 24mm f/1.4L II USM lens, set f/2.8 for a depth of field that highlights architectural photography, design should blend different materials in a way that’s both innovative and aesthetically pleasing, focus on a clean composition"

def init_stream(device: torch.device, torch_dtype: torch.dtype):
    stream = StreamDiffusionWrapper(
        #model_id_or_path="stabilityai/sd-turbo",
        model_id_or_path="D:/NSYNK/StreamDiffusion/models/Model/architecturerealmix_v11.safetensors",
        use_tiny_vae=True,
        device=device,
        dtype=torch_dtype,
        t_index_list=[20, 29],
        frame_buffer_size=1,
        width=512,
        height=512,
        use_lcm_lora=False,
        output_type="np",
        warmup=10,
        acceleration="tensorrt",
        mode="img2img",
        use_denoising_batch=True,
        cfg_type="none",
        # do_add_noise=False
    )

    stream.prepare(
        prompt=_prompt,
        num_inference_steps=30,
        guidance_scale=1.2,
    )

    return stream



async def run_prediction():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16
    #pipeline = Pipeline(config, device, torch_dtype)
    #app = App(config, pipeline).app
    stream_wrapper = init_stream(device, torch_dtype)
    transform = transforms.ToTensor()

    # Open the webcam (0 is usually the default camera)
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./Raser-Hamburg.mp4")

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    try:
        while True:
            last_time = time.time()
            # Capture a frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab a frame.")
                # Restart the video by setting the frame position to 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue  # Skip to the next iteration
            
            # Convert the frame
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (512,512))
            
            # image_tensor = stream.preprocess_image(frame)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #image_tensor = stream_wrapper.stream.image_processor.preprocess(frame, 512, 512).to(device=device, dtype=torch_dtype)

            image_tensor = transform(frame)
            output_image = stream_wrapper(image=image_tensor, prompt=_prompt)
            #output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            #opencv_image = np.array(output_image)
            #cv2.imshow('Webcam', output_image)
            cv2.imshow('Webcam', frame)

            fps = 1 / (time.time() - last_time)
            print(fps)
            # Check for user input to close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
            
            # Allow asyncio to manage other tasks
            # await asyncio.sleep(0.01)

    finally:
        # Release the webcam and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

async def run():
    await run_prediction()




if __name__ == "__main__":
    # import uvicorn

    # uvicorn.run(
    #     "main:app",
    #     host=config.host,
    #     port=config.port,
    #     reload=config.reload,
    #     ssl_certfile=config.ssl_certfile,
    #     ssl_keyfile=config.ssl_keyfile,
    # )
    
    asyncio.run(run())


