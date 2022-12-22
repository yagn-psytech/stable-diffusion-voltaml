import banana_dev as banana
import base64
from io import BytesIO
from PIL import Image

model_inputs = {
	"prompt": "table full of muffins",
	"negative_prompt":"",
	"num_inference_steps":10,
	"guidance_scale":9,
	"height":512,
	"width":512,
	"seed":3242
}

api_key = ""
model_key = ""

# Run the model
import time
t1 = time.time()
out = banana.run(api_key, model_key, model_inputs)
t2 = time.time()
print("Inference in ",t2-t1,"seconds")
# Extract the image and save to output.jpg
image_byte_string = out["modelOutputs"][0]["image_base64"]
image_encoded = image_byte_string.encode('utf-8')
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("output.jpg")
