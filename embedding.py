import openvino as ov
from pathlib import Path
import notebook_utils as utils
from notebook_utils import download_ir_model

if not Path("./notebook_utils.py").exists():
    import requests
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

base_model_dir = "model"
precision = "FP16"
detection_model_name = "person-detection-0202"

download_det_model_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{detection_model_name}/{precision}/{detection_model_name}.xml"

detection_model_path = download_ir_model(
    download_det_model_url, Path(base_model_dir) / detection_model_name / precision
)

reidentification_model_name = "person-reidentification-retail-0287"
download_reid_model_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{reidentification_model_name}/{precision}/{reidentification_model_name}.xml"

reidentification_model_path = download_ir_model(
    download_reid_model_url,
    Path(base_model_dir) / reidentification_model_name / precision,
)
core = ov.Core()

class Model:
    def __init__(self, model_path, batchsize=1, device="AUTO"):
        self.model = core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]
        for layer in self.model.inputs:
            input_shape = layer.partial_shape
            input_shape[0] = batchsize
            self.model.reshape({layer: input_shape})
        self.compiled_model = core.compile_model(model=self.model, device_name=device)
        self.output_layer = self.compiled_model.output(0)

    def predict(self, input_data):
        result = self.compiled_model(input_data)[self.output_layer]
        return result

extractor = Model(reidentification_model_path, batchsize=-1, device="AUTO")

def preprocess(frame, height, width):
    resized_image = cv2.resize(frame, (width, height))
    resized_image = resized_image.transpose((2, 0, 1))
    input_image = np.expand_dims(resized_image, axis=0).astype(np.float32)
    return input_image

def get_embedding(cutout):
    img = preprocess(cutout, extractor.height, extractor.width)
    features = extractor.predict(img)
    return features

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)
