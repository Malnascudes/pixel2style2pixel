from argparse import Namespace
import pprint
import PIL
from PIL import ImageDraw
import torch
import torchvision.transforms as transforms
import os
from models.psp import pSp
import time
import dlib
from scripts.align_all_parallel import align_face
from scripts.align_all_parallel import get_landmark
import numpy as np
import json

# class ModelHandler(BaseHandler): # for TorchServe  it need to inherit from BaseHandler
class ModelHandler():
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.experiment_data_args = {
            "model_path": "pretrained_models/psp_ffhq_encode.pt",
            "image_path": "notebooks/images/input_img.jpg",
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
        self.allowed_extensions = ['png', 'jpg', 'jpeg']

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """

        # From Torchserve on how to init. I guess manifest and other variables are necessary for propper working.
        # Content will be handled by Torchserve and include the necessary information
        '''
        self._context = context

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        
        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        '''

        model_pt_path = self.experiment_data_args['model_path']

        # self.device = "cpu" # Not fully working on cpu yet
        self.device = "cuda"

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.load(model_pt_path, map_location='cpu')
        self.alignment_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        opts = self.model['opts']
        pprint.pprint(opts)
        # update the training options
        opts['checkpoint_path'] = model_pt_path
        if 'learn_in_w' not in opts:
            opts['learn_in_w'] = False
        if 'output_size' not in opts:
            opts['output_size'] = 1024

        opts = Namespace(**opts)
        self.net = pSp(opts)
        self.net.eval()
        # self.net.cuda()
        self.net = self.net.to(self.device)

        self.img_transforms = self.experiment_data_args['transform']

        self.mean_encodings_file_path = 'mean_latents.pt'

        self.initialized = True


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        input_image = self.load_image(data)

        self.processed_input_image = self.preprocess(input_image)

        model_output = self.inference(self.processed_input_image)

        return self.postprocess(model_output)

    def load_image(self, full_image_path: str):
        print(f'Loading {full_image_path}')
        i_t = time.time()
        image_extension = full_image_path.split('.')[-1]
        if image_extension not in self.allowed_extensions:
            print(f'{image_extension} file extension not allowed')
            return None

        input_image = PIL.Image.open(full_image_path)
        print(f'Image loaded in {time.time() - i_t} seconds')

        return input_image

    def preprocess(self, input_image):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        original_image = self.run_alignment(input_image)
        rgb_image = original_image.convert("RGB")
        transformed_image = self.img_transforms(rgb_image)

        preprocessed_image = transformed_image

        return preprocessed_image

    def run_alignment(self, input_image):
        aligned_image = align_face(input_image=input_image, predictor=self.alignment_predictor)
        return aligned_image

    def get_face_landmarks(self, input_image):
        img_array = np.asarray(input_image.convert('RGB')).astype(np.uint8)
        lm = get_landmark(img_array, self.alignment_predictor)

        return lm

    @staticmethod
    def plot_landmarks_over_image(input_image, landmarks):
        # Create a PIL draw object to draw the landmarks on the image
        draw = ImageDraw.Draw(input_image)

        # Iterate over the face landmarks and draw them on the image
        for i, landmark in enumerate(landmarks):
            x, y = landmark
            r = 3
            leftUpPoint = (x-r, y-r)
            rightDownPoint = (x+r, y+r)
            twoPointList = [leftUpPoint, rightDownPoint]
            draw.ellipse(twoPointList, fill='yellow')
            # draw.point((x, y), fill='red', size)
            # draw.regular_polygon((x, y, 3), n_sides = 3, fill='yellow')
            draw.text((x, y), text=str(i), fill='red', anchor='lt')

        # Return the image with the landmarks drawn on it
        return input_image

    @staticmethod
    def save_landmarks(landmarks, output_path):
        with open(output_path, 'w') as file:
            for tup in landmarks:
                file.write(','.join(str(item) for item in tup) + '\n')

        return None

    @staticmethod
    def save_landmarks_structured(landmarks, output_path):
        extension = output_path.split('.')[-1]
        output_path = output_path.replace(extension, 'json')

        lm_chin = landmarks[0: 17]  # left-right
        lm_eyebrow_left = landmarks[17: 22]  # left-right
        lm_eyebrow_right = landmarks[22: 27]  # left-right
        lm_nose = landmarks[27: 31]  # top-down
        lm_nostrils = landmarks[31: 36]  # top-down
        lm_eye_left = landmarks[36: 42]  # left-clockwise
        lm_eye_right = landmarks[42: 48]  # left-clockwise
        lm_mouth_outer = landmarks[48: 60]  # left-clockwise
        lm_mouth_inner = landmarks[60: 68]  # left-clockwise

        landmarks_dict = {
            "chin": lm_chin.tolist(),
            "eyebrow_left": lm_eyebrow_left.tolist(),
            "eyebrow_right": lm_eyebrow_right.tolist(),
            "nose": lm_nose.tolist(),
            "nostrils": lm_nostrils.tolist(),
            "eye_left": lm_eye_left.tolist(),
            "eye_right": lm_eye_right.tolist(),
            "mouth_outer": lm_mouth_outer.tolist(),
            "mouth_inner": lm_mouth_inner.tolist(),
        }

        with open(output_path, 'w') as file:
            json.dump(landmarks_dict, file, indent=4)

        return None

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = model_input

        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        return postprocess_output

if __name__ == "__main__":
    model_handler = ModelHandler()
    model_handler.initialize(None)

    print('Model initialized')

    image_path = '/home/carles/repos/matriu.id/ideal/Datasets/sorolla-test-faces/minimum-subset/CFD-AM-229-224-N.jpg'
    i_t = time.time()

    model_output = model_handler.handle(image_path, None)
    print(f'Image processed in {time.time() - i_t} seconds')
