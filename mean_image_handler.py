from argparse import Namespace
import pprint
import PIL
import torch
import torchvision.transforms as transforms
import os
from models.psp import pSp
import time
import dlib
from scripts.align_all_parallel import align_face
from utils.common import tensor2im
from utils.interpolate import interpolate
from ts.torch_handler.base_handler import BaseHandler
from ts.context import Context

class ModelHandler(BaseHandler): # for TorchServe  it need to inherit from BaseHandler
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.experiment_data_args = {
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
        self._context = context

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        '''
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        
        # Read model serialize/pt file
        '''
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)


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
        image_bytes = data[0]['body']
        input_image = self.load_image(image_bytes)

        self.processed_input_image = self.preprocess(input_image)
        self.mean_image_encoding = self.load_mean_encoding()

        model_output = self.inference(self.processed_input_image)

        return self.postprocess(model_output)

    def load_image(self, image_bytes: str):
        i_t = time.time()

        input_image = PIL.Image.open(io.BytesIO(image_bytes))
        print(f'Image loaded in {time.time() - i_t} seconds')

        return input_image


    def load_mean_encoding(self):
        mean_image_encoding = None

        if os.path.exists(self.mean_encodings_file_path):
            print(f'Loading mean encoding from {self.mean_encodings_file_path}')
            mean_image_encoding = torch.load(self.mean_encodings_file_path)
        else: # means its first image
            print(f'No encoding found in {self.mean_encodings_file_path}. Using input image as mean.')
            mean_image_encoding = self.encode_image(self.processed_input_image)

        return mean_image_encoding

    def save_mean_encoding(self, mean_encoding):
        self.mean_image_encoding = mean_encoding

        torch.save(self.mean_image_encoding, self.mean_encodings_file_path)

        return None

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

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        input_image_encoding = self.encode_image(model_input)

        mean_latent = self.merge_latent_to_mean(input_image_encoding)

        output_image, result_latent = self.decode_image_latent(mean_latent.unsqueeze(0))
        output_pil_image = tensor2im(output_image)

        model_output = output_pil_image, mean_latent, input_image_encoding

        return model_output

    def encode_image(self, image):
        input_image_tensor = image.unsqueeze(0)
        image_latents = self.net.encoder(input_image_tensor.to(self.device).float())

        # normalize with respect to the center of an average face (models/psp.py L75)
        image_latents = image_latents + self.net.latent_avg.repeat(image_latents.shape[0], 1, 1)
        return image_latents

    def decode_image_latent(self, latent):
        images, result_latent = self.net.decoder([latent],
                                                input_is_latent=True,
                                                randomize_noise=False,
                                                return_latents=True)
        return images[0], result_latent

    def merge_latent_to_mean(self, input_latent):
        all_latents = torch.stack([torch.squeeze(self.mean_image_encoding).to(self.device), torch.squeeze(input_latent).to(self.device)], dim=0)
        mean_latent = torch.mean(all_latents, dim=0)
        return mean_latent   

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        return postprocess_output

    def generate_animation(self, encodings, FPS=25, duration_per_image=1):
        encodings = [encoding.squeeze() for encoding in encodings]
        animation_frames = []

        print('Generating morphing animation')
        for i, latent in enumerate(interpolate(
                latents_list=encodings, duration_list=[duration_per_image]*len(encodings),
                interpolation_type="linear",
                loop=False,
                FPS=FPS,
            )):
            frame_image, _ = self.decode_image_latent(latent)
            frame_image = tensor2im(frame_image)
            animation_frames.append(frame_image)

        return animation_frames

    @staticmethod
    def save_frames(video_frames, output_path, output_format = 'tiff'):
        frame_folder = os.path.join(output_path, 'frames')
        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)

        for i, frame_image in enumerate(video_frames):
            frame_image_save_path = os.path.join(frame_folder,f'frame_{i:04}.{output_format}')
            i_t = time.time()
            frame_image.save(frame_image_save_path)
            print(f'Took {time.time() - i_t} to save frame')

if __name__ == "__main__":
    model_dir = "pretrained_models"
    model_name = "psp_ffhq_encode.pt"
    manifest = {'model': {'serializedFile': model_name}}
    context = Context(model_dir=model_dir, model_name="pSp", manifest=manifest,batch_size=1,gpu=0,mms_version="1.0.0")

    model_handler = ModelHandler()
    model_handler.initialize(context)

    print('Model initialized')

    image_path = '/home/carles/repos/matriu.id/ideal/Datasets/sorolla-test-faces/minimum-subset/CFD-AM-229-224-N.jpg'
    i_t = time.time()

    output_image, result_latent, input_image1_encoding = model_handler.handle(image_path, None)
    print(f'Image processed in {time.time() - i_t} seconds')

    model_handler.save_mean_encoding(result_latent)

    image2_path = '/home/carles/repos/matriu.id/ideal/Datasets/sorolla-test-faces/minimum-subset/CFD-BF-051-035-N.jpg'
    iii_t = time.time()

    output_image, result_latent, input_image2_encoding = model_handler.handle(image2_path, None)

    animation_frames = model_handler.generate_animation([input_image1_encoding, input_image2_encoding])

    print('Saving Video Frames')
    output_path = 'tests/output_frames/'
    video_t = time.time()
    model_handler.save_frames(animation_frames, output_path)
    print(f'Saved video in {time.time() - video_t}')
