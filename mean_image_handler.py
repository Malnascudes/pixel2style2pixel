from argparse import Namespace
import time
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import os

sys.path.append(".")
sys.path.append("..")

from datasets import augmentations
from utils.common import tensor2im, log_input_image
from models.psp import pSp
import dlib
from scripts.align_all_parallel import align_face
from utils.interpolate import interpolate
import PIL
import os

######################################################################################################3
# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""

# from ts.torch_handler.base_handler import BaseHandler

# class ModelHandler(BaseHandler):
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

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # self.model = torch.jit.load(model_pt_path)
        self.model = torch.load(model_pt_path, map_location='cpu')

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
        self.net.cuda()

        self.img_transforms = self.experiment_data_args['transform']

        self.mean_encodings_file_path = 'mean_latents.pt'

        self.initialized = True


    def preprocess(self, input_image):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        alignment_i_t = time.time()
        original_image = self.run_alignment(input_image)
        print(f'Alignment run in: {time.time() - alignment_i_t}')
        rgb_image = original_image.convert("RGB")

        transform_i_t = time.time()
        transformed_image = self.img_transforms(rgb_image)
        print(f'Transofrmation run in: {time.time() - transform_i_t}')

        preprocessed_image = transformed_image

        return preprocessed_image


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output

        ## PERFORM INFERENCE
        print(f'Performing inference')
        # result_images, result_latents = perform_inference_on_list(torch.stack(input_images), net)
        tic = time.time()

        # copied from run_on_batch when latent_msak = None
        print(f'model_input.shape: {model_input.shape}')
        result_image = self.net(model_input.unsqueeze(0).to("cuda").float(), randomize_noise=False) # probably need to unsqueeze to treat it as a list
        # result_image = run_on_batch(model_input, net, latent_mask)

        image_latents = self.encode_image(model_input)
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

        mean_latent = self.merge_latent_to_mean(image_latents)

        mean_image, _ = self.decode_image_latent(mean_latent.unsqueeze(0))
        mean_pil_image = tensor2im(mean_image)

        return mean_latent, image_latents, mean_pil_image

    def merge_latent_to_mean(self, input_latent):
        all_latents = torch.stack([torch.squeeze(self.mean_image_encoding), torch.squeeze(input_latent)], dim=0)
        mean_latent = torch.mean(all_latents, dim=0)
        print(f'mean_latent.shape: {mean_latent.shape}')
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

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        print(f'data: {data}')
        i_t = time.time()
        input_image = self.load_image(data)
        print(f'Images loaded in {time.time() - i_t} seconds')

        self.processed_input_image = self.preprocess(input_image)
        self.mean_image_encoding = self.load_mean_encoding()

        model_output = self.inference(self.processed_input_image)

        return self.postprocess(model_output)

    def run_alignment(self, image_path):
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        aligned_image = align_face(input_image=image_path, predictor=predictor)
        print("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image

    def load_image(self, full_image_path: str):
        print(f'Loading {full_image_path}')
        image_extension = full_image_path.split('.')[-1]
        if image_extension not in allowed_extensions:
            print('Extension not allowed')
            return None

        input_image = PIL.Image.open(full_image_path)

        return input_image
    
    def load_mean_encoding(self):
        mean_image_encoding = None

        if os.path.exists(self.mean_encodings_file_path):
            mean_image_encoding = torch.load(self.mean_encodings_file_path)
        else: # means its first image
            mean_image_encoding = self.encode_image(self.processed_input_image)

        return mean_image_encoding

    def save_mean_encoding(self, mean_encoding):
        torch.save(self.mean_image_encoding, self.mean_encodings_file_path)

        self.mean_image_encoding = mean_encoding

        return None

    def encode_image(self, image):
        input_image_tensor = image.unsqueeze(0)
        print(f'image_latents.shape: {input_image_tensor.shape}')
        image_latents = self.net.encoder(input_image_tensor.to("cuda").float())
        image_latents = image_latents + self.net.latent_avg.repeat(image_latents.shape[0], 1, 1)
        print(f'image_latents_single.shape: {image_latents.shape}')
        return image_latents

    def decode_image_latent(self, latent):
        images, result_latent = self.net.decoder([latent],
                                                input_is_latent=True,
                                                randomize_noise=True,
                                                return_latents=True)
        return images[0], result_latent

    def encode_images(self, images): # delete NOT USED, TO PERFORM ENCODING ON MULTIPLE IMAGES OR TENSORS THAT ARE ALREADY 4D ([1, 3, 256, 256])
        image_latents = self.net.encoder(images.to("cuda").float())
        image_latents = image_latents + self.net.latent_avg.repeat(image_latents.shape[0], 1, 1)
        print(f'images.shape: {images.shape}')
        print(f'image_latents.shape: {image_latents.shape}')
        ## Test
        input_images = torch.unbind(images)
        print(f'len(input_images): {len(input_images)}')
        print(f'input_images[0].shape: {input_images[0].shape}')
        image_latents = input_images[0].unsqueeze(0)
        print(f'image_latents.shape: {image_latents.shape}')
        image_latents_single = self.net.encoder(image_latents.to("cuda").float())
        image_latents_single = image_latents_single + self.net.latent_avg.repeat(image_latents.shape[0], 1, 1)
        print(f'image_latents_single.shape: {image_latents_single.shape}')
        return image_latents

    def generate_animation(self, start_encoding, end_encoding):
        print('Generating animation images')

        latents_list = [start_encoding.squeeze(), end_encoding.squeeze()]
        print(f'{latents_list[0].shape}')
        animation_frames = []
        for i, latent in enumerate(interpolate(latents_list=latents_list,duration_list=[1],interpolation_type="linear",loop=False, FPS=25)):
            i_t = time.time()
            frame_image, _ = self.decode_image_latent(latent)
            # print(f'Took {time.time() - i_t} to decode latent to an image')
            frame_i_t = time.time()
            frame_image = tensor2im(frame_image)
            animation_frames.append(frame_image)
            # print(f'Took {time.time() - frame_i_t} to tensor2im')

        return animation_frames

######################################################################################################3


allowed_extensions = ['png', 'jpg', 'jpeg']
EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": "pretrained_models/psp_ffhq_encode.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }
}


if __name__ == "__main__":
    output_path = '/home/carles/repos/matriu.id/ideal/image_enconding_tests/pixel2style2pixel_10_9_23'
    save_video = True

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_handler = ModelHandler()
    model_handler.initialize(None)

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2')
    natalia_image_path = '/home/carles/repos/matriu.id/ideal/image_enconding_tests/input_images/natalia2.png'
    iii_t = time.time()

    mean_latent, natalia_image_latents, mean_pil_image = model_handler.handle(natalia_image_path, None)
    model_handler.save_mean_encoding(mean_latent)

    print(f'Took {time.time() - iii_t} seconds to process one image')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2')

    del mean_latent
    # del image_latents
    del mean_pil_image

    torch.cuda.empty_cache()

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2')
    marina_image_path = '/home/carles/repos/matriu.id/ideal/image_enconding_tests/input_images/marina3.png'
    iii_t = time.time()

    mean_latent, marina_image_latents, mean_pil_image = model_handler.handle(marina_image_path, None)
    model_handler.save_mean_encoding(mean_latent)

    print(f'Took {time.time() - iii_t} seconds to process one image')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2')

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2')
    aimation_i_t = time.time()

    animation_frames = model_handler.generate_animation(marina_image_latents, natalia_image_latents)

    print(f'Animation made in {time.time() - aimation_i_t} seconds')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2')

    if save_video:
        print('Saving Video Frames')
        # Split the tensor along the first dimension (dimension 0)
        # Convert the output to a Python list
        frame_folder = os.path.join(output_path, 'frames')
        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)

        for i, frame_image in enumerate(animation_frames):
            frame_image_save_path = os.path.join(frame_folder,f'frame_{i:04}.png')
            i_t = time.time()
            frame_image.save(frame_image_save_path)
            print(f'Took {time.time() - i_t} to frame_image.save(frame_image_save_path)')
