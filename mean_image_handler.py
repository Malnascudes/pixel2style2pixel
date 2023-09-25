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
        self.mean_image_encoding = self.load_mean_encoding()

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

    def load_mean_encoding(self):
        mean_image_encoding = None

        if os.path.exists(self.mean_encodings_file_path):
            print(f'Loading mean encoding from {self.mean_encodings_file_path}')
            mean_image_encoding = torch.load(self.mean_encodings_file_path)
        else: # means its first image
            print(f'No encoding found in {self.mean_encodings_file_path}. Using input image as mean.')
            mean_image_encoding = self.encode_image(self.processed_input_image)

        return mean_image_encoding

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
        image_encoding = self.encode_image(model_input)
        output_image, result_latent = self.decode_image_latent(image_encoding)
        output__pil_image = tensor2im(output_image)
    
        model_output = output__pil_image, result_latent

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

if __name__ == "__main__":
    model_handler = ModelHandler()
    model_handler.initialize(None)

    print('Model initialized')

    image_path = '/home/carles/repos/matriu.id/ideal/Datasets/sorolla-test-faces/minimum-subset/CFD-AM-229-224-N.jpg'
    i_t = time.time()

    output_image, result_latent = model_handler.handle(image_path, None)
    print(f'Image processed in {time.time() - i_t} seconds')

    output_path = 'test.tiff'
    output_image.save(output_path)
    print(result_latent.shape)
