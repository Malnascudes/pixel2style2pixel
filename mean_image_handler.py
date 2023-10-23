from argparse import Namespace
import pprint
import torch
import torchvision.transforms as transforms
import os
from models.psp import pSp

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
        self.processed_input_image = self.preprocess(data)

        model_output = self.inference(self.processed_input_image)

        return self.postprocess(model_output)

    def preprocess(self, input_image):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        return input_image


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
