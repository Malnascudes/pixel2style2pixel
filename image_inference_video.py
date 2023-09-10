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

def run_alignment(image_path):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

# Method copied from predict.py lin 165
def run_on_batch(inputs, net, latent_mask=None):
    if latent_mask is None:
        # Part of the code that is executed
        # inputs.to("cuda").float() to 
        result_batch = net(inputs.to("cuda").float(), randomize_noise=False)
        return result_batch
    else:
        result_batch = []
        for image_idx, input_image in enumerate(inputs):
            # get latent vector to inject into our input image
            vec_to_inject = np.random.randn(1, 512).astype('float32')
            _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"),
                                      input_code=True,
                                      return_latents=True)
            # get output image with injected style vector
            res = net(input_image.unsqueeze(0).to("cuda").float(),
                      latent_mask=latent_mask,
                      inject_latent=latent_to_inject)
            result_batch.append(res)
        result_batch = torch.cat(result_batch, dim=0)

        return result_batch


def load_images(image_folder: str, img_transforms):
    image_list = []
    for image_path in os.listdir(image_folder):
        full_image_path = os.path.join(image_folder, image_path)
        print(f'Loading {full_image_path}')
        image_extension = full_image_path.split('.')[-1]
        if image_extension not in allowed_extensions:
            continue

        alignment_i_t = time.time()
        original_image = run_alignment(full_image_path)
        print(f'Alignment run in: {time.time() - alignment_i_t}')
        rgb_image = original_image.convert("RGB")
        transform_i_t = time.time()
        transformed_image = img_transforms(rgb_image)
        print(f'Transofmration run in: {time.time() - transform_i_t}')
        input_image = transformed_image

        image_list.append(input_image)

    return image_list


def perform_inference_on_list(input_image_list, net):
    with torch.no_grad():
        latent_mask = None
        tic = time.time()
        result_image = run_on_batch(input_image_list, net, latent_mask)
        result_latents = encode_images(net, input_image_list)
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    return result_image, result_latents

def encode_images(model, images):
    image_latents = model.encoder(images.to("cuda").float())
    image_latents = image_latents + model.latent_avg.repeat(image_latents.shape[0], 1, 1)
    return image_latents

def decode_image_latent(model, latent):
    images, result_latent = model.decoder([latent],
		                                     input_is_latent=True,
		                                     randomize_noise=True,
		                                     return_latents=True)
    return images[0], result_latent

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
    experiment_type = 'ffhq_encode'
    images_path = '/home/carles/repos/matriu.id/ideal/image_enconding_tests/input_images'
    output_path = '/home/carles/repos/matriu.id/ideal/image_enconding_tests/pixel2style2pixel_25_8_23'
    show_images = False
    video = True

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]

    model_path = EXPERIMENT_ARGS['model_path']
    print(f'Loading model from {model_path}')
    i_t = time.time()
    ckpt = torch.load(model_path, map_location='cpu')
    print(f'Model loaded from {model_path}')

    opts = ckpt['opts']
    pprint.pprint(opts)
    # update the training options
    opts['checkpoint_path'] = model_path
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024

    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    print(f'Model successfully loaded in {time.time() - i_t} seconds!')

    img_transforms = EXPERIMENT_ARGS['transform']

    i_t = time.time()
    input_images = load_images(images_path, img_transforms)
    print(f'Images loaded in {time.time() - i_t} seconds')

    ## PERFORM INFERENCE
    print(f'Performing inference')
    result_images, result_latents = perform_inference_on_list(torch.stack(input_images), net)

    mean_latent = torch.mean(result_latents, dim=0)
    print(f'mean_latent.shape: {mean_latent.shape}')
    i_t = time.time()
    mean_image, _ = decode_image_latent(net, mean_latent.unsqueeze(0))
    mean_pil_image = tensor2im(mean_image)
    mean_image_save_path = os.path.join(output_path, 'mean_image.png')
    mean_pil_image.save(mean_image_save_path)


    if video:
        print('Saving Video Frames')
        # Split the tensor along the first dimension (dimension 0)
        # Convert the output to a Python list
        frame_folder = os.path.join(output_path, 'frames')
        if not os.path.exists(frame_folder):
            os.makedirs(frame_folder)

        latents_list = [tensor.squeeze() for tensor in torch.split(result_latents, 1, dim=0)]
        print(f'{latents_list[0].shape}')
        for i, latent in enumerate(interpolate(latents_list=latents_list,duration_list=[1],interpolation_type="linear",loop=False, FPS=25)):
            i_t = time.time()
            frame_image, _ = decode_image_latent(net, latent)
            print(f'Took {time.time() - i_t} to decode latent to an image')
            frame_i_t = time.time()
            frame_image = tensor2im(frame_image)
            print(f'Took {time.time() - frame_i_t} to tensor2im')
            frame_image_save_path = os.path.join(frame_folder,f'frame_{i:04}.png')
            i_t = time.time()
            frame_image.save(frame_image_save_path)
            print(f'Took {time.time() - i_t} to frame_image.save(frame_image_save_path)')

    if show_images:
        mean_pil_image.resize((256,256)).show()
        for og_image, result_img in zip(input_images, result_images):
            input_vis_image = log_input_image(og_image, opts)
            output_image = tensor2im(result_img)
            res = np.concatenate([np.array(input_vis_image.resize((256, 256))),
                                np.array(output_image.resize((256, 256)))], axis=1)

            res_image = Image.fromarray(res)
            res_image.show()
