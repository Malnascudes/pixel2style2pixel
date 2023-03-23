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

def run_alignment(image_path):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image

def run_on_batch(inputs, net, latent_mask=None):
    if latent_mask is None:
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

        if experiment_type not in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
            original_image = run_alignment(full_image_path)
            rgb_image = original_image.convert("RGB")
            transformed_image = img_transforms(rgb_image)
            input_image = transformed_image
        else:
            original_image = Image.open(full_image_path)
            if opts.label_nc == 0:
                original_image = original_image.convert("RGB")
            else:
                original_image = original_image.convert("L")

                original_image.resize((256, 256))

            input_image = original_image

        image_list.append(input_image)

    return image_list

def perform_inference(input_image, net):
    if experiment_type in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
        latent_mask = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    else:
        latent_mask = None

    with torch.no_grad():
        tic = time.time()
        result_image, result_latents = run_on_batch(input_image.unsqueeze(0), net, latent_mask)
        toc = time.time()
        print('Inference took {:.4f} seconds.'.format(toc - tic))

    return result_image, result_latents

def perform_inference_on_list(input_image_list, net):
    if experiment_type in ["celebs_sketch_to_face", "celebs_seg_to_face"]:
        latent_mask = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    else:
        latent_mask = None

    with torch.no_grad():
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
    },
    "ffhq_frontalize": {
        "model_path": "pretrained_models/psp_ffhq_frontalization.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "celebs_sketch_to_face": {
        "model_path": "pretrained_models/psp_celebs_sketch_to_face.pt",
        "image_path": "notebooks/images/input_sketch.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
    },
    "celebs_seg_to_face": {
        "model_path": "pretrained_models/psp_celebs_seg_to_face.pt",
        "image_path": "notebooks/images/input_mask.png",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.ToOneHot(n_classes=19),
            transforms.ToTensor()])
    },
    "celebs_super_resolution": {
        "model_path": "pretrained_models/psp_celebs_super_resolution.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            augmentations.BilinearResize(factors=[16]),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
    "toonify": {
        "model_path": "pretrained_models/psp_ffhq_toonify.pt",
        "image_path": "notebooks/images/input_img.jpg",
        "transform": transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    },
}


if __name__ == "__main__":
    experiment_type = 'ffhq_encode'
    images_path = '/home/carles/repos/matriu.id/ideal/image_enconding/input_images'
    output_path = '/home/carles/repos/matriu.id/ideal/image_enconding/test_average_sqtylegan_3'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]


    model_path = EXPERIMENT_ARGS['model_path']
    print(f'Loading model from {model_path}')
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
    print('Model successfully loaded!')

    img_transforms = EXPERIMENT_ARGS['transform']
    input_images = load_images(images_path, img_transforms)

    ## PERFORM INFERENCE
    print(f'Performing inference')
    result_images, result_latents = perform_inference_on_list(torch.stack(input_images), net)
    print(f'type(result_latents): {type(result_latents)}')
    print(f'result_latents.shape: {result_latents.shape}')

    for og_image, result_img in zip(input_images, result_images):
        input_vis_image = log_input_image(og_image, opts)
        output_image = tensor2im(result_img)
        input_latents.append(result_latents.squeeze(0))

        if experiment_type == "celebs_super_resolution":
            res = np.concatenate([
                                # np.array(input_image.resize((256, 256))),
                                np.array(input_vis_image.resize((256, 256))),
                                np.array(output_image.resize((256, 256)))], axis=1)
        else:
            res = np.concatenate([np.array(input_vis_image.resize((256, 256))),
                                np.array(output_image.resize((256, 256)))], axis=1)

        res_image = Image.fromarray(res)
        res_image.show()

    mean_latent = torch.mean(result_latents, dim=0)
    mean_image, _ = decode_image_latent(net, mean_latent.unsqueeze(0))
    mean_pil_image = tensor2im(mean_image)
    mean_image_save_path = os.path.join(output_path, 'mean_image.png')
    mean_pil_image.save(mean_image_save_path)
    mean_pil_image.resize((256,256)).show()

