import os
from argparse import Namespace

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp
from mean_image_handler import ModelHandler


def run():
    test_opts = TestOptions().parse()

    if test_opts.resize_factors is not None:
        factors = test_opts.resize_factors.split(',')
        assert len(factors) == 1, "When running inference, please provide a single downsampling factor!"
        mixed_path_results = os.path.join(test_opts.exp_dir, 'style_mixing',
                                          'downsampling_{}'.format(test_opts.resize_factors))
    else:
        mixed_path_results = os.path.join(test_opts.exp_dir, 'style_mixing')
    os.makedirs(mixed_path_results, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    latent_mask = [int(l) for l in opts.latent_mask.split(",")]
    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_batch = input_batch.cuda()
            for image_idx, input_image in enumerate(input_batch):
                # generate random vectors to inject into input image
                vecs_to_inject = np.random.randn(opts.n_outputs_to_generate, 512).astype('float32')
                multi_modal_outputs = []
                for vec_to_inject in vecs_to_inject:
                    cur_vec = torch.from_numpy(vec_to_inject).unsqueeze(0).to("cuda")
                    # get latent vector to inject into our input image
                    _, latent_to_inject = net(cur_vec,
                                              input_code=True,
                                              return_latents=True)
                    # get output image with injected style vector
                    res = net(input_image.unsqueeze(0).to("cuda").float(),
                              latent_mask=latent_mask,
                              inject_latent=latent_to_inject,
                              alpha=opts.mix_alpha,
                              resize=opts.resize_outputs)
                    multi_modal_outputs.append(res[0])

                # visualize multi modal outputs
                input_im_path = dataset.paths[global_i]
                image = input_batch[image_idx]
                input_image = log_input_image(image, opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                res = np.array(input_image.resize(resize_amount))
                for output in multi_modal_outputs:
                    output = tensor2im(output)
                    res = np.concatenate([res, np.array(output.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(os.path.join(mixed_path_results, os.path.basename(input_im_path)))
                global_i += 1

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def style_mix(image1_encoding, image2_encoding, latent_mask):
    for i in latent_mask:
        image1_encoding[:, i] = image2_encoding[:, i]

    return image1_encoding

if __name__ == '__main__':
    model_handler = ModelHandler()
    model_handler.initialize(None)
    allowed_extensions = ['png', 'jpg', 'jpeg']
    print('Model initialized')

    latent_mask = [12,13,14,15]
    latent_mask = [10,12,13]
    latent_mask = [10,11,12,13,14]
    latent_mask_str = "-".join([str(l) for l in latent_mask])
    # input_folder = '/home/carles/repos/matriu.id/ideal/Datasets/sorolla-test-faces/minimum-subset'
    # input_folder = '/home/carles/repos/matriu.id/ideal/Datasets/pretty-faces/MiamiModels_crops'
    input_folder = '/home/carles/repos/matriu.id/ideal/Datasets/pretty-faces/MiamiModels_crops/straight_faces_subset'
    output_folder = f"/home/carles/repos/matriu.id/ideal/image_enconding_tests/pixel2style2pixel_15_10_23_style_mixing_demo/various_styles/test1-latent-mask-{latent_mask_str}"
    style_folder_path = "/home/carles/repos/matriu.id/ideal/Datasets/styles/fantasy/TML_demo/various_styles"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for input_image in os.listdir(input_folder):
        file_extension = input_image.split('.')[-1]
        if file_extension not in allowed_extensions:
            continue

        image1_path = f'{input_folder}/{input_image}'
        output_image, result_latent, input_image1_encoding = model_handler.handle(image1_path, None)
        # face_input_image = tensor2im(output_image)

        face_input_image = output_image
        face_image_name = image1_path.split('/')[-1]


        for style_image_name in tqdm(os.listdir(style_folder_path)):
            # style_image_path = '/home/carles/repos/matriu.id/ideal/Datasets/styles/fantasy/Peter_Mohrbacher-0018.jpg'
            file_extension = input_image.split('.')[-1]
            if file_extension not in allowed_extensions:
                continue
            style_image_path = f'{style_folder_path}/{style_image_name}'

            output_image2, result_latent, input_image2_encoding = model_handler.handle(style_image_path, None)
            # style_image = tensor2im(output_image2)
            style_image = output_image2

            output_encoding = style_mix(input_image1_encoding, input_image2_encoding, latent_mask)

            output_image, result_latent = model_handler.decode_image_latent(output_encoding)
            output_pil_image = tensor2im(output_image)

            grid = image_grid(
                [face_input_image, style_image, output_pil_image],
                rows=1,
                cols=3,
            )

            grid_image_name = f'grid-{face_image_name}-{style_image_name}.png'
            grid_image_path = f'{output_folder}/{grid_image_name}'
            grid.save(grid_image_path)

            output_image_name = f'{face_image_name}-{style_image_name}.png'
            output_image_path = f'{output_folder}/{output_image_name}'
            output_pil_image.save(output_image_path)

            del output_image
            del result_latent
            del output_encoding
            del output_image2
            del input_image2_encoding
            torch.cuda.empty_cache()

        face_image_path = f'{output_folder}/{face_image_name}'
        face_input_image.save(face_image_path)
        face_input_image
