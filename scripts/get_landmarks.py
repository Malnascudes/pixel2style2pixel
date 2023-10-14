from mean_image_handler import ModelHandler
from utils.common import tensor2im
import os
from pathlib import Path
import tqdm

if __name__ == "__main__":
    allowed_extensions = ['png', 'jpg', 'jpeg']
    
    model_handler = ModelHandler()
    model_handler.initialize(None)

    preprocess_images = True

    print('Model initialized')

    image_folder_path = '/home/carles/repos/matriu.id/ideal/Datasets/sorolla-test-faces/minimum-subset'
    output_folder = '/home/carles/repos/matriu.id/ideal/image_enconding_tests/landmarks/cfd_aligned'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in tqdm.tqdm(os.listdir(image_folder_path), desc="Computing landmarks for images"):
        file_extension = file.split('.')[-1]
        if file_extension not in allowed_extensions:
            continue

        image_path = Path(image_folder_path, file)

        input_image = model_handler.load_image(str(image_path))

        if preprocess_images:
            processed_image = model_handler.preprocess(input_image)
        else:
            processed_image = input_image

        # Save original image before it gets painted on
        image_save_path = Path(output_folder, file)
        processed_image.save(str(image_save_path))

        face_landmarks = model_handler.get_face_landmarks(processed_image)
        image_with_landmarks = model_handler.plot_landmarks_over_image(processed_image, face_landmarks)

        image_save_name = file.replace('.', '_landmarks.')
        image_save_path = Path(output_folder, image_save_name)
        image_with_landmarks.save(str(image_save_path))


        landmarks_out_file = file.replace(file_extension, 'txt')
        landmarks_out_path = Path(output_folder, landmarks_out_file)
        model_handler.save_landmarks(face_landmarks, landmarks_out_path)

        landmarks_out_file = file.replace(file_extension, 'json')
        landmarks_out_path = Path(output_folder, landmarks_out_file)
        model_handler.save_landmarks_structured(face_landmarks, str(landmarks_out_path))