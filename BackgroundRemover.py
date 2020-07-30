import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

import torch
import torchvision
from torchvision import transforms

# FBA_Matting scripts
if not 'FBA_Matting' in sys.path:
    sys.path.append(os.path.join(os.path.dirname('__file__'), 'FBA_Matting'))
from demo import np_to_torch, pred, scale_input
from dataloader import read_image, read_trimap
from networks.models import build_model


class BackgroundRemover(object):
    def __init__(self, input_image, kernel_size=(5, 5), iteration=3):
        '''
        constructor
        '''
        self.kernel_size = kernel_size
        self.iteration = iteration

        self._initialize(input_image)

    def _initialize(self, input_image):

        self._load_image(input_image)
        self._make_output_dirs()
        self._prepare_trimap_model()

        image_name = os.path.basename(input_image)
        self.image_path = input_image
        self.mask_image_name = os.path.join('./mask/', image_name)
        self.trimap_image_name = os.path.join('./trimaps/', image_name)
        self.output_file = os.path.join(
            './output/', image_name.split('.')[0] + '_fg.png')
        self.output_mask_file = os.path.join(
            './output/', image_name.split('.')[0] + '_fg_mask.png')

    def _load_image(self, input_image):
        self.img = cv2.imread(input_image)
        self.img = self.img[..., ::-1]  # BGR->RGB
        self.h, self.w, _ = self.img.shape
        self.img = cv2.resize(self.img, (320, 320))

    def _make_output_dirs(self):
        os.makedirs('./mask/', exist_ok=True)
        os.makedirs('./trimaps/', exist_ok=True)
        os.makedirs('./output/', exist_ok=True)

    def _prepare_trimap_model(self):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.trimap_model = torchvision.models.segmentation.deeplabv3_resnet101(
            pretrained=True)
        self.trimap_model = self.trimap_model.to(self.device)
        self.trimap_model.eval()

    def make_trimap_image(self):
        self._create_mask_image()
        self._create_trimap_image()

    def _create_mask_image(self):
        print('creating mask image....')
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(self.img)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.trimap_model(input_batch)['out'][0]
        output = output.argmax(0)
        self.mask = output.byte().cpu().numpy()
        self.mask = cv2.resize(self.mask, (self.w, self.h))
        self.img = cv2.resize(self.img, (self.w, self.h))
        cv2.imwrite(self.mask_image_name, self.mask)

    def _create_trimap_image(self):
        print('creating trimap image....')

        def binarize_image(image):
            image_bi = image.copy()
            image_bi[np.where(image_bi > 0)] = 255
            return image_bi

        def gen_trimap(mask, k_size=(5, 5), iteration=1):
            kernel = np.ones(k_size, np.uint8)
            eroded = cv2.erode(mask, kernel, iterations=iteration)
            dilated = cv2.dilate(mask, kernel, iterations=iteration)
            eroded_bi = binarize_image(eroded)
            dilated_bi = binarize_image(dilated)
            trimap = np.full(mask.shape, 128)
            trimap[eroded_bi == 255] = 255
            trimap[dilated_bi == 0] = 0
            return trimap
        self.trimap = gen_trimap(
            self.mask, k_size=self.kernel_size, iteration=self.iteration)
        cv2.imwrite(self.trimap_image_name, self.trimap)

    def run_FBA_Matting(self):
        self._check_matting_model()
        print('start FBA_Matting...')
        self._build_matting_model()
        self._create_output()

    def _check_matting_model(self):
        if not os.path.exists('./FBA_Matting/FBA.pth'):
            print('There is no model file.¥nYou should download it from https://drive.google.com/uc?id=1T_oiKDE_biWf2kqexMEN7ObWqtXAzbB1 and place it to ./FBA_Matting/FBA.pth')
            raise Exception(
                'There is no model file. You should download it from https://drive.google.com/uc?id=1T_oiKDE_biWf2kqexMEN7ObWqtXAzbB1 and place it to ./FBA_Matting/FBA.pth')

    def _build_matting_model(self):
        class Args:
            encoder = 'resnet50_GN_WS'
            decoder = 'fba_decoder'
            weights = './FBA_Matting/FBA.pth'
        args = Args()
        self.model = build_model(args)

    def _create_output(self):
        image = read_image(self.image_path)
        trimap = read_trimap(self.trimap_image_name)

        print('start prediction....')
        fg, bg, alpha = pred(image, trimap, self.model)

        out_fg_img = cv2.cvtColor(
            fg*alpha[:, :, None] * 255.0, cv2.COLOR_RGB2BGR)
        cv2.imwrite(self.output_file, out_fg_img)
        cv2.imwrite(self.output_mask_file, alpha * 255.0)
        print('finish prediction....')


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_image",
                        help="input .png image path")
    parser.add_argument("--kernel_size", type=tuple, default=(5, 5))
    parser.add_argument("--iteration", type=int, default=3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    try:
        print('checking arguments...')
        args = get_args()
        print('start matting...')
        remover = BackgroundRemover(
            args.input_image, args.kernel_size, args.iteration)

        remover.make_trimap_image()
        remover.run_FBA_Matting()
    except Exception as e:
        print('matting failed: ', e)
        print(e.__traceback__)
