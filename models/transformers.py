import numpy as np
import torch
import kornia
import torch.nn.functional as F


def Rotate(x, alphas, step):
    b, c, h, w = x.shape
    start, end = alphas
    x = CopyN(x, step)
    angle = torch.linspace(start, end, step, device=x.device).unsqueeze(0).repeat(b, 1).view(b * step)
    center = torch.zeros((b * step, 2), device=x.device)
    center[:, 0], center[:, 1] = w / 2, h / 2
    scale = torch.ones(b * step, device=x.device)
    M = kornia.get_rotation_matrix2d(center, angle, scale)
    x_hat = kornia.warp_affine(x, M, dsize=(h, w))
    return x_hat


def TranslateX(x, alphas, step):
    b, c, h, w = x.shape
    start, end = alphas
    x = CopyN(x, step)
    translate = torch.zeros(b * step, 2, device=x.device)
    translate[:, 0] = torch.linspace(start, end, step, device=x.device).unsqueeze(0).repeat(b, 1).view(b * step) * w
    return kornia.geometry.translate(x, translate)


def TranslateY(x, alphas, step):
    b, c, h, w = x.shape
    start, end = alphas
    x = CopyN(x, step)
    translate = torch.zeros(b * step, 2, device=x.device)
    translate[:, 1] = torch.linspace(start, end, step, device=x.device).unsqueeze(0).repeat(b, 1).view(b * step) * h
    return kornia.geometry.translate(x, translate)


def ShearX(x, alphas, step):
    b, c, h, w = x.shape
    start, end = alphas
    x = CopyN(x, step)
    translate = torch.zeros(b * step, 2, device=x.device)
    translate[:, 0] = torch.linspace(start, end, step, device=x.device).unsqueeze(0).repeat(b, 1).view(b * step)
    return kornia.geometry.shear(x, translate)


def ShearY(x, alphas, step):
    b, c, h, w = x.shape
    start, end = alphas
    x = CopyN(x, step)
    translate = torch.zeros(b * step, 2, device=x.device)
    translate[:, 1] = torch.linspace(start, end, step, device=x.device).unsqueeze(0).repeat(b, 1).view(b * step)
    return kornia.geometry.shear(x, translate)


def Gray(x):
    return x.mean(1).unsqueeze(1).repeat(1, 3, 1, 1)


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def Invert(x):
    return flip(x, 1)


def Brightness(x, alpha):
    # alpha (0.0, 0.5)
    return kornia.adjust_brightness(x, alpha)


def Contrast(x, alpha):
    # alpha (1.0, 2.0)
    return kornia.adjust_contrast(x, alpha)


def Gamma(x, alpha, beta):
    # alpha (1.0, 3.0), beta (1.0, 3.0)
    return kornia.adjust_gamma(x, gamma=alpha, gain=beta)


def Saturation(x, alpha):
    # alpha (0.0, 1.0)
    return kornia.adjust_saturation(x, alpha)


def Hue(x, alpha):
    # alpha (0.0, 1.0)
    return kornia.adjust_hue(x, alpha)


def Sepia(x):
    x = Gray(x)
    x[:, 0], x[:, 1], x[:, 2] = x[:, 0] * (240 / 255), x[:, 1] * (200 / 255), x[:, 2] * (145 / 255)
    return x


def Blur(x):
    return kornia.gaussian_blur2d(x, (3, 3), (1.0, 1.0))


def Dilation(x):
    return F.max_pool2d(x, (3, 3), padding=1, stride=1)


def Erosion(x):
    return - F.max_pool2d(-x, (3, 3), padding=1, stride=1)


def Red(x):
    x[:, 1] = x[:, 1] * 0.0
    x[:, 2] = x[:, 2] * 0.0
    return x


def Green(x):
    x[:, 0] = x[:, 0] * 0.0
    x[:, 2] = x[:, 2] * 0.0
    return x


def Blue(x):
    x[:, 0] = x[:, 0] * 0.0
    x[:, 1] = x[:, 1] * 0.0
    return x


def Cutout(x, step, mask_size=(16, 16), fill_value=0.5):
    h, w = x.shape[2:]
    # start
    ymin1 = np.random.randint(-int(mask_size[0] / 2), h - int(mask_size[0] / 2))
    xmin1 = np.random.randint(-int(mask_size[1] / 2), w - int(mask_size[1] / 2))
    ymax1 = ymin1 + mask_size[0]
    xmax1 = xmin1 + mask_size[1]
    ymin1, xmin1 = max(0, ymin1), max(0, xmin1)
    ymax1, xmax1 = min(h - 1, ymax1), min(w - 1, xmax1)
    # end
    ymin2 = np.random.randint(-int(mask_size[0] / 2), h - int(mask_size[0] / 2))
    xmin2 = np.random.randint(-int(mask_size[1] / 2), w - int(mask_size[1] / 2))
    ymax2 = ymin2 + mask_size[0]
    xmax2 = xmin2 + mask_size[1]
    ymin2, xmin2 = max(0, ymin2), max(0, xmin2)
    ymax2, xmax2 = min(h - 1, ymax2), min(w - 1, xmax2)
    x = CopyN(x, step).view(x.shape[0], step, x.shape[1], x.shape[2], x.shape[3])
    ymins, ymaxs = np.linspace(ymin1, ymin2, step), np.linspace(ymax1, ymax2, step)
    xmins, xmaxs = np.linspace(xmin1, xmin2, step), np.linspace(xmax1, xmax2, step)
    for i, (ymin, ymax, xmin, xmax) in enumerate(zip(ymins, ymaxs, xmins, xmaxs)):
        x[:, i, :, int(ymin):int(ymax), int(xmin):int(xmax)] = fill_value
    return x.view(-1, x.shape[2], x.shape[3], x.shape[4])


def SamplePairing(x, step, alpha=(0.0, 0.4)):
    x1 = CopyN(x, step).view(x.shape[0], step, x.shape[1], x.shape[2], x.shape[3])
    x2 = Flip(CopyN(x, step).view(x.shape[0], step, x.shape[1], x.shape[2], x.shape[3]), 0)

    alpha_end = alpha[0] + torch.rand(1, device=x.device) * (alpha[1] - alpha[0])
    a = torch.linspace(0.0, alpha_end[0], step, device=x.device)
    a = a.unsqueeze(0).repeat(x1.shape[0], 1).view(x1.shape[0], step, 1, 1, 1)
    return ((1.0 - a) * x1 + a * x2).view(x.shape[0] * step, x.shape[1], x.shape[2], x.shape[3])


def Flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def CopyN(x, n):
    b, c, h, w = x.shape
    return x.unsqueeze(1).repeat(1, n, 1, 1, 1).view(b * n, c, h, w)


def LinearSmoothing(x1, x2, step):
    b, c, h, w = x1.shape
    x1 = CopyN(x1, step)
    x2 = CopyN(x2, step)
    alpha = torch.linspace(0.0, 1.0, step, device=x1.device).unsqueeze(0).repeat(b, 1).view(b * step, 1, 1, 1)
    return alpha * x1 + (1.0 - alpha) * x2


if __name__ == '__main__':
    import os
    import random
    from PIL import Image
    from skimage import io

    def make_dir_one(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def make_dir(path):
        separated_path = path.split('/')
        tmp_path = ''
        for directory in separated_path:
            tmp_path = tmp_path + directory + '/'
            if directory == '.':
                continue
            make_dir_one(tmp_path)
        return True

    def pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    img_path = '/home/nutszebra/Downloads/ccc.jpg'
    save_path = '/home/nutszebra/Downloads/gpu_transform'
    make_dir(save_path)
    x = torch.Tensor(np.asarray(pil_loader(img_path))).unsqueeze(0).repeat(2, 1, 1, 1).permute(0, 3, 1, 2)
    # rotate (-180, 180)
    for i, img in enumerate(Rotate(x, (-45, 45), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/rotate_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # translate x
    for i, img in enumerate(TranslateX(x, (-0.5, 0.5), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/translatex_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # translate y
    for i, img in enumerate(TranslateY(x, (-0.5, 0.5), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/translatey_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # shear x
    for i, img in enumerate(ShearX(x, (-1.0, 1.0), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/shearx_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # shear y
    for i, img in enumerate(ShearY(x, (-1.0, 1.0), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/sheary_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # gray
    for i, img in enumerate(LinearSmoothing(x, (Gray(x / 255.) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/gray_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # invert
    for i, img in enumerate(LinearSmoothing(x, (Invert(x / 255.) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/invert_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # brightness (-0.5, 0.5)
    for i, img in enumerate(LinearSmoothing(x, (Brightness(x / 255., 0.5) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/brightness_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # contrast (0.0, 2.0)
    for i, img in enumerate(LinearSmoothing(x, (Contrast(x / 255., 0.0) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/contrast_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # gamma (1.0, 3.0), gain (1.0, 3.0)
    for i, img in enumerate(LinearSmoothing(x, (Gamma(x / 255., 3.0, 3.0) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/gamma_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # saturation (0.0, 1.0)
    for i, img in enumerate(LinearSmoothing(x, (Saturation(x / 255., 0.0) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/saturation_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # hue (0.0, 1.0)
    for i, img in enumerate(LinearSmoothing(x, (Hue(x / 255., 1.0) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/hue_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # sepia
    for i, img in enumerate(LinearSmoothing(x, (Sepia(x / 255.) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/sepia_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # blur
    for i, img in enumerate(LinearSmoothing(x, (Blur(x / 255.) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/blur_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # dilation
    for i, img in enumerate(LinearSmoothing(x, (Dilation(x / 255.) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/dilation_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # erosion
    for i, img in enumerate(LinearSmoothing(x, (Erosion(x / 255.) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/erosion_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # Red
    for i, img in enumerate(LinearSmoothing(x, (Red(x / 255.) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/red_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # Green
    for i, img in enumerate(LinearSmoothing(x, (Green(x / 255.) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/green_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # Blue
    for i, img in enumerate(LinearSmoothing(x, (Blue(x / 255.) * 255), 10).permute(0, 2, 3, 1)):
        io.imsave('{}/blue_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # Cutout
    for i, img in enumerate((Cutout(x / 255., 10, mask_size=(50, 50)) * 255).permute(0, 2, 3, 1)):
        io.imsave('{}/cutout_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
    # SamplePairing
    x_s = x[:]
    x_s[1] = x_s[1] * 0
    for i, img in enumerate((SamplePairing(x_s / 255., 10, alpha=(0.8, 0.9)) * 255).permute(0, 2, 3, 1)):
        io.imsave('{}/samplepairing_{}.jpg'.format(save_path, i), img.data.numpy().astype(np.uint8))
