import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


def crf_inference(img, probs, t=10, labels=21):
    """
    Perform CRF inference on the given image and probabilities.

    Parameters:
    img (numpy.ndarray): The input image.
    probs (numpy.ndarray): The probabilities for each label.
    t (int): Number of iterations for inference (default is 5).
    labels (int): Number of labels (default is 21).

    Returns:
        numpy.ndarray: The CRF refined probabilities reshaped to (labels, height, width).
    """
    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=1, compat=1)
    d.addPairwiseBilateral(sxy=121, srgb=3, rgbim=np.copy(img), compat=3)
    Q = d.inference(t)
    return np.array(Q).reshape((n_labels, h, w))


def _crf_with_alpha(cam_dict, alpha, original_img):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    # bgcam_score = np.pad(v, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=alpha)
    crf_score = crf_inference(
        original_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = dict()
    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al


def refine_crf_cam(cam_dict, original_image, low_alpha=1, high_alpha=3):
    """
    Refine class activation maps (CAM) using CRF.

    Parameters:
    cam_dict (dict): Dictionary of class activation maps.
    original_image (numpy.ndarray): The original image to refine the CAMs on.
    low_alpha (float): Alpha value for low refinement (default is 1).
    high_alpha (float): Alpha value for high refinement (default is 3).

    Returns:
    tuple: Refined CAMs for low and high alpha values.
    """
    label = {}
    crf_dict = {'low': low_alpha, 'high': high_alpha}
    for alpha, value in crf_dict.items():
        image_ = original_image.astype(np.uint8).copy(order='C')
        label[alpha] = _crf_with_alpha(cam_dict, value, image_)
    
    return label['low'], label['high']