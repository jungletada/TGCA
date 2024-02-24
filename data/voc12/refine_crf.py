import numpy as np


def _crf_with_alpha(cam_dict, alpha, orig_img):
    from tool.imutils import crf_inference
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])
    
    n_crf_al = dict()
    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al


def refine_crf_cam(cam_dict, original_image, low_alpha=1, high_alpha=12,):
    label = {}
    crf_dict = {'low': low_alpha, 'high': high_alpha}
    for alpha, value in crf_dict.items():
        image_ = original_image.astype(np.uint8).copy(order='C')
        label[alpha] = _crf_with_alpha(cam_dict, value, image_)
    
    return label['low'], label['high']