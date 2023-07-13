import gdown
import torch

from .yolo_strategy import YoloStrategy

from boxmot.utils import WEIGHTS
from boxmot.utils.checks import TestRequirements

tr = TestRequirements()
try:
    import yolox  # for linear_assignment
except (ImportError, AssertionError, AttributeError):
    tr.check_packages(('yolox==0.3.0',))  # install

from yolox.utils import postprocess
from yolox.exp import get_exp

from ultralytics.yolo.engine.results import Results


YOLOX_ZOO = {
    'yolox_n': 'https://drive.google.com/uc?id=1AoN2AxzVwOLM0gJ15bcwqZUpFjlDV1dX',
    'yolox_s': 'https://drive.google.com/uc?id=1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj',
    'yolox_m': 'https://drive.google.com/uc?id=11Zb0NN_Uu7JwUd9e6Nk8o2_EUfxWqsun',
    'yolox_l': 'https://drive.google.com/uc?id=1XwfUuCBF4IgWBWK2H7oOhQgEj9Mrb3rz',
    'yolox_x': 'https://drive.google.com/uc?id=1P4mY0Yyd3PPTybgZkjMYhFri88nTmJX5',
}


class YoloXStrategy(YoloStrategy):
    def __init__(self, model, device, args):

        self.args = args
        self.has_run = False

        model = str(model)
        if model == 'yolox_n':
            exp = get_exp(None, 'yolox_nano')
        else:
            exp = get_exp(None, model)
        exp.num_classes = 1  # bytetrack yolox models

        self.model = exp.get_model()
        self.model.eval()

        gdown.download(
            url=YOLOX_ZOO[model],
            output=str(WEIGHTS / (model + '.pth')),
            quiet=False
        )

        ckpt = torch.load(
            str(WEIGHTS / (model + '.pth')),
            map_location=torch.device('cpu')
        )
        
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device)

    def inference(self, im):
        preds = self.model(im)
        return preds

    def postprocess(self, path, preds, im, im0s, predictor):

        if not self.has_run:
            self.w_r, self.h_r = self.get_scaling_factors(im, im0s)
            self.has_run = True

        preds = postprocess(
            preds, 1, conf_thre=self.args.conf,
            nms_thre=0.45, class_agnostic=True
        )[0]

        if preds is None:
            preds = torch.empty(0, 6)
        else:
            # (x, y, x, y, conf, obj, cls) --> (x, y, x, y, conf, cls)
            preds[:, 4] = preds[:, 4] * preds[:, 5]
            preds = preds[:, [0, 1, 2, 3, 4, 6]]

            # scale bboxes to original image
            preds[:, [0, 2]] = preds[:, [0, 2]] * self.w_r
            preds[:, [1, 3]] = preds[:, [1, 3]] * self.h_r

            # everything under zero to zero
            preds = torch.clip(preds, min=0)

            # filter boxes by classes
            if self.args.classes:
                preds = preds[torch.isin(preds[:, 5].cpu(), torch.as_tensor(self.args.classes))]

        preds = self.preds_to_yolov8_results(path, preds, im, im0s, predictor)

        return preds

    def preds_to_yolov8_results(self, path, preds, im, im0s, predictor):
        predictor.results[0] = Results(
            path=path,
            boxes=preds,
            orig_img=im0s[0],
            names=predictor.model.names
        )
        return predictor.results


