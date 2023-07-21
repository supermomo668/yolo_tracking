import cv2
import numpy as np
import torch

from ...appearance.reid_multibackend import ReIDDetectMultiBackend
from ...utils.matching import NearestNeighborDistanceMetric
from ...utils.ops import xyxy2xywh
from .sort.detection import Detection
from .sort.tracker import Tracker


class StrongSORT(object):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        max_dist=0.2,
        max_iou_dist=0.7,
        max_age=70,
        max_unmatched_preds=7,
        n_init=3,
        nn_budget=100,
        mc_lambda=0.995,
        ema_alpha=0.9,
    ):
        self.model = ReIDDetectMultiBackend(
            weights=model_weights, device=device, fp16=fp16
        )

        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric("cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric,
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            max_unmatched_preds=max_unmatched_preds,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
        )
        self.previous_img = None

    def update(self, dets, img):
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        self.tracker.camera_update(previous_img=self.previous_img, current_img=img)
        self.previous_img = img

        xyxys = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]

        xywhs = xyxy2xywh(xyxys)
        confs = confs
        self.height, self.width = img.shape[:2]

        # generate detections
        features = self._get_features(xywhs, img)
        bbox_tlwh = self._xywh_to_tlwh(xywhs)
        detections = [
            Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confs)
        ]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, clss, confs)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)

            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            # queue = track.q
            outputs.append(
                np.array([x1, y1, x2, y2, track_id, conf, class_id], dtype=np.float64)
            )
        outputs = np.asarray(outputs)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    @torch.no_grad()
    def _get_features(self, bbox_xywh, img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features

    def trajectory(self, im0, q, color):
        # Add rectangle to image (PIL-only)
        for i, p in enumerate(q):
            thickness = int(np.sqrt(float(i + 1)) * 1.5)
            if p[0] == "observationupdate":
                cv2.circle(im0, p[1], 2, color=color, thickness=thickness)
            else:
                cv2.circle(im0, p[1], 2, color=(255, 255, 255), thickness=thickness)
