# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
import sys, os
sys.path.append('./')

import argparse
from functools import partial
from pathlib import Path

import torch, numpy as np, cv2

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from examples.detectors import get_yolo_inferer

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from examples.utils import write_mot_results

# Enable SQL
import mysql.connector
import datetime
from collections import defaultdict

# DEFINE DATABASE VARIABLES
DB_CONFIG = {
    'database': os.getenv('MYSQL_DATABASE'),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'host': os.getenv('MYSQL_HOST', 'localhost'),  # Default to localhost if not set
    'port': os.getenv('MYSQL_PORT', '3306'),  # Default to 3306 if not set
}

    # Try to connect to the database
def _connect_to_mysql(DB_CONFIG):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        print(f"SQL connection error: {err}")
        return False, None
    return conn.is_connected(), conn

mysql_conn, conn = _connect_to_mysql(DB_CONFIG)

DB_CONFIG['table'] = os.getenv('MYSQL_TABLE')
    # SQL query to insert data into the table
insert_query = f"""
    INSERT INTO {DB_CONFIG['table']} (dwell, track_id, id_account, id_branch, cam_id, cam_name, count, zone)
    VALUES (%s, %s, -1, -1, -1, -1, %s, -1)
"""
        
def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)
    predictor.trackers = trackers

def format_timedelta(td: datetime.timedelta) -> str:
    # Extract days, seconds and microseconds from timedelta
    days = td.days
    seconds = td.seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Format as '%d%H:%M:%S'
    formatted = f"{days}:{hours:02}:{minutes:02}:{seconds:02}"
    return formatted

class tracker:
    """
    Tracker object intend to store states
    """
    def __init__(self, args):
        self.counter = 0
        self.args = args
        self.yolo = self.reset(args)
        self.t_delta_store, self.t_first = defaultdict(0), defaultdict(lambda: datetime.datetime.now())
        
    def increment(self):
        self.counter += 1
        if self.counter >=1:
            self.yolo.predictor.custom_args = self.args
        return self.counter
    
    def id_t_deltas(self, cur_time, ids):
        t_time = []
        for id in ids:
            self.t_delta_store[id] += cur_time - self.t_delta_store[id]
            t_time.append(self.t_delta_store[id])
        return t_time

    def reset(self, args):
        self.counter = 0
        self.yolo = YOLO(
            args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
        )
        if 'yolov8' not in str(args.yolo_model):
            # replace yolov8 model
            m = get_yolo_inferer(args.yolo_model)
            model = m(
                model=args.yolo_model,
                device=self.yolo.predictor.device,
                args=self.yolo.predictor.args
            )
            self.yolo.predictor.model = model
        self.yolo.add_callback(
            'on_predict_start', 
            partial(on_predict_start, persist=True))
        return self.yolo
    
    @property
    def model(self):
        return self.yolo
        
from fastapi import FastAPI

app = FastAPI()
    
@app.get("/")
def read_root():
    return {"message": f"Hello, {__name__}"}

@torch.no_grad()
def run(args):
    track = tracker(args)
    yolo = track.yolo
    @app.get("/new")
    def refresh_tracker():
        return {"message": f"Tracker refreshed"}
    
    @app.post("/track")
    async def track_objects(file: UploadFile = Form(...), timestamp: str = Form(...)):
        # Streaming Inference Object
        # Read the uploaded image file into a NumPy array
        image_stream = np.frombuffer(file.file.read(), np.uint8)
        # Decode the NumPy array into an image
        payload = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)
        print(f"{type(payload)},{(payload.shape)}")
        try:
            results = yolo.track(
                source=payload,
                conf=args.conf,
                iou=args.iou,
                # show=args.show,
                stream=True,
                device=args.device,
                # show_conf=args.show_conf,
                # save_txt=args.save_txt,
                # show_labels=args.show_labels,
                # save=args.save,
                verbose=args.verbose,
                exist_ok=args.exist_ok,
                project=args.project,
                name=args.name,
                classes=args.classes,
                # imgsz=args.imgsz,
                # vid_stride=args.vid_stride,
                # line_width=args.line_width
            )
            track.increment()
        except Exception as e:
            return JSONResponse(
                content={"message": f"Error while tracking:{e}"}, status_code=400
            )
        # store custom args in predictor
        resp = []
        if hasattr(results, 'len') and len(results) > 1:
            print(f"[Warning] Results are not streamed but has more than 1 object. For image len(results)==1.")
        # iterate over results list obj to unpack
        for frame_idx, r in enumerate(results):
            box = r.boxes.data
            # interative over inner results
            if box.shape[1] == 7:
                track_id = r.boxes.id.int().cpu().tolist()
                boxes = r.boxes.xywh.cpu().tolist()
                centroids = r.boxes.xywh.cpu()[:, :2].tolist()
                conf = r.boxes.conf.float().cpu().tolist()
                cls = r.boxes.cls.int().cpu().tolist()
                t_deltas = track.id_t_deltas(datetime.datetime.now(), track_id)
                resp.append({
                    'id': track.counter,
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'track_id': track_id,
                    'duration': t_deltas,
                    'box': boxes, 'centroids': centroids,
                    'labels': cls, 'confidence': conf
                })
            # pt 2
            if box.shape[1] == 7:
                # Log & output results here
                if yolo.predictor.source_type.webcam or args.source.endswith(VID_FORMATS):
                    p = yolo.predictor.save_dir / 'mot' / (args.source + '.txt')
                    yolo.predictor.mot_txt_path = p
                elif 'MOT16' or 'MOT17' or 'MOT20' in args.source:
                    p = yolo.predictor.save_dir / 'mot' / (Path(args.source).parent.name + '.txt')
                    yolo.predictor.mot_txt_path = p

                if args.save_mot:
                    print(f"Wrote results to: {yolo.predictor.mot_txt_path}")
                    write_mot_results(
                        yolo.predictor.mot_txt_path,
                        r,
                        frame_idx,
                    )
                if mysql_conn:
                    # Insert data into the database
                    for d , tid in zip(t_deltas, track_id):
                        conn.cursor().execute(insert_query, (d, tid, np.unique(tid)))
                        conn.commit()
                if args.save_id_crops:
                    for d in r.boxes:
                        print('args.save_id_crops', d.data)
                        save_one_box(
                            d.xyxy,
                            r.orig_img.copy(),
                            file=(
                                yolo.predictor.save_dir / 'crops' /
                                str(int(d.cls.cpu().numpy().item())) /
                                str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                            ),
                            BGR=True
                        )
            else:
                print(f"[WARNING] box format/len < 7")
        if args.save_mot:
            print(f'MOT results saved to {yolo.predictor.mot_txt_path}')
        return {"results": resp}

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--vid_stride', default=1, type=int,
                        help='video frame-rate stride')
    opt = parser.parse_args()
    return opt

# Serve Credentials

from types import SimpleNamespace
from pathlib import PosixPath
serve_args = {
    'yolo_model': PosixPath('splitted-dataset-yolov82/weights/best.pt'), 
    'reid_model': PosixPath('examples/weights/osnet_x0_25_msmt17.pt'),
    'tracking_method': 'deepocsort', 'source': '0', 
    'imgsz': [640], 'conf': 0.5, 'iou': 0.7, 'device': 'cuda', 'show': False, 'save': False, 'classes': None, 'project': PosixPath('runs/track'), 'name': 'exp', 'exist_ok': False, 'half': False, 'vid_stride': 1, 'show_labels': True, 'show_conf': True, 'save_txt': False, 'save_id_crops': False, 'save_mot': False, 'line_width': None, 'per_class': False, 'verbose': True
    }
run(SimpleNamespace(**serve_args))

if __name__ == "__main__":
    
    import uvicorn
    args = parse_opt()
    print(f"Arguments:{args.__dict__}")
    run(args)