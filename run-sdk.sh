model="${1:-../runs/detect/splitted-dataset-yolov82/weights/best.pt}"
source_video="${2:-../data/agrosuper/demo_video/videos/nvr_ch1_20230915120000_20230915135942.mp4}"
track_conf="${3:-data/conf/botsort.yaml}"
# hyperparm

mkdir -p results
python examples/track.py --yolo-model $model --source $source_video  --tracking-method deepocsort --conf 0.5 --iou 0.3 --save --save-mot --vid-stride 10 --project agrosuper --name $(dirname source_video)