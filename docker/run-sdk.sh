source_video="${1:$demo_video}"

model="${2:-splitted-dataset-yolov82/weights/best.pt}"
track_conf="${3:-data/conf/botsort.yaml}"
# hyperparm

mkdir -p results
python examples/track.py --yolo-model $model --source $source_video  --tracking-method deepocsort --conf 0.5 --iou 0.3 --save --save-mot --vid-stride 10 --project agrosuper --name $(dirname source_video)