FROM ultralytics/ultralytics as base
ARG model
RUN adduser model-server
ENV model_path docker/splitted-dataset-yolov82/weights/best.pt


WORKDIR /home/yolotrack
COPY examples /home/yolotrack/examples
COPY ./requirements.txt ${model_path} setup.sh /home/yolotrack/
USER root
RUN chmod +x ./setup.sh && ./setup.sh

USER model-server

RUN python -m pip install -U pip && pip install -r requirements.txt 
FROM base as serve
USER root
COPY serve-sdk.sh /home/yolotrack
RUN chmod +x ./serve-sdk.sh && chown -R model-server ./serve-sdk.sh .
USER model-server
WORKDIR /home/yolotrack
ARG demo_video 
# COPY ${demo_video}

ENTRYPOINT ["uvicorn"]
CMD ["examples.serve:app"]


