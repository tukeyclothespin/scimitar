docker build -f Dockerfile.arabic_detection -t hammer.arabic_detection .
#Ports 6006 for tensorboard and 8888 for jupyter
NV_GPU=1 nvidia-docker run -p 8888:8888 -p 6006:6006 -v ~/arabic-text/:/arabic_text -v ~/pretrained_tensorflow/:/prog/models/model/pretrained hammer.arabic_detection

docker build -f Dockerfile.arabic_detection_altport -t hammer.arabic_detection_altport .
#Port 9999 for jupyter
NV_GPU=1 nvidia-docker run -p 9999:9999 -v ~/arabic-text/:/arabic_text -v ~/pretrained_tensorflow/:/prog/models/model/pretrained hammer.arabic_detection_altport

# Found at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
#ssd_inception_v2_coco
http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_08.tar.gz

#faster_rcnn_resnet50_coco
http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2017_11_08.tar.gz

#faster_rcnn_resnet101_coco
http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2017_11_08.tar.gz

#1. parse data

python3 parse_activ.py

# 2. run tensorboard
tensorboard --logdir /prog/models/model/eval

# 3. start training

python3 object_detection/train.py --logtostderr --pipeline_config_path=/prog/models/model/faster_rcnn_resnet50_coco.config --train_dir=/prog/models/model/train

OR
python3 object_detection/train.py --logtostderr --pipeline_config_path=/prog/models/model/faster_rcnn_resnet101_coco.config --train_dir=/prog/models/model/train
python3 object_detection/train.py --logtostderr --pipeline_config_path=/prog/models/model/ssd_inception_v2_coco.config --train_dir=/prog/models/model/train


# 4. start eval
python3 object_detection/eval.py --logtostderr --pipeline_config_path=/prog/models/model/faster_rcnn_resnet50_coco.config --checkpoint_dir=/prog/models/model/train --eval_dir=/prog/models/model/eval

OR
python3 object_detection/eval.py --logtostderr --pipeline_config_path=/prog/models/model/faster_rcnn_resnet101_coco.config --checkpoint_dir=/prog/models/model/train --eval_dir=/prog/models/model/eval
python3 object_detection/eval.py --logtostderr --pipeline_config_path=/prog/models/model/ssd_inception_v2_coco.config --checkpoint_dir=/prog/models/model/train --eval_dir=/prog/models/model/eval


#5 export model

python3 object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /prog/models/model/faster_rcnn_resnet50_coco.config \
    --trained_checkpoint_prefix /prog/models/model/train/model.ckpt-200000 \
    --output_directory /prog/models/exported/


ln -s [folder] [folder]


/prog
    parse_activ.py
    Dockerfile.arabic_detection
    +data
        object-detection.pbtxt
        eval.tfrecord
        training.tfrecord


    +models
        +model
            +eval
            +train
            -faster_rcnn_resnet101_coco.config
            +faster_rcnn_resnet101_coco_2017_11_08
            -faster_rcnn_resnet50_coco.config
            +faster_rcnn_resnet50_coco_2017_11_08
            -ssd_inception_v2_coco.config
            +ssd_inception_v2_coco_2017_11_08



lessons learned
- batch size is 1 because it can't take two objects with diff shape in same batch
- multiple bounding boxes in one image needs to have list of things
- what about examples with no bounding boxes?
- hanging memory issue
- change classes to 1 in config
- data aug options
- In eval.py:

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""



