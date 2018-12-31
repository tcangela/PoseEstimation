#python3 src/gen_segmentation_frozen_pb.py --checkpoint=/data3/tangc/segmentation/models/resnet18_seg_batch-128_lr-0.001_gpus-1_192x192_experiments-resnet18_seg/model-218000 --output_graph=./frozen_pb/frozen_model.pb

#python3 src/gen_segmentation_frozen_pb.py --checkpoint=/data3/tangc/segmentation/models/mobilenetv1_batch-128_lr-0.001_gpus-1_192x192_experiments-mobilenet_seg/model-100000 --output_graph=./frozen_pb/frozen_model.pb

#python3 src/gen_segmentation_frozen_pb.py --checkpoint=/data3/tangc/segmentation/models_quantize/mobilenetv1_batch-128_lr-0.001_gpus-1_192x192_experiments-mobilenet_seg_quantize/model-100 --output_graph=./frozen_pb/frozen_model.pb --quantize=True


#python3 src/gen_segmentation_frozen_pb.py --checkpoint=/data3/tangc/segmentation/0.25-112-sigmoid-models/mobilenetv1_batch-128_lr-0.001_gpus-1_112x112_experiments-0.25-112-sigmoid_mobilenet_seg/model-210000 --output_graph=./frozen_pb/frozen_model.pb


python3 src/gen_segmentation_frozen_pb.py --checkpoint=/data3/tangc/segmentation/0.25-112-models/mobilenetv1_batch-128_lr-0.001_gpus-1_112x112_experiments-0.25-112-mobilenet_seg/model-210000 --output_graph=./frozen_pb/frozen_model.pb
