#tflite_convert \
#  --output_file=./frozen_pb/segmentation.tflite \
#  --graph_def_file=./frozen_pb/frozen_model.pb \
#  --inference_type=QUANTIZED_UINT8 \
#  --input_arrays=image \
#  --output_arrays=conv2d/act_quant/FakeQuantWithMinMaxVars \
#  --mean_values=113 \
#  --std_dev_values=6

tflite_convert \
  --output_file=./frozen_pb/segmentation.tflite \
  --graph_def_file=./frozen_pb/frozen_model.pb \
  --input_arrays=image \
  --output_arrays=sigmoid_output
