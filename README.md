# Fish tracking using deepSORT and tflite object detection

Fish tracking implemented with tflite object detection and DeepSort (Simple Online and Realtime Tracking with a Deep Association Metric).
This project contain a colab notebook to train a detecction model and export it as a tflite model. Then this model is used as a fish detector, that produces input to the DeepSORT tracking.
Instructions for using this project can be found in instructions.pdf.
data_annotation.pdf contain details on how to prepare ground-truth bounding boxes to evaluate the tracking.
Results of parameter tuning can be found in performance_evaluation.pdf.
