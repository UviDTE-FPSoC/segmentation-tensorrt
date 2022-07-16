# segmentation-tensorrt
Code to train in cloud and run in Jetson the Severstal defect segmentation model for metal parts 

## Repository contents
* `inference`: Source code and for the TensorRT inferece. Uses OpenCV 4.5.0 and TensorRT 7.1.3 with CUDA 10.2.
    * `binary`: Source code for the binary inference. Reduced precision can be modified before compilation in infer_bin.cpp, the required changes are specified in the code.
    
        Compilation (inside folder):
        <pre><code>mkdir build</code></pre>
        <pre><code>cd build && cmake -DOpenCV_DIR=/usr/include/opencv4 -DTensorRT_DIR=/usr/src/tensorrt .. && make -j8 && cd .. && cp build/infer_bin infer_bin </code></pre>
        Inference (inside folder):
        <pre><code>./infer_bin {network name}.onnx</code></pre>

    * `seg`: Source code for the segmentation inference. Reduced precision can be modified before compilation in infer_seg.cpp, the required changes are specified in the code.

        Compilation (inside folder):
        <pre><code>mkdir build</code></pre>
        <pre><code>cd build && cmake -DOpenCV_DIR=/usr/include/opencv4 -DTensorRT_DIR=/usr/src/tensorrt .. && make -j8 && cd .. && cp build/infer_seg infer_seg </code></pre>
        Inference (inside folder):
        <pre><code>./infer_seg {network name}.onnx</code></pre>

    * `calib_files.csv`: CSV file with image names to calibrate INT8 quantization.

    * `valid_files.csv`: CSV file with image names, labels, and mask names for validation.

* training: Training Jupyter notebook, includes the original notebook [from Reynaldo Vazquez](https://colab.research.google.com/github/reyvaz/steel-defect-segmentation/blob/master/steel_defect_detection.ipynb#scrollTo=lu0YlVHE8H2T) and additional code used in this project:
    * Evaluation of a single segmentation network
    * Exporting model to ONNX
    * Inference execution time, approximate measurements
    * Preparing image labels and validation CSV

* utils: Auxiliary Python scripts
    * `calib_files.py`: Script to generate calib_files.csv from the dataset, excluding images used for validation.
    * `monitor_gpu.py`: Script to measure the average power consumption through the jtop library.
    * `dice_coef.py`: Script to post-process the output masks and evaluate the detection rate of the segmentation, comparing to the ground truth.
    * `dice_coef_total.py`: Script to post-process the output masks and evaluate the detection rate of the segmentation, discarding those images that were deemed non-defective, and comparing to the ground truth.
