# inference
Used for batch inference on images. Configuration file is `settings.ini` in the same directory.

## Usage
In the command line, run
```bash
python infer.py
```

## Config File
Here are the descriptions for each element in the configuration file.

### [Tensorflow]

* `GPUDevice` - id of the GPU device tensorflow should use
* `BatchSize` - image batch size to be used
* `ModelFile` - path to the Keras model yaml file
* `WeightsFile` - path to the model weights h5 file
* `InputWidth` - model input image width
* `InputHeight` - model input image height

### [Files]

* `ImageDir` - path to the directory containing png images for inference
* `OutputDir` - output directory for JSON files
