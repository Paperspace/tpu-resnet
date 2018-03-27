# Paperspace Gradient updates to Google's TPU ResNet tutorial code

This is a straightforward port of the Resnet-50 model done by Google's TPU team, but with slight modifications to enable the code to run as a Gradient job. The code was based on the original version from the tensorflow/models repository.

The only adjustments have been to add environment variables as sources of information for the main python script, to enable the script to access Gradient allocated TPUs and Google cloud storage buckets.  (Currently TPUs running in Google's cloud need to read and write data to/from Google cloud storage directly, as opposed to a local file system location or network mount.)

### Running the model on Gradient
This model assumes the input data is availabe in a publicly accessible Google cloud storage bucket--in this case we are using the randomly generated fake dataset located at `gs://cloud-tpu-test-datasets/fake_imagenet`

python resnet_main.py \
  --master=$TPU_GRPC_URL \
  --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet \
  --model_dir=$TPU_MODEL_DIR \
  --train_batch_size=1024 \
  --eval_batch_size=128 \

The $TPU_GRPC_URL parameter is an environment variable provided by the Gradient job runner cluster machine which identifies the ip address and port of the TPU device assigned to the job.

The $TPU_MODEL_DIR is a Google cloud storage bucket provided to the job for output of the model results.  Any results created in this path by the TPU code will be uploaded to the job artifacts list on completion of the job.

### Running the model with Real Data
The Google cloud storage bucket specified by the $TPU_MODEL_DIR environment variable can also be used as the root of a staging location for training data to be read by the model.  To use this bucket as both an input source and an output destination we recommend you put the input data and output data in separate subfolders, e.g., by using options of the form:

  --data_dir=$TPU_MODEL_DIR/input_data
  --model_dir=$TPU_MODEL_DIR/output_data

Also to prevent uploading of the input data at the completion of the job, you may wish to delete that subirectory tree within the bucket.  You can do this with the gsutil tool from Google, from within your job script.

To create an ImageNet dataset in the correct format for tensorflow you can use this Google provided [script](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py).  See the script comments for more information.
