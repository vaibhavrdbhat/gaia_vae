	�&1�
@�&1�
@!�&1�
@	�������?�������?!�������?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�&1�
@�"��~j�?AbX9��@Y)\���(�?*	     ��@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate+���?!.�߯7W@)�����M�?1Xڡ��`V@:Preprocessing2U
Iterator::Model::ParallelMapV2;�O��n�?!!�
��@);�O��n�?1!�
��@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�� �rh�?!��g�H@)�� �rh�?1��g�H@:Preprocessing2F
Iterator::ModelL7�A`�?!�?NC�@)���Q��?1E�3��@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���Q��?!E�3��@)���Q��?1E�3��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����x��?!�n�?W@)����Mb`?1�d{4�?:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�������?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�"��~j�?�"��~j�?!�"��~j�?      ��!       "      ��!       *      ��!       2	bX9��@bX9��@!bX9��@:      ��!       B      ��!       J	)\���(�?)\���(�?!)\���(�?R      ��!       Z	)\���(�?)\���(�?!)\���(�?JCPU_ONLYY�������?b 