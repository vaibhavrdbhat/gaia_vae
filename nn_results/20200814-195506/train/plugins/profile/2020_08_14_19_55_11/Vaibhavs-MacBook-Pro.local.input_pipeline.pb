	��K7�A@��K7�A@!��K7�A@	�w�Zn�?�w�Zn�?!�w�Zn�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��K7�A@V-��?A{�G�z@Y+�����?*	     �T@2F
Iterator::Model+�����?!A	o4u~G@)X9��v��?1��k��B@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���Q��?!��ˊ�B@)���Q��?1��ˊ�B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�I+��?!���ˊ�*@)�I+��?1���ˊ�*@:Preprocessing2U
Iterator::Model::ParallelMapV2����Mb�?!]V��F#@)����Mb�?1]V��F#@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t�h?!���h�@)�~j�t�h?1���h�@:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�w�Zn�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	V-��?V-��?!V-��?      ��!       "      ��!       *      ��!       2	{�G�z@{�G�z@!{�G�z@:      ��!       B      ��!       J	+�����?+�����?!+�����?R      ��!       Z	+�����?+�����?!+�����?JCPU_ONLYY�w�Zn�?b 