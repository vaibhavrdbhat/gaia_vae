	�~j�t@�~j�t@!�~j�t@	����?����?!����?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�~j�t@�V-�?A5^�I@Y;�O��n�?*	     �T@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����Mb�?!��18�C@)V-��?1�D�JԮA@:Preprocessing2U
Iterator::Model::ParallelMapV29��v���?!�+Q��?@)9��v���?1�+Q��?@:Preprocessing2F
Iterator::Model�l����?!�����F@)�I+��?1ծD�J�*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~j�t��?!D�JԮD-@)�I+��?1ծD�J�*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t�h?!D�JԮD@)�~j�t�h?1D�JԮD@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapL7�A`�?! ��18D@)����MbP?1��18��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����MbP?!��18��?)����MbP?1��18��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9����?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�V-�?�V-�?!�V-�?      ��!       "      ��!       *      ��!       2	5^�I@5^�I@!5^�I@:      ��!       B      ��!       J	;�O��n�?;�O��n�?!;�O��n�?R      ��!       Z	;�O��n�?;�O��n�?!;�O��n�?JCPU_ONLYY����?b 