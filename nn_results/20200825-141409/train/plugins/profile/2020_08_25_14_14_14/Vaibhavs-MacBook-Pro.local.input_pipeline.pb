	���(\�@���(\�@!���(\�@	
��ߖ3�?
��ߖ3�?!
��ߖ3�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$���(\�@y�&1��?AJ+�@Y;�O��n�?*	      V@2F
Iterator::Model�&1��?!�袋.�L@)+�����?1颋.�(F@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat/�$��?!]t�E�7@)/�$��?1]t�E�7@:Preprocessing2U
Iterator::Model::ParallelMapV2�~j�t��?!E]t�E+@)�~j�t��?1E]t�E+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�~j�t��?!E]t�E+@)�~j�t��?1E]t�E+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!/�袋.@)����Mbp?1/�袋.@:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9	��ߖ3�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	y�&1��?y�&1��?!y�&1��?      ��!       "      ��!       *      ��!       2	J+�@J+�@!J+�@:      ��!       B      ��!       J	;�O��n�?;�O��n�?!;�O��n�?R      ��!       Z	;�O��n�?;�O��n�?!;�O��n�?JCPU_ONLYY	��ߖ3�?b 