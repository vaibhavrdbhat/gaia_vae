�	%��C@%��C@!%��C@	'�X�T��?'�X�T��?!'�X�T��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$%��C@�x�&1�?AL7�A`�@Y�� �rh�?*	     �M@2F
Iterator::ModelX9��v��?!��}ylEJ@)�~j�t��?1�Iݗ�VD@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!��/��6@){�G�z�?1"h8���0@:Preprocessing2U
Iterator::Model::ParallelMapV2y�&1�|?!d+����'@)y�&1�|?1d+����'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate;�O��n�?!���c+�.@)�~j�t�x?1�Iݗ�V$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy�&1��?!d+����G@)����Mbp?1'u_@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�~j�t�h?!�Iݗ�V@)�~j�t�h?1�Iݗ�V@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�~j�t�h?!�Iݗ�V@)�~j�t�h?1�Iݗ�V@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�I+��?!�؊��2@)����Mb`?1'u_@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 4.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9'�X�T��?>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�x�&1�?�x�&1�?!�x�&1�?      ��!       "      ��!       *      ��!       2	L7�A`�@L7�A`�@!L7�A`�@:      ��!       B      ��!       J	�� �rh�?�� �rh�?!�� �rh�?R      ��!       Z	�� �rh�?�� �rh�?!�� �rh�?JCPU_ONLYY'�X�T��?b Y      Y@qr� ��I@"�
both�Your program is POTENTIALLY input-bound because 4.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb�51.8596% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 