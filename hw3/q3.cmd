executable = build_NB2.sh
getenv     = true
error      = hw2-q3-1.err
output     = acc-q3-1
notification = never
arguments  = "examples/train.vectors.txt examples/test.vectors.txt 0 0.1 model-q3-1 sys_output-q3-1"
transfer_executable = false
request_memory = 2*1024
queue

executable = build_NB2.sh
getenv     = true
error      = hw2-q3-2.err
output     = acc-q3-2
notification = never
arguments  = "examples/train.vectors.txt examples/test.vectors.txt 0 0.5 model-q3-2 sys_output-q3-2"
transfer_executable = false
request_memory = 2*1024
queue

executable = build_NB2.sh
getenv     = true
error      = hw2-q3-3.err
output     = acc-q3-3
notification = never
arguments  = "examples/train.vectors.txt examples/test.vectors.txt 0 1 model-q3-3 sys_output-q3-3"
transfer_executable = false
request_memory = 2*1024
queue
