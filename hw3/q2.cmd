executable = build_NB1.sh
getenv     = true
error      = hw2-q2-1.err
output     = acc-q2-1
notification = never
arguments  = "examples/train.vectors.txt examples/test.vectors.txt 0 0.1 model-q2-1 sys_output-q2-1"
transfer_executable = false
request_memory = 2*1024
queue

executable = build_NB1.sh
getenv     = true
error      = hw2-q2-2.err
output     = acc-q2-2
notification = never
arguments  = "examples/train.vectors.txt examples/test.vectors.txt 0 0.5 model-q2-2 sys_output-q2-2"
transfer_executable = false
request_memory = 2*1024
queue

executable = build_NB1.sh
getenv     = true
error      = hw2-q2-3.err
output     = acc-q2-3
notification = never
arguments  = "examples/train.vectors.txt examples/test.vectors.txt 0 1 model-q2-3 sys_output-q2-3"
transfer_executable = false
request_memory = 2*1024
queue
