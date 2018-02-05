executable = build_kNN.sh
getenv     = true
error      = hw4-E-1.err
output     = acc-E-1
notification = never
arguments  = "examples/train.vectors.txt examples/test.vectors.txt 1 1 sys_output-E-1"
transfer_executable = false
request_memory = 2*1024
queue

executable = build_kNN.sh
getenv     = true
error      = hw4-E-5.err
output     = acc-E-5
notification = never
arguments  = "examples/train.vectors.txt examples/test.vectors.txt 5 1 sys_output-E-5"
transfer_executable = false
request_memory = 2*1024
queue

executable = build_kNN.sh
getenv     = true
error      = hw4-E-10.err
output     = acc-E-10
notification = never
arguments  = "examples/train.vectors.txt examples/test.vectors.txt 10 1 sys_output-E-10"
transfer_executable = false
request_memory = 2*1024
queue


executable = build_kNN.sh
getenv     = true
error      = hw4-C-1.err
output     = acc-C-1
notification = never
arguments  = "examples/train.vectors.txt examples/test.vectors.txt 1 2 sys_output-C-1"
transfer_executable = false
request_memory = 2*1024
queue

executable = build_kNN.sh
getenv     = true
error      = hw4-C-5.err
output     = acc-C-5
notification = never
arguments  = "examples/train.vectors.txt examples/test.vectors.txt 5 2 sys_output-C-5"
transfer_executable = false
request_memory = 2*1024
queue

executable = build_kNN.sh
getenv     = true
error      = hw4-C-10.err
output     = acc-C-10
notification = never
arguments  = "examples/train.vectors.txt examples/test.vectors.txt 10 2 sys_output-C-10"
transfer_executable = false
request_memory = 2*1024
queue
