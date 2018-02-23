executable = TBL_classify.sh
getenv     = true
output     = out_1
notification = never
arguments  = "examples/train2.txt model_file sys_output_1 1"
transfer_executable = false
request_memory = 2*1024
queue

executable = TBL_classify.sh
getenv     = true
output     = out_5
notification = never
arguments  = "examples/train2.txt model_file sys_output_5 5"
transfer_executable = false
request_memory = 2*1024
queue

executable = TBL_classify.sh
getenv     = true
output     = out_10
notification = never
arguments  = "examples/train2.txt model_file sys_output_10 10"
transfer_executable = false
request_memory = 2*1024
queue

executable = TBL_classify.sh
getenv     = true
output     = out_20
notification = never
arguments  = "examples/train2.txt model_file sys_output_20 20"
transfer_executable = false
request_memory = 2*1024
queue

executable = TBL_classify.sh
getenv     = true
output     = out_50
notification = never
arguments  = "examples/train2.txt model_file sys_output_50 50"
transfer_executable = false
request_memory = 2*1024
queue
executable = TBL_classify.sh
getenv     = true
output     = out_100
notification = never
arguments  = "examples/train2.txt model_file sys_output_100 100"
transfer_executable = false
request_memory = 2*1024
queue

executable = TBL_classify.sh
getenv     = true
output     = out_200
notification = never
arguments  = "examples/train2.txt model_file sys_output_200 200"
transfer_executable = false
request_memory = 2*1024
queue
