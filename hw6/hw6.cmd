executable = beamsearch_maxent.sh
getenv     = true
error      = hw6-1.err
output     = acc-1
notification = never
arguments  = "examples/sec19_21.txt examples/sec19_21.boundary examples/m1.txt sys_output-1 0 1 1"
transfer_executable = false
request_memory = 2*1024
queue

executable = beamsearch_maxent.sh
getenv     = true
error      = hw6-2.err
output     = acc-2
notification = never
arguments  = "examples/sec19_21.txt examples/sec19_21.boundary examples/m1.txt sys_output-2 1 3 5"
transfer_executable = false
request_memory = 2*1024
queue

executable = beamsearch_maxent.sh
getenv     = true
error      = hw6-3.err
output     = acc-3
notification = never
arguments  = "examples/sec19_21.txt examples/sec19_21.boundary examples/m1.txt sys_output-3 2 5 10"
transfer_executable = false
request_memory = 2*1024
queue

executable = beamsearch_maxent.sh
getenv     = true
error      = hw6-4.err
output     = acc-4
notification = complete
arguments  = "examples/sec19_21.txt examples/sec19_21.boundary examples/m1.txt sys_output-4 3 10 100"
transfer_executable = false
request_memory = 2*1024
queue
