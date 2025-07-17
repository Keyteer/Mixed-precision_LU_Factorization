# Modern CUDA debugging commands for cuda-gdb
# Skip the deprecated memcheck and use these commands:

set cuda break_on_launch application
set cuda api_failures stop
break hgetf2_kernel.cu:26
run

# When breakpoint hits:
info cuda threads
cuda thread (0,0,0)
print tid
print j
cuda thread (0,0,1)
print tid
step
continue
