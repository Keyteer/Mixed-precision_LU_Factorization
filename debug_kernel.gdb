# Commands to debug HGETF2_kernel execution
set cuda break_on_launch application
set cuda api_failures stop

# Set breakpoint inside the kernel where we compute IPIV
break hgetf2_kernel.cu:60

# Run the program
run matrix_3x3.txt -v

# When we hit the breakpoint, inspect the kernel state:
# info cuda threads
# cuda thread (0,0,0)
# print tid
# print j
# print shared_piv
# print ipiv_panel[j]
# continue
