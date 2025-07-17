# CUDA-GDB commands for debugging HGETF2_kernel
# Usage: cuda-gdb -x debug_commands.gdb ./debug_program matrix_3x3.txt -v

# Enable CUDA debugging
set cuda memcheck on
set cuda break_on_launch application

# Set environment for better CUDA debugging
set environment CUDA_DEBUGGER_SOFTWARE_PREEMPTION=1

# Set breakpoint at the exact kernel launch line in MPF.cu
break MPF.cu:126

# Also set breakpoint inside the kernel
break HGETF2_kernel

# Run the program with arguments
run matrix_3x3.txt -v

# When the first breakpoint hits (kernel launch), step into kernel
step

# Now you should see CUDA threads. Use these commands:
# info cuda threads
# info cuda blocks  
# info cuda grids
# cuda thread (0,0,0)
# print tid
# print rows
# print cols
# step
# next

# Interactive debugging commands (use manually):
printf "=== Breakpoint hit. Use these commands manually: ===\n"
printf "info cuda threads    - List all CUDA threads\n" 
printf "cuda thread (0,0,0)  - Switch to block(0,0) thread 0\n"
printf "cuda thread (0,0,1)  - Switch to block(0,0) thread 1\n"
printf "print tid            - Print current thread ID\n"
printf "print rows           - Print number of rows\n"
printf "print cols           - Print number of columns\n"
printf "step                 - Step into next line\n"
printf "next                 - Step over next line\n"
printf "continue             - Continue execution\n"
