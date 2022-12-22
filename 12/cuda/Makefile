try: try.cu
	/opt/nvidia/hpc_sdk/Linux_x86_64/2022/compilers/bin/nvcc -ccbin=g++-11 --gpu-code sm_75 --gpu-architecture compute_75 -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING $< -o $@ -lcudadevrt -rdc=true -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/include -I/home/pseyfert/coding/explore-rangev3/install-range-v3/include/

both: load.o try.o
	/opt/nvidia/hpc_sdk/Linux_x86_64/2022/compilers/bin/nvcc -ccbin=g++-11 --gpu-code sm_75 --gpu-architecture compute_75 -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING load.o try.o -o $@ -lcudadevrt -rdc=true

try.o: try.cu
	/opt/nvidia/hpc_sdk/Linux_x86_64/2022/compilers/bin/nvcc -ccbin=g++-11 --gpu-code sm_75 --gpu-architecture compute_75 -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING $< -c -o $@ -rdc=true -I/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/include -I/home/pseyfert/coding/explore-rangev3/install-range-v3/include/ --expt-relaxed-constexpr

load.o: load.cpp
	g++ -Wall -Wextra -std=c++20 $< -o $@ -c -I/home/pseyfert/coding/explore-rangev3/install-range-v3/include/

clean:
	rm -f try *.o

.PHONY: clean
