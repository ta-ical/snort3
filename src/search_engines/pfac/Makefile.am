AUTOMAKE_OPTIONS=foreign no-dependencies

lib_LIBRARIES = libspfacnv.a

AM_CXXFLAGS=-std=gnu++11

libspfacnv_a_SOURCES = \
    pfac_file.cpp pfac_file.h \
    pfac_table.cu pfac_table.h \
    pfac_match.cu pfac_match.cuh \
    pfac.cu pfac.h \
    pfac_kernel.cu

libspfacnv_a_LDFLAGS = $(AM_LDFLAGS) -L/usr/local/cuda-8.0/lib64 -lcudart
libspfacnv_a_CFLAGS = $(CUDA_CFLAGS) -std=c++11
libspfacnv_a_AR = $(NVCC) -static -arch=sm_35 -lib -o

.cu.o: pfac_table.h
	$(NVCC) $(libspfacnv_a_CFLAGS) -arch=sm_35 -dc -o $@ $<

INCLUDES = @INCLUDES@
