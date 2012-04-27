#***************************************************************************
#                               Makefile
#                          -------------------
#                           W. Michael Brown
#
#  _________________________________________________________________________
#    Mike's Build for Geryon Unified Coprocessor Library (UCL)
#
#  _________________________________________________________________________
#
#    begin                : Thu November 12 2009
#    copyright            : (C) 2009 by W. Michael Brown
#    email                : brownw@ornl.gov
# ***************************************************************************/

BIN_DIR = ../../../bin
OBJ_DIR = ../../../obj/geryon

OCL_LINK  = -lOpenCL

# ---------------------------
#  Uncomment on snow leopard
#SL32      = -m32 -I./opencl_1_0
#OCL_LINK  = -framework OpenCL
# ---------------------------

CPP       = g++ -O2 $(SL32) -Wall
NVC_CPP   = -O2 -I/usr/local/cuda/include
OCL_CPP   = -O2 -I./opencl
LINK      = g++ -O2 $(SL32)
NVC_LINK  = -L/usr/local/cuda/lib64/ -lcudart -lcuda

CUDA_CPP  = nvcc -g -I/usr/local/cuda/include -DUNIX -O2 -Xptxas -v \
            --use_fast_math $(GPU_FLAG) $(SL32)
CUDA_ARCH = -arch=sm_20
CUDA_LINK = -L/usr/local/cuda/lib64 -lcudart $(CUDA_LIB)

AR = ar

# -----------------------------------------------------------------------------

CUDA = $(CUDA_CPP) $(CUDA_ARCH) $(CUDA_PREC)
OCL  = $(CPP) $(OCL_CPP)
NVC  = $(CPP) $(NVC_CPP)

# Headers for UCL
UCL_H  = $(wildcard ucl*.h)

# Headers for CUDA-Driver Specific UCL
NVD_H  = $(wildcard nvd*.h) $(UCL_H)

# Headers for CUDA-RT Specific UCL
NVC_H  = $(wildcard nvc*.h) $(UCL_H)

# Headers for OpenCL Specific UCL
OCL_H  = $(wildcard ocl*.h) $(UCL_H)

EXECS = $(BIN_DIR)/nvc_get_devices $(BIN_DIR)/ocl_get_devices \
        $(BIN_DIR)/nvd_get_devices

EXAMPLES = $(BIN_DIR)/ucl_example_cdr $(BIN_DIR)/ucl_example_ocl \
           $(BIN_DIR)/ucl_example_crt

DEVICE_BIN = $(OBJ_DIR)/ucl_test_kernel.cubin $(OBJ_DIR)/ucl_test_kernel.ptx \
             $(OBJ_DIR)/ucl_test_kernel_d.ptx

all: $(EXECS)

$(BIN_DIR)/nvc_get_devices: ucl_get_devices.cpp $(NVC_H)
	$(NVC) -o $@ ucl_get_devices.cpp -DUCL_CUDART $(NVC_LINK) 

$(BIN_DIR)/ocl_get_devices: ucl_get_devices.cpp
	$(OCL) -o $@ ucl_get_devices.cpp -DUCL_OPENCL $(OCL_LINK) 

$(BIN_DIR)/nvd_get_devices: ucl_get_devices.cpp $(NVD_H)
	$(NVC) -o $@ ucl_get_devices.cpp -DUCL_CUDADR $(NVC_LINK) 

$(OBJ_DIR)/ucl_test_kernel.o: ucl_test_kernel.cu
	$(CUDA) -DNV_KERNEL -DOrdinal=int -DScalar=float -c -o $@ ucl_test_kernel.cu 

$(OBJ_DIR)/ucl_test_kernel.cubin: ucl_test_kernel.cu
	$(CUDA) -DNV_KERNEL -DOrdinal=int -DScalar=float -cubin -o $@ ucl_test_kernel.cu 

$(OBJ_DIR)/ucl_test_kernel.ptx: ucl_test_kernel.cu
	$(CUDA) -DNV_KERNEL -DOrdinal=int -DScalar=float -ptx -o $@ ucl_test_kernel.cu 

$(OBJ_DIR)/ucl_test_kernel_d.ptx: ucl_test_kernel.cu
	$(CUDA) -DNV_KERNEL -DOrdinal=int -DScalar=double -ptx -o $@ ucl_test_kernel.cu 

$(BIN_DIR)/ucl_test: ucl_test.cpp ucl_test_source.h $(NVC_H) $(OCL_H) $(NVD_H) $(OBJ_DIR)/ucl_test_kernel.cubin $(OBJ_DIR)/ucl_test_kernel.o
	$(CPP) $(OCL_CPP) $(NVC_CPP) -o $@ ucl_test.cpp $(OBJ_DIR)/ucl_test_kernel.o $(OCL_LINK) $(NVC_LINK)

$(BIN_DIR)/ucl_test_debug: ucl_test.cpp ucl_test_source.h $(NVC_H) $(OCL_H) $(NVD_H) $(OBJ_DIR)/ucl_test_kernel.cubin $(OBJ_DIR)/ucl_test_kernel.o
	$(CPP) $(OCL_CPP) $(NVC_CPP) -o $@ -DUCL_DEBUG -DUCL_NO_EXIT -g ucl_test.cpp $(OBJ_DIR)/ucl_test_kernel.o $(OCL_LINK) $(NVC_LINK)

$(BIN_DIR)/ucl_example_ocl: example.cpp $(OCL_H)
	$(OCL) -o $@ example.cpp -DUSE_OPENCL $(OCL_LINK) 

$(BIN_DIR)/ucl_example_cdr: example.cpp $(NVD_H)
	$(NVC) -o $@ example.cpp -DUSE_CUDA_DRIVER $(NVC_LINK) 

$(OBJ_DIR)/example_kernel.o: ucl_test_kernel.cu example_kernel.cu
	$(CUDA) -DNV_KERNEL -DOrdinal=int -DScalar=float -c -o $@ example_kernel.cu 

$(BIN_DIR)/ucl_example_crt: example.cpp $(NVC_H) $(OBJ_DIR)/example_kernel.o
	$(NVC) -DUSE_CUDA_RUNTIME -o $@ example.cpp $(OBJ_DIR)/example_kernel.o $(NVC_LINK)

test: $(BIN_DIR)/ucl_test $(DEVICE_BIN) $(EXAMPLES)
	ucl_test $(OBJ_DIR); 

dbg_test: $(BIN_DIR)/ucl_test_debug $(DEVICE_BIN) $(EXAMPLES)
	ucl_test_debug $(OBJ_DIR); 

clean:
	rm -f $(EXECS) $(OBJS) $(OCL_EXECS) $(OCL_OBS) *.linkinfo \
	      $(BIN_DIR)/ucl_test $(BIN_DIR)/ucl_test_debug $(EXAMPLES) \
	      $(DEVICE_BIN)

veryclean: clean
	rm -f *~ *.linkinfo

