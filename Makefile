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

# Include specific makefile 
ifndef arch
ifeq (,$(findstring clean, $(MAKECMDGOALS)))
ifeq (,$(findstring setup, $(MAKECMDGOALS)))
ifeq (,$(findstring dist, $(MAKECMDGOALS)))
ifeq (,$(findstring dox, $(MAKECMDGOALS)))
$(error Please use: make arch=<your_arch>)
else
DOX = doxygen
endif
endif
endif
endif
else
include Makefile.$(arch)
endif

SH = /bin/sh

# Target binary, object and source(include) directory 
BIN_DIR = bin
OBJ_DIR = obj
INC_DIR = include

# Example: vec_add
nvc_example = nvc_example.o example_kernel.o 
nvd_example = nvd_example.o  
nvd_example_dep = example_kernel_bin.h 
ocl_example = ocl_example.o  
ocl_example_dep = example_kernel_str.h  

# Example: get_devices
nvc_get_devices = nvc_get_devices.o
nvd_get_devices = nvd_get_devices.o
ocl_get_devices = ocl_get_devices.o

# UCL test
ucl_test      = ucl_test.o ucl_test_kernel.o 
ucl_test_dep  = ucl_test_kernel.ptx ucl_test_kernel_d.ptx
ucl_test_dep += ucl_test_kernel.cubin ucl_test_kernel.cu

# Stand-alone tests
nvc_test      = nvc_test.o ucl_test_kernel.o
nvd_test      = nvd_test.o
nvd_test_dep  = ucl_test_kernel.ptx ucl_test_kernel_d.ptx
ocl_test      = ocl_test.o
ocl_test_dep  = ucl_test_kernel.cu

ocl_compiler = ocl_compiler.o

# Define your executables here
NVC_EXE = nvc_get_devices nvc_example  
NVD_EXE = nvd_get_devices nvd_example 
OCL_EXE = ocl_get_devices ocl_example ocl_compiler 
TST_EXE = ucl_test nvd_test nvc_test ocl_test

# Generate binary targets
NVC_BIN = $(addprefix $(BIN_DIR)/,$(NVC_EXE))
NVD_BIN = $(addprefix $(BIN_DIR)/,$(NVD_EXE))
OCL_BIN = $(addprefix $(BIN_DIR)/,$(OCL_EXE))
TST_BIN = $(addprefix $(BIN_DIR)/,$(TST_EXE))

# Compile lines
NVC += $(NVC_FLAGS) $(NVC_INCS) -I$(INC_DIR) -I$(OBJ_DIR)
OCL += $(OCL_FLAGS) $(OCL_INCS) -I$(INC_DIR) -I$(OBJ_DIR)
CPP += $(CPP_FLAGS) $(CPP_INCS) -I$(INC_DIR) -I$(OBJ_DIR)

# Prevent from deleating itermediate files
.SECONDARY: $(OBJ)
.PHONY: test all setup clean
.NOTPARALLEL: setup clean

# Default makefile target
all: setup $(NVC_EXE) $(NVD_EXE) $(OCL_EXE) $(TST_EXE)

define nvc_template
OBJ += $$(addprefix $$(OBJ_DIR)/,$$($(1)))
$(BIN_DIR)/$(1): $$(addprefix $$(OBJ_DIR)/,$$($(1)))
	$(NVC) -o $$@ $$^ -DUCL_CUDART $$(NVC_LIBS)

$(1): setup $$(addprefix $$(OBJ_DIR)/,$$($(1)_dep)) $(BIN_DIR)/$(1) 
endef

define nvd_template
OBJ += $$(addprefix $$(OBJ_DIR)/,$$($(1)))
$(BIN_DIR)/$(1): $$(addprefix $$(OBJ_DIR)/,$$($(1)))
	$(NVC) -o $$@ $$^ -DUCL_CUDADR $$(NVC_LIBS)

$(1): setup $$(addprefix $$(OBJ_DIR)/,$$($(1)_dep)) $(BIN_DIR)/$(1) 
endef

define ocl_template
OBJ += $$(addprefix $$(OBJ_DIR)/,$$($(1)))
$(BIN_DIR)/$(1): $$(addprefix $$(OBJ_DIR)/,$$($(1)))
	$(OCL) -o $$@ $$^ -DUCL_OPENCL $$(OCL_LIBS)

$(1): setup $$(addprefix $$(OBJ_DIR)/,$$($(1)_dep)) $(BIN_DIR)/$(1) 
endef

define tst_template
OBJ += $$(addprefix $$(OBJ_DIR)/,$$($(1)))
$(BIN_DIR)/$(1): $$(addprefix $$(OBJ_DIR)/,$$($(1)))
	$(OCL) -o $$@ $$^ $$(OCL_LIBS) $$(CPP_LIBS) $$(NVC_LIBS)

$(1): setup $$(addprefix $$(BIN_DIR)/,$$($(1)_dep)) $(BIN_DIR)/$(1)
endef

$(foreach p,$(NVC_EXE),$(eval $(call nvc_template,$(p))))
$(foreach p,$(NVD_EXE),$(eval $(call nvd_template,$(p))))
$(foreach p,$(OCL_EXE),$(eval $(call ocl_template,$(p))))
$(foreach p,$(TST_EXE),$(eval $(call tst_template,$(p))))

# Add dependencies
OBJ_DEP = $(OBJ:%.o=%.d)
-include $(OBJ_DEP)

# Rules for compiling the tools object files
$(OBJ_DIR)/nvc_%.o: tools/%.cpp 
	$(NVC) -Xcompiler=-MMD -o $@ -c $< -DUCL_CUDART

$(OBJ_DIR)/nvd_%.o: tools/%.cpp
	$(NVC) -Xcompiler=-MMD -o $@ -c $< -DUCL_CUDADR

$(OBJ_DIR)/ocl_%.o: tools/%.cpp
	$(OCL) -MMD -o $@ -c $< -DUCL_OPENCL

# Rules for compiling the example object files
$(OBJ_DIR)/nvc_%.o: examples/%.cpp 
	$(NVC) -Xcompiler=-MMD -o $@ -c $< -DUCL_CUDART

$(OBJ_DIR)/nvd_%.o: examples/%.cpp
	$(NVC) -Xcompiler=-MMD -o $@ -c $< -DUCL_CUDADR

$(OBJ_DIR)/ocl_%.o: examples/%.cpp
	$(OCL) -MMD -o $@ -c $< -DUCL_OPENCL

$(OBJ_DIR)/%.o: examples/%.cu
	$(NVC) -DNV_KERNEL -o $@ -c $<

$(OBJ_DIR)/%_bin.h: examples/%.cu
	$(NVC) -DNV_KERNEL -cubin -o $(OBJ_DIR)/$*.cubin $<
	bin2c -c -n kernel_string $(OBJ_DIR)/$*.cubin > $(OBJ_DIR)/$*_bin.h

$(OBJ_DIR)/%_str.h: examples/%.cu
	$(SH) tools/file_to_cstr.sh kernel_string examples/$*.cu $(OBJ_DIR)/$*_str.h
	
# Rules to build the test cases
$(OBJ_DIR)/%.o: test/%.cpp 
	$(CPP) -MMD $(NVC_INCS) $(OCL_INCS) -o $@ -c $<

$(BIN_DIR)/%.ptx: test/%.cu
	$(NVC) -DNV_KERNEL -DOrdinal=int -DScalar=float -ptx -o $@ $<	

$(BIN_DIR)/%_d.ptx: test/%.cu
	$(NVC) -DNV_KERNEL -DOrdinal=int -DScalar=double -ptx -o $@ $<	

$(BIN_DIR)/%.cubin: test/%.cu
	$(NVC) -DNV_KERNEL -DOrdinal=int -DScalar=float -cubin -o $@ $<	

$(BIN_DIR)/%.cu: test/%.cu
	cp $^ $@

$(OBJ_DIR)/%.o: test/%.cu
	$(NVC) -DNV_KERNEL -DOrdinal=int -DScalar=float -o $@ -c $<

test: setup $(TST_EXE)
	
# Rule for doxygen
dox:
	@echo "Running doxygen"
	$(DOX) Doxyfile

# Rules for setup, dist and clean
setup:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

clean:
	@echo "Clean"
	@rm -rf $(BIN_DIR) $(OBJ_DIR) ./doc/api geryon.*.tar.gz

VERSION = `date +'%y.%j'`

dist: clean
	@echo "Creating version $(VERSION)"
	@echo "Geryon Version $(VERSION)" > VERSION.txt
	@echo "#define GERYON_VERSION \0042$(VERSION)\0042" > $(INC_DIR)/ucl_version.h
	@tar -cz doc examples test include Makefile* > geryon.$(VERSION).tar.gz


