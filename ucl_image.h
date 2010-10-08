/***************************************************************************
                                 ucl_image.h
                             -------------------
                               W. Michael Brown

  2D Image Container

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Thu Jun 25 2009
    copyright            : (C) 2009 by W. Michael Brown
    email                : brownw@ornl.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

// Only allow this file to be included by CUDA and OpenCL specific headers
#ifdef _UCL_MAT_ALLOW

/*
/// 2D Image on device
template <class numtyp>
class UCL_Image : public UCL_BaseMat {
 public:
  // Traits for copying data
  // MEM_TYPE is 0 for device, 1 for host, and 2 for image
  enum traits {
    DATA_TYPE = _UCL_DATA_ID<numtyp>::id,
    MEM_TYPE = 2,
    PADDED = 0,
    ROW_MAJOR = 1,
    VECTOR = 0
  };
  typedef numtyp data_type; 

  UCL_ConstMat() { _rows=0; }
  ~UCL_ConstMat() { if (_rows>0) _device_image_free(*this); }

  /// Assign a texture to matrix
  inline void assign_texture(textureReference *t) { _tex_ptr=t; }  
      
  /// Row major matrix on device
  inline void safe_alloc(const size_t rows, const size_t cols) {
    _rows=rows;
    _cols=cols;

    cuda_gb_get_channel<numtyp>(_channel);
    CUDA_SAFE_CALL(cudaMallocArray(&_array, &_channel, cols, rows));
  }
  
  /// Row major matrix on device (Allocate and bind texture)
  inline void safe_alloc(const size_t rows, const size_t cols, 
                         textureReference *t) 
    { safe_alloc(rows,cols); assign_texture(t); bind(); }

  /// Bind to texture
  inline void bind() {
    (*_tex_ptr).addressMode[0] = cudaAddressModeClamp;
    (*_tex_ptr).addressMode[1] = cudaAddressModeClamp;
    (*_tex_ptr).filterMode = cudaFilterModePoint;
    (*_tex_ptr).normalized = false;
    CUDA_SAFE_CALL(cudaBindTextureToArray(_tex_ptr,_array,&_channel));
  }
  
  /// Unbind texture
  inline void unbind() { CUDA_SAFE_CALL(cudaUnbindTexture(_tex_ptr)); }
  
  /// Free any memory associated with device and unbind
  inline void clear() {
    if (_rows>0) { 
      _rows=0; 
      CUDA_SAFE_CALL(cudaUnbindTexture(_tex_ptr)); 
      CUDA_SAFE_CALL(cudaFreeArray(_array)); 
    } 
  }

  inline size_t numel() const { return _cols*_rows; }
  inline size_t rows() const { return _rows; }
  inline size_t cols() const { return _cols; }
  inline size_t row_size() const { return _cols; }
  inline size_t row_bytes() const { return _cols*sizeof(numtyp); }

 private:
  size_t _rows, _cols;
  cudaArray *_array;
  cudaChannelFormatDesc _channel;
  textureReference *_tex_ptr;
};
*/

#endif

