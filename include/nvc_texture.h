/***************************************************************************
                                nvc_texture.h
                             -------------------
                               W. Michael Brown

  Utilities for dealing with CUDA Runtime textures

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Fri Jul 2 2010
    copyright            : (C) 2010 by W. Michael Brown
    email                : brownw@ornl.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2010) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

#ifndef NVC_TEXTURE
#define NVC_TEXTURE

#include "nvc_mat.h"

namespace ucl_cudart {
    
/// Class storing a texture reference
class UCL_Texture {
 public:
  UCL_Texture() {}
  ~UCL_Texture() {}
  /// Construct with a specified texture reference
  inline UCL_Texture(textureReference *t) { get_texture(t); }
  /// Set the texture reference for this object
  inline void get_texture(textureReference *t) { _tex_ptr=t; }
  /// Set the texture reference for this object
  inline void get_texture(UCL_Program &prog, const char *texture_name) {
    CUDA_SAFE_CALL(cudaGetTextureReference((const textureReference **)&_tex_ptr,  
                                           texture_name));
  }

  /// Bind a float array where each fetch grabs a vector of length numel
  template<class numtyp>
  inline void bind_float(UCL_D_Vec<numtyp> &vec, const unsigned numel) 
    { _bind_float(vec,numel); }

  /// Bind a float array where each fetch grabs a vector of length numel
  template<class numtyp>
  inline void bind_float(UCL_D_Mat<numtyp> &vec, const unsigned numel) 
    { _bind_float(vec,numel); }

  /// Bind a float array where each fetch grabs a vector of length numel
  template<class numtyp, class devtyp>
  inline void bind_float(UCL_Vector<numtyp, devtyp> &vec, const unsigned numel) 
    { _bind_float(vec.device,numel); }

  /// Bind a float array where each fetch grabs a vector of length numel
  template<class numtyp, class devtyp>
  inline void bind_float(UCL_Matrix<numtyp, devtyp> &vec, const unsigned numel) 
    { _bind_float(vec.device,numel); }

  /// Unbind the texture reference from the memory allocation
  inline void unbind() { CUDA_SAFE_CALL(cudaUnbindTexture(_tex_ptr)); }

 private:
  textureReference *_tex_ptr;
  cudaChannelFormatDesc _channel;

  /// Bind a float array where each fetch grabs a vector of length numel
  template<class mat_typ>
  inline void _bind_float(mat_typ &vec, const unsigned numel) {
    #ifdef UCL_DEBUG
    assert(numel!=0 && numel<5);
    #endif
    int bits[4]={0,0,0,0};
    for (int i=0; i<numel; i++) bits[i]=32;
    if (vec.element_size()==sizeof(float))
      _channel = cudaCreateChannelDesc(bits[0], bits[1], bits[2], bits[3], 
                                       cudaChannelFormatKindFloat);
    else
      _channel = cudaCreateChannelDesc(bits[0], bits[1], bits[2], bits[3], 
                                       cudaChannelFormatKindSigned);
    (*_tex_ptr).addressMode[0] = cudaAddressModeClamp;
    (*_tex_ptr).addressMode[1] = cudaAddressModeClamp;
    (*_tex_ptr).filterMode = cudaFilterModePoint;
    (*_tex_ptr).normalized = false;
    CUDA_SAFE_CALL(cudaBindTexture(NULL,_tex_ptr,vec.cbegin(),&_channel));
  }
};

} // namespace

#endif

