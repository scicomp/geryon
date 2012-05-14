/***************************************************************************
                                 ucl_vector.h
                             -------------------
                               W. Michael Brown

  Vector Container on Host

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Thu Jun 25 2009
    copyright            : (C) 2009 by W. Michael Brown
    email                : brownw@ornl.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
    This software is distributed under the Simplified BSD License.
   ----------------------------------------------------------------------- */

// Only allow this file to be included by CUDA and OpenCL specific headers
#ifdef _UCL_MAT_ALLOW

/// Row Vector S-Object
template <class hosttype, class devtype>
class UCL_Vector {
 public:
  // Traits for copying data
  // MEM_TYPE is 0 for device, 1 for host, and 2 for image
  enum traits {
    DATA_TYPE = _UCL_DATA_ID<hosttype>::id,
    MEM_TYPE = 1,
    PADDED = 0,
    ROW_MAJOR = 1,
    VECTOR = 1
  };
  typedef hosttype data_type; 

  /// Host Allocation
  UCL_H_Vec<hosttype> host;
  
  /// Device Allocation
  UCL_D_Vec<devtype> device;
  
  UCL_Vector() { }
  ~UCL_Vector() { }

  /// Construct with n columns
  /** \sa alloc() **/
  UCL_Vector(const size_t n, UCL_Device &device, 
             const enum UCL_MEMOPT kind1=UCL_RW_OPTIMIZED,
             const enum UCL_MEMOPT kind2=UCL_READ_WRITE) {
    host.alloc(n,device,kind1);
    if (ucl_same_type<hosttype,devtype>::ans && device.shared_memory())
      device.view(host);
    else
      device.alloc(n,device,kind2);
  }
  
  /// Set up the vector with 'cols' columns and reserve memory
  /** The kind1 parameter controls memory pinning as follows:
    * - UCL_NOT_PINNED      - Memory is not pinned
    * - UCL_WRITE_OPTIMIZED - Memory can be pinned (write-combined)
    * - UCL_RW_OPTIMIZED    - Memory can be pinned 
    * The kind2 parameter controls memory optimizations as follows:
    * - UCL_READ_WRITE - Specify that you will read and write in kernels
    * - UCL_WRITE_ONLY - Specify that you will only write in kernels
    * - UCL_READ_ONLY  - Specify that you will only read in kernels
    * \param cq Default command queue for operations copied from another mat
    * \return UCL_SUCCESS if the memory allocation is successful **/
  template <class mat_type>
  inline int alloc(const size_t cols, mat_type &cq,
                   const enum UCL_MEMOPT kind1=UCL_RW_OPTIMIZED,
                   const enum UCL_MEMOPT kind2=UCL_READ_WRITE) {
    int e1=host.alloc(cols,cq,kind1);
    if (e1!=UCL_SUCCESS)
      return e1;
    if (ucl_same_type<hosttype,devtype>::ans && device.shared_memory()) {
      device.view(host);
      return UCL_SUCCESS;
    } else
      return device.alloc(cols,cq,kind2); 
  }

  /// Set up host vector with 'cols' columns and reserve memory
  /** The kind1 parameter controls memory pinning as follows:
    * - UCL_NOT_PINNED      - Memory is not pinned
    * - UCL_WRITE_OPTIMIZED - Memory can be pinned (write-combined)
    * - UCL_RW_OPTIMIZED    - Memory can be pinned 
    * The kind2 parameter controls memory optimizations as follows:
    * - UCL_READ_WRITE - Specify that you will read and write in kernels
    * - UCL_WRITE_ONLY - Specify that you will only write in kernels
    * - UCL_READ_ONLY  - Specify that you will only read in kernels
    * \param device Used to get the default command queue for operations
    * \return UCL_SUCCESS if the memory allocation is successful **/
  inline int alloc(const size_t cols, UCL_Device &device,
                   const enum UCL_MEMOPT kind1=UCL_RW_OPTIMIZED,
                   const enum UCL_MEMOPT kind2=UCL_READ_WRITE) {
    int e1=host.alloc(cols,device,kind1);
    if (e1!=UCL_SUCCESS)
      return e1;
    if (ucl_same_type<hosttype,devtype>::ans && device.shared_memory()) {
      device.view(host);
      return UCL_SUCCESS;
    } else
      return device.alloc(cols,device,kind2); 
  }
  
  /// Free memory and set size to 0
  inline void clear() 
    { host.clear(); device.clear();}}

  /// Set each element to zero
  inline void zero() { host.zero(); device.zero(); }
  
  /// Set first n elements to zero
  inline void zero(const int n) { host.zero(n); device.zero(n); }

  /// Get the number of elements
  inline size_t numel() const { return host.numel(); }
  /// Get the number of rows
  inline size_t rows() const { return host.rows(); }
  /// Get the number of columns
  inline size_t cols() const { return host.cols(); }
    
  /// Get element at index i
  inline hosttype & operator[](const int i) { return host[i]; }
  /// Get element at index i
  inline const hosttype & operator[](const int i) const { return host[i]; }
  /// 2D access (row should always be 0) 
  inline hosttype & operator()(const int row, const int col) 
    { return _host[col]; }
  /// 2D access (row should always be 0) 
  inline const hosttype & operator()(const int row, const int col) const
    { return host[col]; }
  
  /// Returns pointer to memory pointer for allocation on host
  inline hosttype ** host_ptr() { return host.host_ptr(); }
  
  /// Return the default command queue/stream associated with this data
  inline command_queue & cq() { return host.cq(); }
  /// Block until command_queue associated with matrix is complete
  inline void sync() { host.sync(); }

 private:
};

#endif

