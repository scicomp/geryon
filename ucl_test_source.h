/***************************************************************************
                              ucl_test_source.h
                             -------------------
                               W. Michael Brown

  Test code for UCL under any of the namespaces.

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Wed Jan 28 2009
    copyright            : (C) 2009 by W. Michael Brown
    email                : wmbrown@sandia.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2009) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

  // -------------------------------------------------------------------------
  // - DEVICE TESTS
  // -------------------------------------------------------------------------
  UCL_Device cop;
  cerr << "Found " << cop.num_platforms() << " platform(s).\n";
  cerr << "Using platform: " << cop.platform_name() << endl;
  cerr << "Found " << cop.num_devices() << " device(s).\n";
  cerr << "Setting to device 0...";
  cop.set(0);
  cerr << "Done.\n";
  cerr << "Device 0 is called " << cop.name() << endl;
  cerr << "Device 0 has " << cop.gigabytes() << "GB of memory\n";
  cerr << "Adding a command queue to device...";
  cop.push_command_queue();
  assert(cop.num_queues()==2);
  cerr << "Done.\n";
  cerr << "Syncing command queues...";
  ucl_sync(cop.cq());
  ucl_sync(cop.cq(0));
  ucl_sync(cop.cq(1));
  cop.sync();
  cop.sync(1);
  cerr << "Done.\n";
  cerr << "Removing command queue from device...";
  cop.pop_command_queue();
  cerr << "Done.\n";
  if (sizeof(numtyp)==sizeof(double) && cop.double_precision()==false) {
    cerr << "DOUBLE PRECISION TESTS WILL NOT BE PERFORMED ON THIS DEVICE.\n"
         << "DONE.\n";
    return;
  }
  cerr << "Starting device timer...";
  UCL_Timer timer;
  timer.init(cop);
  timer.start();
  cerr << "Done.\n";
  
  // -------------------------------------------------------------------------
  // - HOST VECTOR ALLOCATION TESTS
  // -------------------------------------------------------------------------
  cerr << "Creating/destroying an empty host vector...";
  { UCL_H_Vec<numtyp> tmat; }
  cerr << "Done.\n";
  
  cerr << "Create/fill/destroy (5) host vector (not pinned)...";
  { 
    UCL_H_Vec<numtyp> tmat(5,cop);
    fill_test(tmat,5); 
    UCL_H_Vec<numtyp> rmat; 
    rmat.alloc(5,cop); 
    fill_test(rmat,5);
  }
  cerr << "Done.\n";
 
  cerr << "Create/fill/destroy (5) host vector (write-combined)...";
  { 
    UCL_H_Vec<numtyp> tmat(5,cop,UCL_WRITE_OPTIMIZED);
    fill_test(tmat,5); 
    UCL_H_Vec<numtyp> rmat; 
    rmat.alloc(5,cop,UCL_WRITE_OPTIMIZED);
    fill_test(tmat,5); 
  }
  cerr << "Done.\n";
  
  cerr << "Create/fill/destroy (5) host vector (pinned)...";
  { 
    UCL_H_Vec<numtyp> tmat(5,cop,UCL_RW_OPTIMIZED);
    fill_test(tmat,5); 
    UCL_H_Vec<numtyp> rmat;
    rmat.alloc(5,cop,UCL_RW_OPTIMIZED);
    fill_test(rmat,5); 
  }
  cerr << "Done.\n";
  
  // -------------------------------------------------------------------------
  // - HOST MATRIX ALLOCATION TESTS
  // -------------------------------------------------------------------------
  cerr << "Creating/destroying an empty host matrix...";
  { UCL_H_Mat<numtyp> tmat; }
  cerr << "Done.\n";
  
  cerr << "Create/fill/destroy (2,3) host matrix (not pinned)...";
  { 
    UCL_H_Mat<numtyp> tmat(2,3,cop); 
    fill_test(tmat,6);
    fill_test(tmat,2,3); 
    UCL_H_Mat<numtyp> rmat; 
    rmat.alloc(2,3,cop); 
    fill_test(rmat,6); 
    fill_test(rmat,2,3); 
  }
  cerr << "Done.\n";
 
  cerr << "Create/fill/destroy (2,3) host matrix (write-combined)...";
  { 
    UCL_H_Mat<numtyp> tmat(2,3,cop,UCL_WRITE_OPTIMIZED); 
    fill_test(tmat,6); 
    fill_test(tmat,2,3); 
    UCL_H_Mat<numtyp> rmat; 
    rmat.alloc(2,3,cop,UCL_WRITE_OPTIMIZED); 
    fill_test(rmat,6); 
    fill_test(rmat,2,3); 
  }
  cerr << "Done.\n";
  
  cerr << "Create/fill/destroy (2,3) host matrix (pinned)...";
  { 
    UCL_H_Mat<numtyp> tmat(2,3,cop,UCL_RW_OPTIMIZED); 
    fill_test(tmat,6); 
    fill_test(tmat,2,3); 
    UCL_H_Mat<numtyp> rmat; 
    rmat.alloc(2,3,cop,UCL_RW_OPTIMIZED); 
    fill_test(rmat,6); 
    fill_test(rmat,2,3); 
  }
  cerr << "Done.\n";
  
  // -------------------------------------------------------------------------
  // - ALLOCATE SOME MATRICES FOR COPY TESTS
  // -------------------------------------------------------------------------
  UCL_H_Vec<float> count_vec4_single(4,cop,UCL_RW_OPTIMIZED);
  UCL_H_Vec<float> count_4_single(4,cop,UCL_RW_OPTIMIZED);
  for (int i=0; i<4; i++) count_vec4_single[i]=static_cast<float>(i);
  UCL_H_Vec<double> count_vec4_double(4,cop,UCL_RW_OPTIMIZED);
  UCL_H_Vec<double> count_4_double(4,cop,UCL_RW_OPTIMIZED);;
  for (int i=0; i<4; i++) count_vec4_double[i]=static_cast<double>(i);

  UCL_H_Vec<float> count_vec_single(6,cop,UCL_RW_OPTIMIZED);
  UCL_H_Vec<float> count_6_single(6,cop,UCL_RW_OPTIMIZED);
  for (int i=0; i<6; i++) count_vec_single[i]=static_cast<float>(i);
  UCL_H_Vec<double> count_vec_double(6,cop,UCL_RW_OPTIMIZED);
  UCL_H_Vec<double> count_6_double(6,cop,UCL_RW_OPTIMIZED);
  for (int i=0; i<6; i++) count_vec_double[i]=static_cast<double>(i);
  UCL_H_Vec<int> count_vec_int(6,cop,UCL_RW_OPTIMIZED);
  UCL_H_Vec<int> count_6_int(6,cop,UCL_RW_OPTIMIZED);
  for (int i=0; i<6; i++) count_vec_int[i]=static_cast<int>(i);

  UCL_H_Mat<float> count_mat_single(2,3,cop,UCL_RW_OPTIMIZED);
  UCL_H_Mat<float> count_23_single(2,3,cop,UCL_RW_OPTIMIZED);
  for (int i=0; i<6; i++) count_mat_single[i]=static_cast<float>(i);
  UCL_H_Mat<double> count_mat_double(2,3,cop,UCL_RW_OPTIMIZED);
  UCL_H_Mat<double> count_23_double(2,3,cop,UCL_RW_OPTIMIZED);
  for (int i=0; i<6; i++) count_mat_double[i]=static_cast<double>(i);
  UCL_H_Mat<int> count_mat_int(2,3,cop,UCL_RW_OPTIMIZED);
  UCL_H_Mat<int> count_23_int(2,3,cop,UCL_RW_OPTIMIZED);
  for (int i=0; i<6; i++) count_mat_int[i]=static_cast<int>(i);

  // -------------------------------------------------------------------------
  // - ALLOCATE SOME MATRICES FOR VIEW TESTS
  // -------------------------------------------------------------------------
  numtyp h_view_cpp1[9]={ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  numtyp h_view_cpp2[9];
  UCL_H_Vec<numtyp> h_view_ucl1(9,cop), h_view_ucl2(9,cop);
  for (int i=0; i<9; i++) h_view_ucl1[i]=static_cast<numtyp>(i+1);
  UCL_H_Mat<numtyp> h_view_ucl3(3,3,cop), h_view_ucl4(3,3,cop);
  for (int i=0; i<9; i++) h_view_ucl3[i]=static_cast<numtyp>(i+1);   
  
  // -------------------------------------------------------------------------
  // - HOST VECTOR TESTS
  // -------------------------------------------------------------------------
  cerr << "Host (6) vector tests.\n";
  {
    cerr << "  Constructor allocation...";
    UCL_H_Vec<numtyp> mat(6,cop);
    cerr << "Done.\n";
    cerr << "  Fill test...";
    fill_test(mat,6);
    cerr << "Done.\n";
    cerr << "  Clearing...";
    mat.clear();
    cerr << "Done.\n";
    cerr << "  Member allocation...";
    mat.alloc(6,cop);
    cerr << "Done.\n";
    cerr << "  Fill test...";
    fill_test(mat,6);
    cerr << "Done.\n";
    cerr << "  Zeroing...";
    mat.zero();
    for (int i=0; i<6; i++) assert(mat[i]==static_cast<numtyp>(0));
    cerr << "Done.\n";
    cerr << "  Member tests...";
    assert(mat.numel()==6 && mat.rows()==1 && mat.cols()==6);
    assert(mat.row_size()==6 && mat.row_bytes()==6*sizeof(numtyp));
    cerr << "Done.\n";
    cerr << "  Allocation with mat instead of device...";
    UCL_H_Vec<numtyp> mat2;
    mat2.alloc(6,mat);
    cerr << "Done.\n";
    
    cerr << "  Copy single precision vector into...";
    ucl_copy(mat,count_vec_single,async);
    mat.sync();
    for (int i=0; i<6; i++) assert(mat[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    mat.zero();
    cerr << "  Copy double precision vector into...";
    ucl_copy(mat,count_vec_double,async);
    mat.sync();
    for (int i=0; i<6; i++) assert(mat[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    mat.zero();
    cerr << "  Copy int vector into...";
    ucl_copy(mat,count_vec_int,async);
    mat.sync();
    for (int i=0; i<6; i++) assert(mat[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    cerr << "  Copy into single precision vector...";
    count_6_single.zero();
    ucl_copy(count_6_single,mat,async);
    count_6_single.sync();
    for (int i=0; i<6; i++) assert(count_6_single[i]==static_cast<float>(i));
    cerr << "Done.\n";
    cerr << "  Copy into double precision vector...";
    count_6_double.zero();
    ucl_copy(count_6_double,mat,async);
    count_6_double.sync();
    for (int i=0; i<6; i++) assert(count_6_double[i]==static_cast<double>(i));
    cerr << "Done.\n";
    cerr << "  Copy into int precision vector...";
    count_6_int.zero();
    ucl_copy(count_6_int,mat,async);
    count_6_int.sync();
    for (int i=0; i<6; i++) assert(count_6_int[i]==static_cast<int>(i));
    cerr << "Done.\n";
    cerr << "  Print test...";
    ostringstream out1,out2,out3,out4,out5;
    ucl_print(mat,out1);
    assert(out1.str()=="0 1 2 3 4 5");
    cerr << "Done.\n";
    cerr << "  Print as mat test...";
    ucl_print(mat,2,2,out2);
    assert(out2.str()=="0 1\n2 3");
    cerr << "Done.\n";
    cerr << "  Print 4 test...";
    ucl_print(mat,4,out3);
    assert(out3.str()=="0 1 2 3");
    cerr << "Done.\n";
    cerr << "  Print with device test...";
    ucl_print(mat,out4,cop);
    assert(out4.str()=="0 1 2 3 4 5");
    cerr << "Done.\n";
    cerr << "  Operator << test...";
    out5 << mat;
    assert(out5.str()=="0 1 2 3 4 5");
    cerr << "Done.\n";
    
    {
    ostringstream v1,v2,v3,v4,v5,v6,v7;
    cerr << "  Full vector view...";
    UCL_H_Vec<numtyp> view1;
    view1.view(h_view_ucl1);
    v1 << view1;
    assert(v1.str()=="1 2 3 4 5 6 7 8 9");
    cerr << "Done.\n";
    cerr << "  Partial vector view...";
    view1.view(h_view_ucl1,6);
    v2 << view1;
    assert(v2.str()=="1 2 3 4 5 6");
    view1.view(h_view_ucl1,1,4);
    v3 << view1;
    assert(v3.str()=="1 2 3 4");
    cerr << "Done.\n";
    cerr << "  Full matrix view...";
    view1.view(h_view_ucl3);
    v4 << view1;
    assert(v4.str()=="1 2 3 4 5 6 7 8 9");
    cerr << "Done.\n";
    cerr << "  Partial matrix view...";
    view1.view(h_view_ucl3,6);
    v5 << view1;
    assert(v5.str()=="1 2 3 4 5 6");
    view1.view(h_view_ucl3,1,4);
    v6 << view1;
    assert(v6.str()=="1 2 3 4");
    cerr << "Done.\n";
    h_view_ucl2.zero();
    cerr << "  Copy into partial view...";
    view1.view(h_view_ucl2,6);
    ucl_copy(view1,h_view_ucl1,6,async);
    view1.sync();
    for (int i=0; i<6; i++)
      assert(h_view_ucl2[i]==static_cast<numtyp>(i+1));
    cerr << "Done.\n";
    cerr << "  Partial vector view from pointer...";
    view1.view(h_view_cpp1,6,cop);
    v7 << view1;
    assert(v7.str()=="1 2 3 4 5 6");
    cerr << "Done.\n";
    }
    
    {
    ostringstream v1,v2,v3,v4,v5,v6,v7;
    cerr << "  Offset full vector view...";
    UCL_H_Vec<numtyp> view1;
    view1.view_offset(3,h_view_ucl1);
    v1 << view1;
    assert(v1.str()=="4 5 6 7 8 9");
    cerr << "Done.\n";
    cerr << "  Offset partial vector view...";
    view1.view_offset(3,h_view_ucl1,3);
    v2 << view1;
    assert(v2.str()=="4 5 6");
    view1.view_offset(2,h_view_ucl1,1,2);
    v3 << view1;
    assert(v3.str()=="3 4");
    cerr << "Done.\n";
    cerr << "  Offset full matrix view...";
    view1.view_offset(3,h_view_ucl3);
    v4 << view1;
    assert(v4.str()=="4 5 6 7 8 9");
    cerr << "Done.\n";
    cerr << "  Offset partial matrix view...";
    view1.view_offset(3,h_view_ucl3,3);
    v5 << view1;
    assert(v5.str()=="4 5 6");
    view1.view_offset(2,h_view_ucl3,1,2);
    v6 << view1;
    assert(v6.str()=="3 4");
    cerr << "Done.\n";
    h_view_ucl2.zero();
    cerr << "  Copy into offset partial view...";
    view1.view_offset(3,h_view_ucl2,2);
    ucl_copy(view1,h_view_ucl1,2,async);
    view1.sync();
    for (int i=3; i<5; i++)
      assert(h_view_ucl2[i]==static_cast<numtyp>(i+1-3));
    cerr << "Done.\n";
    cerr << "  Offset partial vector view from pointer...";
    view1.view_offset(3,h_view_cpp1,3,cop);
    v7 << view1;
    assert(v7.str()=="4 5 6");
    cerr << "Done.\n";
    }
        
    cerr << "  Destructing...";   
  }
  cerr << "Done.\n";
  
  // -------------------------------------------------------------------------
  // - HOST MATRIX TESTS
  // -------------------------------------------------------------------------
  cerr << "Host (2,3) matrix tests.\n";
  {
    cerr << "  Constructor allocation...";
    UCL_H_Mat<numtyp> mat(2,3,cop);
    cerr << "Done.\n";
    cerr << "  Fill test...";
    fill_test(mat,6);
    fill_test(mat,2,3);
    cerr << "Done.\n";
    cerr << "  Clearing...";
    mat.clear();
    cerr << "Done.\n";
    cerr << "  Member allocation...";
    mat.alloc(2,3,cop);
    cerr << "Done.\n";
    cerr << "  Fill test...";
    fill_test(mat,6);
    fill_test(mat,2,3);
    cerr << "Done.\n";
    cerr << "  Zeroing...";
    mat.zero();
    for (int i=0; i<6; i++) assert(mat[i]==static_cast<numtyp>(0));
    cerr << "Done.\n";
    cerr << "  Member tests...";
    assert(mat.numel()==6 && mat.rows()==2 && mat.cols()==3);
    assert(mat.row_size()==3 && mat.row_bytes()==3*sizeof(numtyp));
    cerr << "Done.\n";
    cerr << "  Allocation with mat instead of device...";
    UCL_H_Mat<numtyp> mat2;
    mat2.alloc(2,3,mat);
    cerr << "Done.\n";
    cerr << "  Destructing...";

    cerr << "  Copy single precision vector into...";
    ucl_copy(mat,count_vec_single,async);
    mat.sync();
    for (int i=0; i<6; i++) assert(mat[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    mat.zero();
    cerr << "  Copy double precision vector into...";
    ucl_copy(mat,count_vec_double,async);
    mat.sync();
    for (int i=0; i<6; i++) assert(mat[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    mat.zero();
    cerr << "  Copy int vector into...";
    ucl_copy(mat,count_vec_int,async);
    mat.sync();
    for (int i=0; i<6; i++) assert(mat[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    cerr << "  Copy into single precision vector...";
    count_6_single.zero();
    ucl_copy(count_6_single,mat,async);
    count_6_single.sync();
    for (int i=0; i<6; i++) assert(count_6_single[i]==static_cast<float>(i));
    cerr << "Done.\n";
    cerr << "  Copy into double precision vector...";
    count_6_double.zero();
    ucl_copy(count_6_double,mat,async);
    count_6_double.sync();
    for (int i=0; i<6; i++) assert(count_6_double[i]==static_cast<double>(i));
    cerr << "Done.\n";
    cerr << "  Copy into int precision vector...";
    count_6_int.zero();
    ucl_copy(count_6_int,mat,async);
    count_6_int.sync();
    for (int i=0; i<6; i++) assert(count_6_int[i]==static_cast<int>(i));
    cerr << "Done.\n";
    
    cerr << "  Copy single precision matrix into...";
    ucl_copy(mat,count_mat_single,async);
    mat.sync();
    for (int i=0; i<6; i++) assert(mat[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    mat.zero();
    cerr << "  Copy double precision matrix into...";
    ucl_copy(mat,count_mat_double,async);
    mat.sync();
    for (int i=0; i<6; i++) assert(mat[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    mat.zero();
    cerr << "  Copy int matrix into...";
    ucl_copy(mat,count_mat_int,async);
    mat.sync();
    for (int i=0; i<6; i++) assert(mat[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    cerr << "  Copy into single precision matrix...";
    count_23_single.zero();
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<6; i++) assert(count_23_single[i]==static_cast<float>(i));
    cerr << "Done.\n";
    cerr << "  Copy into double precision matrix...";
    count_23_double.zero();
    ucl_copy(count_23_double,mat,async);
    count_23_double.sync();
    for (int i=0; i<6; i++) assert(count_23_double[i]==static_cast<double>(i));
    cerr << "Done.\n";
    cerr << "  Copy into int precision matrix...";
    count_23_int.zero();
    ucl_copy(count_23_int,mat,async);
    count_23_int.sync();
    for (int i=0; i<6; i++) assert(count_23_int[i]==static_cast<int>(i));
    cerr << "Done.\n";

    mat.zero();
    cerr << "  Copy single precision vector into slice...";
    ucl_copy(mat,count_vec4_single,2,2,async);
    mat.sync();
    for (int i=0; i<2; i++) assert(mat(0,i)==static_cast<numtyp>(i));
    for (int i=0; i<2; i++) assert(mat(1,i)==static_cast<numtyp>(i+2));
    cerr << "Done.\n";
    mat.zero();
    cerr << "  Copy double precision vector into slice...";
    ucl_copy(mat,count_vec4_double,2,2,async);
    mat.sync();
    for (int i=0; i<2; i++) assert(mat(0,i)==static_cast<numtyp>(i));
    for (int i=0; i<2; i++) assert(mat(1,i)==static_cast<numtyp>(i+2));
    cerr << "Done.\n";
    mat.zero();
    cerr << "  Copy single precision vector slice into slice...";
    ucl_copy(mat,count_vec_single,2,2,async);
    mat.sync();
    for (int i=0; i<2; i++) assert(mat(0,i)==static_cast<numtyp>(i));
    for (int i=0; i<2; i++) assert(mat(1,i)==static_cast<numtyp>(i+2));
    cerr << "Done.\n";
    mat.zero();
    cerr << "  Copy double precision vector slice into slice...";
    ucl_copy(mat,count_vec_double,2,2,async);
    mat.sync();
    for (int i=0; i<2; i++) assert(mat(0,i)==static_cast<numtyp>(i));
    for (int i=0; i<2; i++) assert(mat(1,i)==static_cast<numtyp>(i+2));
    cerr << "Done.\n";

    ucl_copy(mat,count_mat_single,async);
    mat.sync();
    
    count_4_single.zero();
    cerr << "  Copy slice into single precision vector...";
    ucl_copy(count_4_single,mat,2,2,async);
    count_4_single.sync();
    for (int i=0; i<2; i++) assert(count_4_single[i]==static_cast<float>(i));
    for (int i=2; i<4; i++) assert(count_4_single[i]==static_cast<float>(i+1));
    cerr << "Done.\n";
    count_4_double.zero();
    cerr << "  Copy double precision vector into...";
    ucl_copy(count_4_double,mat,2,2,async);
    count_4_double.sync();
    for (int i=0; i<2; i++) assert(count_4_double[i]==static_cast<double>(i));
    for (int i=2; i<4; i++) assert(count_4_double[i]==static_cast<double>(i+1));
    cerr << "Done.\n";

    mat.zero();
    cerr << "  Copy single precision slice into slice...";
    ucl_copy(mat,count_mat_single,2,2,async);
    mat.sync();
    for (int i=0; i<2; i++) assert(mat(0,i)==static_cast<numtyp>(i));
    for (int i=0; i<2; i++) assert(mat(1,i)==static_cast<numtyp>(i+3));
    cerr << "Done.\n";
    mat.zero();
    cerr << "  Copy double precision slice into...";
    ucl_copy(mat,count_mat_double,2,2,async);
    mat.sync();
    for (int i=0; i<2; i++) assert(mat(0,i)==static_cast<numtyp>(i));
    for (int i=0; i<2; i++) assert(mat(1,i)==static_cast<numtyp>(i+3));
    cerr << "Done.\n";

    ucl_copy(mat,count_mat_single,async);
    mat.sync();

    cerr << "  Print test...";
    ostringstream out1,out2,out3,out4,out5;
    ucl_print(mat,out1);
    assert(out1.str()=="0 1 2\n3 4 5");
    cerr << "Done.\n";
    cerr << "  Print slice test...";
    ucl_print(mat,2,2,out2);
    assert(out2.str()=="0 1\n3 4");
    cerr << "Done.\n";
    cerr << "  Print vector slice test...";
    ucl_print(mat,5,out3);
    assert(out3.str()=="0 1 2 3 4");
    cerr << "Done.\n";
    cerr << "  Print with device test...";
    ucl_print(mat,out4,cop);
    assert(out4.str()=="0 1 2\n3 4 5");
    cerr << "Done.\n";
    cerr << "  Operator << test...";
    out5 << mat;
    assert(out5.str()=="0 1 2\n3 4 5");
    cerr << "Done.\n";
    
    {
    ostringstream v1,v2,v3,v4,v5,v6,v7,v8;
    cerr << "  Full vector view...";
    UCL_H_Mat<numtyp> view1;
    view1.view(h_view_ucl1);
    v1 << view1;
    assert(v1.str()=="1 2 3 4 5 6 7 8 9");
    cerr << "Done.\n";
    cerr << "  Partial vector view...";
    view1.view(h_view_ucl1,6);
    v2 << view1;
    assert(v2.str()=="1 2 3 4 5 6");
    view1.view(h_view_ucl1,1,4);
    v3 << view1;
    assert(v3.str()=="1 2 3 4");
    cerr << "Done.\n";
    cerr << "  Full matrix view...";
    view1.view(h_view_ucl3);
    v4 << view1;
    assert(v4.str()=="1 2 3\n4 5 6\n7 8 9");
    cerr << "Done.\n";
    cerr << "  Partial matrix view...";
    view1.view(h_view_ucl3,2,3);
    v5 << view1;
    assert(v5.str()=="1 2 3\n4 5 6");
    view1.view(h_view_ucl3,1,2);
    v6 << view1;
    assert(v6.str()=="1 2");
    cerr << "Done.\n";
    h_view_ucl2.zero();
    cerr << "  Copy into partial view...";
    view1.view(h_view_ucl2,6);
    ucl_copy(view1,h_view_ucl1,6,async);
    view1.sync();
    for (int i=0; i<6; i++)
      assert(h_view_ucl2[i]==static_cast<numtyp>(i+1));
    cerr << "Done.\n";
    cerr << "  Partial vector view from pointer...";
    view1.view(h_view_cpp1,6,cop);
    v7 << view1;
    assert(v7.str()=="1 2 3 4 5 6");
    cerr << "Done.\n";
    cerr << "  Partial matrix view from pointer...";
    view1.view(h_view_cpp1,3,2,cop);
    v8 << view1;
    assert(v8.str()=="1 2\n3 4\n5 6");
    cerr << "Done.\n";
    }
    
    {
    ostringstream v1,v2,v3,v4,v5,v6,v7,v8;
    cerr << "  Offset full vector view...";
    UCL_H_Mat<numtyp> view1;
    view1.view_offset(3,h_view_ucl1);
    v1 << view1;
    assert(v1.str()=="4 5 6 7 8 9");
    cerr << "Done.\n";
    cerr << "  Offset partial vector view...";
    view1.view_offset(3,h_view_ucl1,3);
    v2 << view1;
    assert(v2.str()=="4 5 6");
    view1.view_offset(2,h_view_ucl1,1,2);
    v3 << view1;
    assert(v3.str()=="3 4");
    cerr << "Done.\n";
    cerr << "  Offset full matrix view...";
    view1.view_offset(3,h_view_ucl3);
    v4 << view1;
    assert(v4.str()=="4 5 6\n7 8 9");
    cerr << "Done.\n";
    cerr << "  Offset partial matrix view...";
    view1.view_offset(3,h_view_ucl3,3);
    v5 << view1;
    assert(v5.str()=="4 5 6");
    view1.view_offset(2,h_view_ucl3,1,2);
    v6 << view1;
    assert(v6.str()=="3 4");
    cerr << "Done.\n";
    h_view_ucl2.zero();
    cerr << "  Copy into offset partial view...";
    view1.view_offset(3,h_view_ucl2,2);
    ucl_copy(view1,h_view_ucl1,2,async);
    view1.sync();
    for (int i=3; i<5; i++)
      assert(h_view_ucl2[i]==static_cast<numtyp>(i+1-3));
    cerr << "Done.\n";
    cerr << "  Offset partial vector view from pointer...";
    view1.view_offset(3,h_view_cpp1,3,cop);
    v7 << view1;
    assert(v7.str()=="4 5 6");
    cerr << "Done.\n";
    cerr << "  Offset partial matrix view from pointer...";
    view1.view_offset(3,h_view_cpp1,2,2,cop);
    v8 << view1;
    assert(v8.str()=="4 5\n6 7");
    cerr << "Done.\n";
    }
    
    cerr << "  Destructing..."; 
  }
  cerr << "Done.\n";

  // -------------------------------------------------------------------------
  // - DEVICE VECTOR TESTS
  // -------------------------------------------------------------------------
  cerr << "Device (6) vector tests.\n";
  {
    cerr << "  Constructor allocation...";
    UCL_D_Vec<numtyp> mat(6,cop);
    cerr << "Done.\n";
    cerr << "  Clearing...";
    mat.clear();
    cerr << "Done.\n";
    cerr << "  Member allocation...";
    mat.alloc(6,cop);
    cerr << "Done.\n";
    cerr << "  Zeroing...";
    mat.zero();
    cerr << "Done.\n";
    cerr << "  Member tests...";
    assert(mat.numel()==6 && mat.rows()==1 && mat.cols()==6);
    assert(mat.row_size()==6 && mat.row_bytes()==6*sizeof(numtyp));
    cerr << "Done.\n";
    cerr << "  Allocation with mat instead of device...";
    UCL_D_Vec<numtyp> mat2;
    mat2.alloc(6,mat);
    cerr << "Done.\n";

    mat.zero();
    count_6_single.zero();
    cerr << "  Copy single precision vector into...";
    ucl_copy(mat,count_vec_single,async);
    mat.sync();
    ucl_copy(count_6_single,mat,async);
    count_6_single.sync();
    for (int i=0; i<6; i++) assert(count_6_single[i]==static_cast<float>(i));
    cerr << "Done.\n";
    cerr << "  Zero test...";
    mat.zero();
    ucl_copy(count_6_single,mat,async);
    count_6_single.sync();
    for (int i=0; i<6; i++) assert(count_6_single[i]==static_cast<float>(0));       
    cerr << "Done.\n";
    count_6_double.zero();
    cerr << "  Copy double precision vector into...";
    ucl_copy(mat,count_vec_double,async);
    ucl_copy(count_6_double,mat,async);
    count_6_double.sync();
    for (int i=0; i<6; i++) assert(count_6_double[i]==static_cast<double>(i));
    cerr << "Done.\n";
    mat.zero();
    count_6_int.zero();
    cerr << "  Copy int vector into...";
    ucl_copy(mat,count_vec_int,async);
    ucl_copy(count_6_int,mat,async);
    count_6_int.sync();
    for (int i=0; i<6; i++) assert(count_6_int[i]==static_cast<int>(i));
    cerr << "Done.\n";
    
    mat.zero();
    count_23_single.zero();
    cerr << "  Copy single precision matrix into...";
    ucl_copy(mat,count_mat_single,async);
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<6; i++) assert(count_23_single[i]==static_cast<float>(i));
    cerr << "Done.\n";
    mat.zero();
    count_23_double.zero();
    cerr << "  Copy double precision matrix into...";
    ucl_copy(mat,count_mat_double,async);
    ucl_copy(count_23_double,mat,async);
    count_23_double.sync();
    for (int i=0; i<6; i++) assert(count_23_double[i]==static_cast<double>(i));
    cerr << "Done.\n";
    mat.zero();
    count_23_int.zero();
    cerr << "  Copy int matrix into...";
    ucl_copy(mat,count_mat_int,async);
    ucl_copy(count_23_int,mat,async);
    count_23_int.sync();
    for (int i=0; i<6; i++) assert(count_23_int[i]==static_cast<int>(i));
    cerr << "Done.\n";

    mat.zero();
    count_6_single.zero();
    cerr << "  Copy single precision matrix slice into slice...";
    ucl_copy(mat,count_mat_single,2,2,async);
    ucl_copy(count_6_single,mat,async);
    count_6_single.sync();
    for (int i=0; i<2; i++) assert(count_6_single[i]==static_cast<float>(i));
    for (int i=0; i<2; i++) assert(count_6_single[i+2]==
                                   static_cast<float>(i+3));
    cerr << "Done.\n";
    mat.zero();
    count_6_double.zero();
    cerr << "  Copy double precision matrix slice into slice...";
    ucl_copy(mat,count_mat_double,2,2,async);
    ucl_copy(count_6_double,mat,async);
    count_6_double.sync();
    for (int i=0; i<2; i++) assert(count_6_double[i]==static_cast<double>(i));
    for (int i=0; i<2; i++) assert(count_6_double[i+2]==
                                   static_cast<double>(i+3));
    cerr << "Done.\n";

    ucl_copy(mat,count_vec_single,async);

    count_23_single.zero();
    cerr << "  Copy into single precision matrix slice...";
    ucl_copy(count_23_single,mat,2,2,async);
    count_23_single.sync();
    for (int i=0; i<2; i++) assert(count_23_single[i]==static_cast<float>(i));
    for (int i=0; i<2; i++) assert(count_23_single[i+3]==
                                   static_cast<float>(i+2));
    cerr << "Done.\n";
    count_23_double.zero();
    cerr << "  Copy into double precision matrix slice...";
    ucl_copy(count_23_double,mat,2,2,async);
    count_23_double.sync();
    for (int i=0; i<2; i++) assert(count_23_double[i]==static_cast<double>(i));
    for (int i=0; i<2; i++) assert(count_23_double[i+3]==
                                   static_cast<double>(i+2));
    cerr << "Done.\n";
    
    count_6_single.zero();
    cerr << "  Copy into another device vector...";
    ucl_copy(mat2,mat,async);
    ucl_copy(count_6_single,mat2,async);
    count_6_single.sync();
    for (int i=0; i<6; i++) assert(count_6_single[i]==static_cast<float>(i));
    cerr << "Done.\n";
    
    count_6_single.zero();
    cerr << "  Copy into another device vector slice...";
    ucl_copy(mat2,mat,4,async);
    ucl_copy(count_6_single,mat2,async);
    count_6_single.sync();
    for (int i=0; i<4; i++) assert(count_6_single[i]==static_cast<float>(i));
    cerr << "Done.\n";

    cerr << "  Print test...";
    ostringstream out1,out2,out3,out4,out5;
    ucl_print(mat,out1);
    assert(out1.str()=="0 1 2 3 4 5");
    cerr << "Done.\n";
    cerr << "  Print as mat test...";
    ucl_print(mat,2,2,out2);
    assert(out2.str()=="0 1\n2 3");
    cerr << "Done.\n";
    cerr << "  Print 4 test...";
    ucl_print(mat,4,out3);
    assert(out3.str()=="0 1 2 3");
    cerr << "Done.\n";

    cerr << "  Print with device test...";
    ucl_print(mat,out4,cop);
    assert(out4.str()=="0 1 2 3 4 5");
    cerr << "Done.\n";
    cerr << "  Operator << test...";
    out5 << mat;
    assert(out5.str()=="0 1 2 3 4 5");
    cerr << "Done.\n";

    
    // -------------------------------------------------------------------------
    // - ALLOCATE SOME MATRICES FOR VIEW TESTS
    // -------------------------------------------------------------------------
    UCL_D_Vec<numtyp> d_view_ucl1(9,cop), d_view_ucl2(9,cop);
    ucl_copy(d_view_ucl1,h_view_ucl1,false);
    UCL_D_Mat<numtyp> d_view_ucl3(3,3,cop), d_view_ucl4(3,3,cop);
    ucl_copy(d_view_ucl3,h_view_ucl3,false);

    {
    ostringstream v1,v2,v3,v4,v5,v6,v7;
    cerr << "  Full vector view...";
    UCL_D_Vec<numtyp> view1;
    view1.view(d_view_ucl1);
    v1 << view1;
    assert(v1.str()=="1 2 3 4 5 6 7 8 9");
    cerr << "Done.\n";
    cerr << "  Partial vector view...";
    view1.view(d_view_ucl1,6);
    v2 << view1;
    assert(v2.str()=="1 2 3 4 5 6");
    view1.view(d_view_ucl1,1,4);
    v3 << view1;
    assert(v3.str()=="1 2 3 4");
    cerr << "Done.\n";
    cerr << "  Partial matrix view...";
    view1.view(d_view_ucl3,3);
    v5 << view1;
    assert(v5.str()=="1 2 3");
    view1.view(d_view_ucl3,1,2);
    v6 << view1;
    assert(v6.str()=="1 2");
    cerr << "Done.\n";
    d_view_ucl2.zero();
    cerr << "  Copy into partial view...";
    view1.view(d_view_ucl2,6);
    ucl_copy(view1,d_view_ucl1,6,async);
    view1.sync();
    UCL_H_Vec<numtyp> tcomp(d_view_ucl2.numel(),cop);
    ucl_copy(tcomp,d_view_ucl2,false);
    for (int i=0; i<6; i++)
      assert(tcomp[i]==static_cast<numtyp>(i+1));
    cerr << "Done.\n";
    cerr << "  Partial vector view from pointer...";
    view1.view(d_view_ucl1.begin(),6,cop);
    v7 << view1;
    assert(v7.str()=="1 2 3 4 5 6");
    cerr << "Done.\n";
    }
    
    {
    ostringstream v1,v2,v3,v4,v5,v6,v7;
    cerr << "  Offset full vector view...";
    UCL_D_Vec<numtyp> view1;
    view1.view_offset(3,d_view_ucl1);
    v1 << view1;
    assert(v1.str()=="4 5 6 7 8 9");
    cerr << "Done.\n";
    cerr << "  Offset partial vector view...";
    view1.view_offset(3,d_view_ucl1,3);
    v2 << view1;
    assert(v2.str()=="4 5 6");
    view1.view_offset(2,d_view_ucl1,1,2);
    v3 << view1;
    assert(v3.str()=="3 4");
    cerr << "Done.\n";
    cerr << "  Offset partial matrix view...";
    view1.view_offset(d_view_ucl3.row_size(),d_view_ucl3,3);
    v5 << view1;
    assert(v5.str()=="4 5 6");
    view1.view_offset(d_view_ucl3.row_size()+1,d_view_ucl3,1,2);
    v6 << view1;
    assert(v6.str()=="5 6");
    cerr << "Done.\n";
    d_view_ucl2.zero();
    cerr << "  Copy into offset partial view...";
    view1.view_offset(3,d_view_ucl2,2);
    ucl_copy(view1,d_view_ucl1,2,async);
    view1.sync();
    UCL_H_Vec<numtyp> tcomp(d_view_ucl2.numel(),cop);
    ucl_copy(tcomp,d_view_ucl2,false);
    for (int i=3; i<5; i++)
      assert(tcomp[i]==static_cast<numtyp>(i+1-3));
    
    cerr << "Done.\n";
    cerr << "  Offset partial vector view from pointer...";
    view1.view_offset(3,d_view_ucl1.begin(),3,cop);
    v7 << view1;
    assert(v7.str()=="4 5 6");
    cerr << "Done.\n";
    
    cerr << "  Host view from device object...";
    UCL_H_Vec<numtyp> tview(9,cop);
    view1.view(tview);
    cerr << "Done.\n";
    }
        
    cerr << "  Destructing...";
  }
  cerr << "Done.\n";
  
  // -------------------------------------------------------------------------
  // - DEVICE MATRIX TESTS
  // -------------------------------------------------------------------------
  cerr << "Device (2,3) matrix tests.\n";
  {
    cerr << "  Constructor allocation...";
    UCL_D_Mat<numtyp> mat(2,3,cop);
    cerr << "Done.\n";
    cerr << "  Clearing...";
    mat.clear();
    cerr << "Done.\n";
    cerr << "  Member allocation...";
    mat.alloc(2,3,cop);
    cerr << "Done.\n";
    cerr << "  Zeroing...";
    mat.zero();
    cerr << "Done.\n";
    cerr << "  Member tests...";
    assert(mat.numel()==6 && mat.rows()==2 && mat.cols()==3);
    cerr << "Done.\n";
    cerr << "  Allocation with mat instead of device...";
    UCL_D_Mat<numtyp> mat2;
    mat2.alloc(2,3,mat);
    cerr << "Done.\n";
    cerr << "  Destructing...";
  
    count_23_single.zero();
    cerr << "  Copy single precision vector into...";
    ucl_copy(mat,count_vec_single,async);
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<6; i++) assert(count_23_single[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    mat.zero();
    count_23_single.zero();
    cerr << "  Copy double precision vector into...";
    ucl_copy(mat,count_vec_double,async);
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<6; i++) assert(count_23_single[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    mat.zero();
    count_23_single.zero();
    cerr << "  Copy int vector into...";
    ucl_copy(mat,count_vec_int,async);
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<6; i++) assert(count_23_single[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    cerr << "  Copy into single precision vector...";
    count_6_single.zero();
    ucl_copy(count_6_single,mat,async);
    count_6_single.sync();
    for (int i=0; i<6; i++) assert(count_6_single[i]==static_cast<float>(i));
    cerr << "Done.\n";
    cerr << "  Copy into double precision vector...";
    count_6_double.zero();
    ucl_copy(count_6_double,mat,async);
    count_6_double.sync();
    for (int i=0; i<6; i++) assert(count_6_double[i]==static_cast<double>(i));
    cerr << "Done.\n";
    cerr << "  Copy into int precision vector...";
    count_6_int.zero();
    ucl_copy(count_6_int,mat,async);
    count_6_int.sync();
    for (int i=0; i<6; i++) assert(count_6_int[i]==static_cast<int>(i));
    cerr << "Done.\n";
    
    mat.zero();
    count_23_single.zero();
    cerr << "  Copy single precision matrix into...";
    ucl_copy(mat,count_mat_single,async);
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<6; i++) assert(count_23_single[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    mat.zero();
    count_23_single.zero();
    cerr << "  Copy double precision matrix into...";
    ucl_copy(mat,count_mat_double,async);
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<6; i++) assert(count_23_single[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    mat.zero();
    count_23_single.zero();
    cerr << "  Copy int matrix into...";
    ucl_copy(mat,count_mat_int,async);
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<6; i++) assert(count_23_single[i]==static_cast<numtyp>(i));
    cerr << "Done.\n";
    cerr << "  Copy into single precision matrix...";
    count_23_single.zero();
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<6; i++) assert(count_23_single[i]==static_cast<float>(i));
    cerr << "Done.\n";
    cerr << "  Copy into double precision matrix...";
    count_23_double.zero();
    ucl_copy(count_23_double,mat,async);
    count_23_double.sync();
    for (int i=0; i<6; i++) assert(count_23_double[i]==static_cast<double>(i));
    cerr << "Done.\n";
    cerr << "  Copy into int precision matrix...";
    count_23_int.zero();
    ucl_copy(count_23_int,mat,async);
    count_23_int.sync();
    for (int i=0; i<6; i++) assert(count_23_int[i]==static_cast<int>(i));
    cerr << "Done.\n";

    mat.zero();
    count_23_single.zero();
    cerr << "  Copy single precision vector into slice...";
    ucl_copy(mat,count_vec4_single,2,2,async);
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<2; i++) assert(count_23_single(0,i)==
                                   static_cast<numtyp>(i));
    for (int i=0; i<2; i++) assert(count_23_single(1,i)==
                                   static_cast<numtyp>(i+2));
    cerr << "Done.\n";
    mat.zero();
    count_23_single.zero();
    cerr << "  Copy double precision vector into slice...";
    ucl_copy(mat,count_vec4_double,2,2,async);
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<2; i++) assert(count_23_single(0,i)==
                                   static_cast<numtyp>(i));
    for (int i=0; i<2; i++) assert(count_23_single(1,i)==
                                   static_cast<numtyp>(i+2));
    cerr << "Done.\n";
    mat.zero();
    count_23_single.zero();
    cerr << "  Copy single precision vector slice into slice...";
    ucl_copy(mat,count_vec_single,2,2,async);
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<2; i++) assert(count_23_single(0,i)==
                                   static_cast<numtyp>(i));
    for (int i=0; i<2; i++) assert(count_23_single(1,i)==
                                   static_cast<numtyp>(i+2));
    cerr << "Done.\n";
    mat.zero();
    count_23_single.zero();
    cerr << "  Copy double precision vector slice into slice...";
    ucl_copy(mat,count_vec_double,2,2,async);
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<2; i++) assert(count_23_single(0,i)==
                                   static_cast<numtyp>(i));
    for (int i=0; i<2; i++) assert(count_23_single(1,i)==
                                   static_cast<numtyp>(i+2));
    cerr << "Done.\n";

    ucl_copy(mat,count_mat_single,async);
    
    count_4_single.zero();
    cerr << "  Copy slice into single precision vector...";
    ucl_copy(count_4_single,mat,2,2,async);
    count_4_single.sync();
    for (int i=0; i<2; i++) assert(count_4_single[i]==static_cast<float>(i));
    for (int i=2; i<4; i++) assert(count_4_single[i]==static_cast<float>(i+1));
    cerr << "Done.\n";
    count_4_double.zero();
    cerr << "  Copy double precision vector into...";
    ucl_copy(count_4_double,mat,2,2,async);
    count_4_double.sync();
    for (int i=0; i<2; i++) assert(count_4_double[i]==static_cast<double>(i));
    for (int i=2; i<4; i++) assert(count_4_double[i]==static_cast<double>(i+1));
    cerr << "Done.\n";

    mat.zero();
    count_23_single.zero();
    cerr << "  Copy single precision slice into slice...";
    ucl_copy(mat,count_mat_single,2,2,async);
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<2; i++) assert(count_23_single(0,i)==
                                   static_cast<numtyp>(i));
    for (int i=0; i<2; i++) assert(count_23_single(1,i)==
                                   static_cast<numtyp>(i+3));
    cerr << "Done.\n";
    mat.zero();
    count_23_single.zero();
    cerr << "  Copy double precision slice into...";
    ucl_copy(mat,count_mat_double,2,2,async);
    ucl_copy(count_23_single,mat,async);
    count_23_single.sync();
    for (int i=0; i<2; i++) assert(count_23_single(0,i)==
                                   static_cast<numtyp>(i));
    for (int i=0; i<2; i++) assert(count_23_single(1,i)==
                                   static_cast<numtyp>(i+3));
    cerr << "Done.\n";

    ucl_copy(mat,count_mat_single,async);
    
    count_23_single.zero();
    cerr << "  Copy into another device matrix...";
    ucl_copy(mat2,mat,async);
    ucl_copy(count_23_single,mat2,async);
    count_23_single.sync();
    for (int i=0; i<6; i++) assert(count_23_single[i]==static_cast<float>(i));
    cerr << "Done.\n";
    
    count_23_single.zero();
    mat2.zero();
    cerr << "  Copy into another device matrix slice...";
    ucl_copy(mat2,mat,2,2,async);
    ucl_copy(count_23_single,mat2,async);
    count_23_single.sync();
    for (int i=0; i<2; i++) assert(count_23_single(0,i)==static_cast<float>(i));
    for (int i=0; i<2; i++) assert(count_23_single(1,i)==static_cast<float>(i+3));
    cerr << "Done.\n";
   
    UCL_D_Vec<numtyp> mat3;
    mat3.alloc(6,mat);
    count_6_single.zero();
    cerr << "  Copy into another device vector...";
    ucl_copy(mat3,mat,async);
    ucl_copy(count_6_single,mat3,async);
    count_6_single.sync();
    for (int i=0; i<6; i++) assert(count_6_single[i]==static_cast<float>(i));
    cerr << "Done.\n";
    
    mat.zero();
    count_6_single.zero();
    cerr << "  Copy from another device vector...";
    ucl_copy(mat,mat3,async);
    ucl_copy(count_6_single,mat,async);
    count_6_single.sync();
    for (int i=0; i<6; i++) assert(count_6_single[i]==static_cast<float>(i));
    cerr << "Done.\n";
    
    //cerr << "Copy into 
  
    cerr << "  Print test...";
    ostringstream out1,out2,out3,out4,out5;
    ucl_print(mat,out1);
    assert(out1.str()=="0 1 2\n3 4 5");
    cerr << "Done.\n";
    cerr << "  Print slice test...";
    ucl_print(mat,2,2,out2);
    assert(out2.str()=="0 1\n3 4");
    cerr << "Done.\n";
    cerr << "  Print vector slice test...";
    ucl_print(mat,3,out3);
    assert(out3.str()=="0 1 2");
    cerr << "Done.\n";
    
    cerr << "  Print with device test...";
    ucl_print(mat,out4,cop);
    assert(out1.str()=="0 1 2\n3 4 5");
    cerr << "Done.\n";
    cerr << "  Operator << test...";
    out5 << mat;
    assert(out1.str()=="0 1 2\n3 4 5");
    cerr << "Done.\n";
    
    // -------------------------------------------------------------------------
    // - ALLOCATE SOME MATRICES FOR VIEW TESTS
    // -------------------------------------------------------------------------
    UCL_D_Vec<numtyp> d_view_ucl1(9,cop), d_view_ucl2(9,cop);
    ucl_copy(d_view_ucl1,h_view_ucl1,false);
    UCL_D_Mat<numtyp> d_view_ucl3(3,3,cop), d_view_ucl4(3,3,cop);
    ucl_copy(d_view_ucl3,h_view_ucl3,false);

    {
    ostringstream v1,v2,v3,v4,v5,v6,v7;
    cerr << "  Full vector view...";
    UCL_D_Mat<numtyp> view1;
    view1.view(d_view_ucl1);
    v1 << view1;
    assert(v1.str()=="1 2 3 4 5 6 7 8 9");
    cerr << "Done.\n";
    cerr << "  Partial vector view...";
    view1.view(d_view_ucl1,6);
    v2 << view1;
    assert(v2.str()=="1 2 3 4 5 6");
    view1.view(d_view_ucl1,1,4);
    v3 << view1;
    assert(v3.str()=="1 2 3 4");
    cerr << "Done.\n";
    cerr << "  Full matrix view...";
    view1.view(d_view_ucl3);
    v4 << view1;
    assert(v4.str()=="1 2 3\n4 5 6\n7 8 9");
    cerr << "Done.\n";
    cerr << "  Partial matrix view...";
    view1.view(d_view_ucl3,2,3);
    v5 << view1;
    assert(v5.str()=="1 2 3\n4 5 6");
    view1.view(d_view_ucl3,3,1);
    v6 << view1;
    assert(v6.str()=="1\n4\n7");
    cerr << "Done.\n";
    d_view_ucl2.zero();
    cerr << "  Copy into partial view...";
    view1.view(d_view_ucl2,6);
    ucl_copy(view1,d_view_ucl1,6,async);
    view1.sync();
    UCL_H_Vec<numtyp> tcomp(d_view_ucl2.numel(),cop);
    ucl_copy(tcomp,d_view_ucl2,false);
    for (int i=0; i<6; i++)
      assert(tcomp[i]==static_cast<numtyp>(i+1));
    cerr << "Done.\n";
    cerr << "  Partial vector view from pointer...";
    view1.view(d_view_ucl1.begin(),6,cop);
    v7 << view1;
    assert(v7.str()=="1 2 3 4 5 6");
    cerr << "Done.\n";
    }
    
    {
    ostringstream v1,v2,v3,v4,v5,v6,v7;
    cerr << "  Offset full vector view...";
    UCL_D_Mat<numtyp> view1;
    view1.view_offset(3,d_view_ucl1);
    v1 << view1;
    assert(v1.str()=="4 5 6 7 8 9");
    cerr << "Done.\n";
    cerr << "  Offset partial vector view...";
    view1.view_offset(3,d_view_ucl1,3);
    v2 << view1;
    assert(v2.str()=="4 5 6");
    view1.view_offset(2,d_view_ucl1,1,2);
    v3 << view1;
    assert(v3.str()=="3 4");
    cerr << "Done.\n";
    cerr << "  Offset full matrix view...";
    view1.view_offset(d_view_ucl3.row_size(),d_view_ucl3);
    v4 << view1;
    assert(v4.str()=="4 5 6\n7 8 9");
    cerr << "Done.\n";
    cerr << "  Offset partial matrix view...";
    view1.view_offset(1,d_view_ucl3,2,2);
    v5 << view1;
    assert(v5.str()=="2 3\n5 6");
    view1.view_offset(d_view_ucl3.row_size()+1,d_view_ucl3,2,1,
                      d_view_ucl3.row_size()+1);
    v6 << view1;
    assert(v6.str()=="5\n9");
    cerr << "Done.\n";
    d_view_ucl2.zero();
    cerr << "  Copy into offset partial view...";
    view1.view_offset(3,d_view_ucl2,2);
    ucl_copy(view1,d_view_ucl1,2,async);
    view1.sync();
    UCL_H_Vec<numtyp> tcomp(d_view_ucl2.numel(),cop);
    ucl_copy(tcomp,d_view_ucl2,false);
    for (int i=3; i<5; i++)
      assert(tcomp[i]==static_cast<numtyp>(i+1-3));
    
    cerr << "Done.\n";
    cerr << "  Offset partial vector view from pointer...";
    view1.view_offset(3,d_view_ucl1.begin(),3,cop);
    v7 << view1;
    assert(v7.str()=="4 5 6");
    cerr << "Done.\n";
    
    cerr << "  Host view from device object...";
    UCL_H_Mat<numtyp> tview(3,3,cop);
    view1.view(tview);
    cerr << "Done.\n";
    }
        
    cerr << "  Destructing..."; 
  }
  cerr << "Done.\n";
  cerr << "Stopping device timer...";
  timer.stop();
  cerr << timer.seconds() << " seconds.\n";
