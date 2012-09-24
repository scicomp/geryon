/***************************************************************************
                              ucl_test_vecadd.h
                             -------------------
                               W. Michael Brown

  Test code for UCL (vector add).

 __________________________________________________________________________
    This file is part of the Geryon Unified Coprocessor Library (UCL)
 __________________________________________________________________________

    begin                : Thu Feb 11 2010
    copyright            : (C) 2010 by W. Michael Brown
    email                : brownw@ornl.gov
 ***************************************************************************/

/* -----------------------------------------------------------------------
   Copyright (2010) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the Simplified BSD License.
   ----------------------------------------------------------------------- */

  cerr << "Kernel (vector add) tests.\n";
//  {
//    cerr << "  Loading/compiling kernel from string...";
//    string flags=string("-D Ordinal=int -D Scalar=")+
//                 ucl_template_name<numtyp>();
//    UCL_Program p_vec_add(cop);
//    p_vec_add.load_string(kernel_string,flags.c_str());
//    UCL_Kernel k_vec_add(p_vec_add,"vec_add");
//    cerr << "Done.\n";
//  }
  
  cerr << "  Loading/compiling kernel from file...";
  string flags=string("-D Ordinal=int -D Scalar=")+ucl_template_name<numtyp>();
  UCL_Program p_vec_add(cop);
  p_vec_add.load(kernel_name.c_str(),flags.c_str());
  UCL_Kernel k_vec_add(p_vec_add,"vec_add");
  cerr << "Done.\n";

  UCL_H_Vec<numtyp> a(6,cop,UCL_WRITE_OPTIMIZED), b(6,cop,UCL_WRITE_OPTIMIZED); 
  UCL_D_Vec<numtyp> cop_a(6,cop,UCL_READ_ONLY), cop_b(6,cop,UCL_READ_ONLY);
  UCL_D_Vec<numtyp> answer(6,cop,UCL_WRITE_ONLY);
  UCL_Timer timer_com(cop), timer_kernel(cop);

  for (int i=0; i<6; i++) { a[i]=i; b[i]=2.0*i; }

  timer_com.start();
  ucl_copy(cop_a,a,true);
  ucl_copy(cop_b,b,true);
  timer_com.stop();

  timer_kernel.start();
  cerr << "  Setting kernel arguments...";
  k_vec_add.set_size(2,3);
  k_vec_add.add_args(&cop_a.begin(),&cop_b.begin(),&answer.begin());
  cerr << "Done.\n";
  cerr << "  Running kernel...";
  k_vec_add.run();
  cerr << "Done.\n";
  timer_kernel.stop();
  
  cerr << "  Checking answer...";
  ostringstream out1;
  out1 << answer;
  assert(out1.str()=="0 3 6 9 12 15");
  cerr << "Done.\n";
  
  timer_kernel.start();
  answer.zero();
  cerr << "  Running kernel with run...";
  k_vec_add.clear_args();
  k_vec_add.set_size(2,3);
  k_vec_add.add_args(&cop_a.begin(),&cop_b.begin(),&answer.begin());
  k_vec_add.run(&cop_a.begin(),&cop_b.begin(),&answer.begin());
  cerr << "Done.\n";
  timer_kernel.stop();
  
  cerr << "  Checking answer...";
  ostringstream out2;
  out2 << answer;
  assert(out2.str()=="0 3 6 9 12 15");
  cerr << "Done.\n";
    
  k_vec_add.set_size(2,3,cop.cq());
  timer_kernel.start();
  answer.zero();
  cerr << "  Running kernel with run_cq...";
  k_vec_add.run(&cop_a.begin(),&cop_b.begin(),&answer.begin());
  cerr << "Done.\n";
  timer_kernel.stop();
  
  cerr << "  Checking answer...";
  ostringstream out3;
  out3 << answer;
  assert(out3.str()=="0 3 6 9 12 15");
  cerr << "Done.\n";
    
  k_vec_add.set_size(2,3,cop.cq());
  timer_kernel.start();
  answer.zero();
  cerr << "  Running kernel with implicit begin...";
  k_vec_add.run(&cop_a,&cop_b,&answer);
  cerr << "Done.\n";
  timer_kernel.stop();
  
  cerr << "  Checking answer...";
  ostringstream out4;
  out4 << answer;
  assert(out4.str()=="0 3 6 9 12 15");
  cerr << "Done.\n";
    
  double timer_com_s=timer_com.seconds();
  double timer_kernel_s=timer_kernel.seconds(); 
  assert(timer_com_s>-1.0 && timer_kernel_s>-1.0);
