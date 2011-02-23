#!/bin/tcsh
foreach arg ( $argv )
  echo "-----------------------------------------------"
  echo "             $arg (sm_13)"
  echo "-----------------------------------------------"
  nvcc -DUNIX -DNV_KERNEL -O3 -Xptxas -v -arch=sm_13 $arg |& awk '$6=="function"{printf("%s",$7)}$4=="Used"{print $0}' | sed 's/ptxas info   //g' | sed 's/Used //g'
  echo "-----------------------------------------------"
  echo "             $arg (sm_20)"
  echo "-----------------------------------------------"
  nvcc -DUNIX -DNV_KERNEL -O3 -Xptxas -v -arch=sm_20 $arg |& awk '$6=="function"{printf("%s",$7)}$4=="Used"{print $0}' | sed 's/ptxas info   //g' | sed 's/Used //g'
end
  
