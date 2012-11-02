#include <cstdlib>
#include "geryon.h"

int main(int argc, char* argv[])
{
   // Create device
   UCL_Device device(0);
   // Create Tracer
   UCL_Tracer tracer(device);
   
   // Start trace
   tracer.start("Function 1");
   // Do something ...
   sleep(1);
   // Stop trace
   tracer.stop("Function 1");
   
   // Start trace
   tracer.start("Function 2");
   // Do something more ...
   sleep(2);
   // Stop trace
   tracer.stop("Function 2");
   
   // Print total trace
   tracer.print_all(std::cout);

   return EXIT_SUCCESS;
}
