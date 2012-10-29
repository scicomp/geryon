#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include "geryon.h"

int main(int argc, char* argv[])
{
   std::string flags("");

   // Check for arguments
   if(argc < 2)
   {
      std::cerr << "Please input the kernel name!" << std::endl;
      exit(EXIT_FAILURE);
   }
   else if(argc == 2)
   {
      std::cout << "Compiling without flags" << std::endl;
   }
   else if(argc == 3)
   {
      // Compile options
      std::cout << "Flags: " << argv[2] << std::endl;
      flags = argv[2];
   }

   // Initialize device
   UCL_Device device;
   std::cout << "Using platform: " << device.platform_name() << std::endl 
      << "Found " << device.num_devices() << " device(s)." << std::endl 
      << "Setting to device 0..." << std::endl;
   device.set(0);

   // Create compilation timer
   UCL_Timer timer;
   timer.init(device);
   timer.start();
   
   // Create program
   UCL_Program program(device);
   std::string log;
   int result = program.load(argv[1], flags.c_str(), &log);
 
   // Check Compilation
   if(result != UCL_SUCCESS)
   {
      std::cerr << "Error was: " << ucl_check(result) << std::endl;
      std::cerr << log << std::endl;
      exit(EXIT_FAILURE);      
   }
   else
   {
      std::cout << "Compilation was successful" << std::endl;
      result = EXIT_SUCCESS;
   }
   
   // Stop compilation timer
   timer.stop();
   std::cout << timer.seconds() << " seconds." << std::endl;

   return result;
}
