#include <cstdlib>
#include <iostream>

#include "geryon.h"

int main(int argc, char* argv[])
{
   // Create device
   UCL_Device device(0);
   // Create timer
   UCL_Timer timer(device);
   
   // Start timer
   timer.start();
   // Do something ...
   sleep(1);
   // Stop timer
   timer.stop();
   // Print time for first stamp
   std::cout << "Time 1: " << timer.time() << std::endl;
   // Add first time to total timer
   timer.add_to_total();
   
   // Start timer
   timer.start();
   // Do something more ...
   sleep(2);
   // Stop timer
   timer.stop();
   // Print time for second stamp
   std::cout << "Time 2: " << timer.time() << std::endl;
   // Add second time to total timer
   timer.add_to_total();
   
   // Print total time
   std::cout << "Total: " << timer.total_time() << std::endl;

   return EXIT_SUCCESS;
}
