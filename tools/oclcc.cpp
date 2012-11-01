#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <vector>
#include "geryon.h"

std::string filename("");
std::string flags("");
int target = -1;
bool debug = false;

UCL_Device device;

void read_cmd(std::vector<std::string> args)
{
   std::vector<std::string>::iterator cmd;

   // Check for debug flag
   cmd = std::find(args.begin(), args.end(), "-d");
   if(cmd != args.end()) debug = true;
         
   // Check if there are targets on this machine
   if(device.num_devices() == 0)
   {
      std::cerr << "No target devices found." << std::endl;
      exit(EXIT_FAILURE);
   }
   else
   {
      // Targets found on the machine
      std::cout << "Found " << device.num_devices() << " device|s" << std::endl;
      
      // Check for user specified target
      cmd = std::find(args.begin(), args.end(), "-t");
      if(cmd != args.end())
      {
         if(debug) std::cout << "Found command -t" << std::endl;
         cmd++;
         if(cmd != args.end())
         {
            if(debug) std::cout << "Specified target is: " << *cmd << std::endl;
            std::stringstream id(*cmd);
            id >> target;
            
            if(id == NULL)
            {
               if(debug) std::cout << "Could not parse target" << std::endl;
               target = -1;
            }
            else
            {
               if(target >= device.num_devices()) 
               { 
                  if(debug) std::cout << "Target does not exist" << std::endl;
                  target = -1;
               }
               else
               {
                  if(debug) std::cout << "Specified target found" << std::endl;
               }
            }
         }
      }
      
      // Set default target if no target specified or syntax error
      if(target == -1)
      {
         if(debug) std::cout << "No target specified set to 0" << std::endl;
         target = 0; 
      }
   }
   
   // Check for kernel file name argument
   cmd = std::find(args.begin(), args.end(), "-k");
   if(cmd != args.end())
   {
      if(debug) std::cout << "Found command -k" << std::endl;
      cmd++;
      if(cmd != args.end())
      {
         filename = *cmd;
         if(debug) std::cout << "Using kernel file: " << filename << std::endl;
      }
   }
   
   if(filename == "")
   {
      std::cerr << "Please specify kernel file with -k" << std::endl;
      exit(EXIT_FAILURE);
   }

   // Check for compiler flags
   cmd = std::find(args.begin(), args.end(), "-f");
   if(cmd != args.end())
   {
      if(debug) std::cout << "Found command -f" << std::endl;
      cmd++;
      if(cmd != args.end())
      {
         flags = *cmd;
         if(debug) std::cout << "Compiler flags are: " << flags << std::endl;
      }
      else
      {
         if(debug) std::cout << "No compiler flags specified" << std::endl;
      }
   }
   else
   {
      if(debug) std::cout << "No compiler flags specified" << std::endl;
   }
}

void print_help()
{
   std::cout << "Usage: oclcc [option] -k [filename] -f [flags]" << std::endl;
   std::cout << "Option:" << std::endl;
   std::cout << " -t <number>    Target device id" << std::endl;
   std::cout << " -d             Turn on debug" << std::endl;
   std::cout << "Mandatory:" << std::endl;
   std::cout << " -k <filename>  The name of the kernel file" << std::endl;
   std::cout << " -f <flags>     String of compiler flags" << std::endl;
}

int main(int argc, char* argv[])
{
   if(argc == 1) 
   {
      print_help();
      exit(EXIT_SUCCESS);
   }
   
   // Store all parameters in a vector
   std::vector<std::string> args;
   for(int i = 0; i < argc; i++)
   {
      args.push_back(argv[i]);
   }
   read_cmd(args);

   // Initialize device   
   std::cout << "Using platform: " << device.platform_name() << std::endl; 
   std::cout << "Setting to device ..." << target << std::endl;
   device.set(target);

   // Create compilation timer
   UCL_Timer timer;
   timer.init(device);
   timer.start();
   
   // Create program
   UCL_Program program(device);
   std::string log;
   int result = program.load(filename.c_str(), flags.c_str(), &log);
 
   // Check Compilation
   if(result != UCL_SUCCESS)
   {
      std::cerr << "Error: " << ucl_check(result) << " " << log << std::endl;
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
