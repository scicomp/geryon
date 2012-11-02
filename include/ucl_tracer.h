#ifndef UCL_TRACER_H
#define UCL_TRACER_H

#include <map>
#include <iomanip>
#include <iostream>
#include <string>

#if !defined(UCL_OPENCL) && !defined(UCL_CUDADR) && !defined(UCL_CUDART)
#error "Must define a target platform"
#endif

// OpenCL headers
#ifdef UCL_OPENCL
#include "ocl_device.h"
#include "ocl_timer.h"
using namespace ucl_opencl;
#endif

// CUDA driver headers
#ifdef UCL_CUDADR
#include "nvd_device.h"
#include "nvd_timer.h"
using namespace ucl_cudadr;
#endif

// CUDA runtime headers
#ifdef UCL_CUDART
#include "nvc_device.h"
#include "nvc_timer.h"
using namespace ucl_cudart;
#endif

/**
 * \brief Implements the tracing class.
 */
class UCL_Tracer {
public:
  /**
   * \brief Ordinary constructor.
   * \param dev UCL_Device refernce.
   */
  UCL_Tracer(UCL_Device &dev) { _dev = &dev; }
  
  /**
   * \brief Starts the trace for a named timer.
   * 
   * If the timer does not exists, it will be created. Timer names have 
   * to be unique for each tracing object. They are stored in a map.
   *
   * \param name The name of the timer.
   */
  void start(std::string name)
  {
    getTimer(name)->start();
  }
  
  /**
   * \brief Stops a named timer and adds the time to the total time.
   * \param name The name of the timer.
   */
  void stop(std::string name)
  {
    getTimer(name)->stop();
    getTimer(name)->add_to_total();
  }
  
  /**
   * \brief Prints all the trace information on the stream.
   * \param stream The stream to print on (e.g. std::cout)
   */
  void print_all(std::ostream& stream)
  {
      // Maximum lendth of the timer name
      unsigned int maxlen = 50;
      
      // Calculate the total time
      double totalTime = 0;
      std::map<std::string, UCL_Timer*>::iterator it;
      for(it = _timers.begin(); it != _timers.end(); ++it) 
      {
         totalTime += it->second->total_time();
      }
      stream << "-------------------------------------------------------------" 
             << "---------------" << std::endl;
      
      // Iterate over all timers and print the total time
      for(it = _timers.begin(); it != _timers.end(); ++it) 
      {
         unsigned int respace = maxlen;
         std::string output = it->first;
         if(it->first.length() > maxlen)
         {
            // Resize the string
            output.resize(maxlen);
         } 
         else
         {
            respace = it->first.length()-1;
         }
         
         stream << output << std::setw(maxlen-respace) << " "
                << std::scientific << it->second->total_time() << " "
                << std::resetiosflags(::std::ios::scientific)
                << "(" << (it->second->total_time()/totalTime*100) << "%)"
                << std::endl;
      }
      
      // Print out the total time
      stream << "-------------------------------------------------------------" 
             << "---------------" << std::endl
             << std::scientific
             << "Total" << std::setw(maxlen-4) << " " << totalTime
             << std::resetiosflags(::std::ios::scientific) << std::endl;  
  }
  
  /**
   * \brief Construcor that deletes all timers.
   */
  ~UCL_Tracer() 
  {
    std::map<std::string, UCL_Timer*>::iterator it;
    for(it = _timers.begin(); it != _timers.end(); ++it)
    {
      delete it->second;
    }
    _timers.erase(_timers.begin(), _timers.end());
  }
private:
  /**
   * \brief looks for a named timer in the timer map. 
   * 
   * The timer will be created if it does not exist. Timers are unique.
   * \param name The name of the timer.
   */
  UCL_Timer* getTimer(std::string name)
  {
      // Look if the timer exists   
      std::map<std::string, UCL_Timer*>::iterator it = _timers.find(name);
      if (it == _timers.end())
      {
        // Create timer if there is no timer for the name
        std::pair<std::map<std::string, UCL_Timer*>::iterator,bool> timer;
        timer = _timers.insert(
            std::pair<std::string, UCL_Timer*>(name, new UCL_Timer(*_dev)));
         return timer.first->second;
      }
      else
      {
         return it->second;
      }     
  }
  
  /**< \brief The timer map. */
  std::map<std::string, UCL_Timer*> _timers;
  
  /**< \brief The UCL_Device on which the timings are taken. */
  UCL_Device* _dev;
};

#endif 

