#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>

namespace common{
class Timer
{
  public:
    Timer()
    {
        tic();
    }

    void tic()
    {
        start_ = std::chrono::system_clock::now();
    }

    double end()
    {
        end_ = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_ - start_;
        return elapsed_seconds.count() * 1000;
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> start_, end_;
};

}
