#include <iostream>
#include "ARDepth.h"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "Usage: ./AR_DEPTH /path/to/sample_data/frames "
                 "/path/to/sample_data/reconsrtruction"
              << std::endl;
    return 0;
  }

  std::vector<std::string> args;
  args.assign(argv + 1, argv + argc);
  std::string input_frames = args[0];
  std::string input_colmap = args[1];
  bool resize = true;
  bool visualize = true;

  ARDepth ardepth(input_frames, input_colmap, resize, visualize);
  try {
    ardepth.run();
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}
