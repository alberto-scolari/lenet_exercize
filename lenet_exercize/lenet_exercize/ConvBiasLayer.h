#pragma once

#include <vector>

/**
 * Represents a convolutional layer with bias.
 */
struct ConvBiasLayer {
  int in_channels, out_channels, kernel_size;
  int in_width, in_height, out_width, out_height;

  std::vector<float> pconv, pbias;

  ConvBiasLayer(int in_channels_, int out_channels_, int kernel_size_,
                int in_w_, int in_h_);

  bool FromFile(const char *fileprefix);

  void ToFile(const char *fileprefix);
};
