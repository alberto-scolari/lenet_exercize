#pragma once

/**
 * Represents a max-pooling layer.
 */
struct MaxPoolLayer {
  int size, stride;
  inline MaxPoolLayer(int size_, int stride_) : size(size_), stride(stride_) {}
};
