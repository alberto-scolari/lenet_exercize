#pragma once

#include <vector>

/**
 * Represents a fully-connected neural network layer with bias.
 */
struct FullyConnectedLayer
{
    int inputs, outputs;
    std::vector<float> pneurons, pbias;

    FullyConnectedLayer(int inputs_, int outputs_);

    bool FromFile(const char *fileprefix);

    void ToFile(const char *fileprefix);
};
