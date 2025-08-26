#include <sstream>
#include <cstdio>

#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(int inputs_, int outputs_) : outputs(outputs_), inputs(inputs_),
                                                    pneurons(inputs_ * outputs_), pbias(outputs_) {}

bool FullyConnectedLayer::FromFile(const char *fileprefix)
{
    std::stringstream ssf, ssbf;
    ssf << fileprefix << ".bin";
    ssbf << fileprefix << ".bias.bin";

    // Read weights file
    FILE *fp = fopen(ssf.str().c_str(), "rb");
    if (!fp)
    {
        printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
        return false;
    }
    fread(&pneurons[0], sizeof(float), inputs * outputs, fp);
    fclose(fp);

    // Read bias file
    fp = fopen(ssbf.str().c_str(), "rb");
    if (!fp)
    {
        printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
        return false;
    }
    fread(&pbias[0], sizeof(float), outputs, fp);
    fclose(fp);
    return true;
}

void FullyConnectedLayer::ToFile(const char *fileprefix)
{
    std::stringstream ssf, ssbf;
    ssf << fileprefix << ".bin";
    ssbf << fileprefix << ".bias.bin";

    // Write weights file
    FILE *fp = fopen(ssf.str().c_str(), "wb");
    if (!fp)
    {
        printf("ERROR: Cannot open file %s\n", ssf.str().c_str());
        exit(2);
    }
    fwrite(&pneurons[0], sizeof(float), inputs * outputs, fp);
    fclose(fp);

    // Write bias file
    fp = fopen(ssbf.str().c_str(), "wb");
    if (!fp)
    {
        printf("ERROR: Cannot open file %s\n", ssbf.str().c_str());
        exit(2);
    }
    fwrite(&pbias[0], sizeof(float), outputs, fp);
    fclose(fp);
}
