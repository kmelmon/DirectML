#pragma once

#include "Common.h"
#include "DmlNetworkBuilder.h"
#include "WeightMapType.h"

// FNS Candy
class DmlLumingNetwork : public DmlNetworkBuilder
{
public:
    DmlLumingNetwork(IDMLDevice* device, DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT32, DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE, uint32_t lrHeight = 720, uint32_t lrWidth = 1280);

    void PopulateWeightMap(WeightMapType& weights);

    dml::Expression Conv2d(
        std::string&& name,
        dml::Expression input,
        uint32_t inChannels,
        uint32_t outChannels,
        uint32_t kernelSize = 3,
        uint32_t stride = 1,
        uint32_t padding = 1,
        DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE,
        std::optional<DML_OPERATOR_TYPE> activationType = std::nullopt);

    dml::Expression ResidualDenseBlock_4C(std::string&& name, dml::Expression x, uint32_t gc, DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE);

    uint64_t m_modelInputBufferSize = 0;
    uint64_t m_modelOutputBufferSize = 0;
};
