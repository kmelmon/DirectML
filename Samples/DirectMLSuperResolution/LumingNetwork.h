#pragma once

#include "Common.h"
#include "DmlNetworkBuilder.h"

// FNS Candy
class DmlLumingNetwork : public DmlNetworkBuilder
{
public:
    DmlLumingNetwork(IDMLDevice* device, DML_TENSOR_DATA_TYPE dataType = DML_TENSOR_DATA_TYPE_FLOAT32, uint32_t lrHeight = 720, uint32_t lrWidth = 1280) : DmlNetworkBuilder(L"FNS Candy", device)
    {
        auto lr = Input(dml::TensorDesc(dataType, { 1, 3, lrHeight, lrWidth }));

        PushNameScope("Pre-process upsample");
        auto hr = dml::Resample(lr, dml::TensorDimensions{1, 3, lrHeight * 2, lrWidth * 2}, DML_INTERPOLATION_MODE_LINEAR);
        PopNameScope();

        constexpr uint32_t gc = 32;

        auto x1 = Conv2d("conv1", hr, 3, gc, 3, 2, 1);
        auto out = ResidualDenseBlock_4C("rdb1", x1, gc);
        out = ResidualDenseBlock_4C("rdb2", out, gc);
        out = out * 0.2f + x1; // TODO: should be fused into a single op
        out = Conv2d("conv_final", out, gc, 12, 3, 1, 1);
        out = dml::DepthToSpace(out, 2);

        m_outputs = { out };
    }

    dml::Expression Conv2d(
        std::string&& name, 
        dml::Expression input, 
        uint32_t inChannels, 
        uint32_t outChannels, 
        uint32_t kernelSize = 3, 
        uint32_t stride = 1,
        uint32_t padding = 1,
        std::optional<DML_OPERATOR_TYPE> activationType = std::nullopt)
    {
        auto nameScope = m_graph.CreateNameScope(dml::StringView(name));

        auto filter = ConstantInput(dml::TensorDesc(
            input.GetOutputDesc().dataType, 
            { outChannels, inChannels, kernelSize, kernelSize })
        );

        auto bias = ConstantInput(dml::TensorDesc(
            input.GetOutputDesc().dataType, 
            { 1, outChannels, 1, 1 })
        );

        std::array<uint32_t, 2> paddings = { padding, padding };
        std::array<uint32_t, 2> strides = { stride, stride };
        auto convBuilder = dml::ConvolutionBuilder(input, filter, bias).StartPadding(paddings).EndPadding(paddings).Strides(strides);

        if (activationType.has_value())
        {
            switch (*activationType)
            {
                case DML_OPERATOR_ACTIVATION_LEAKY_RELU:
                    convBuilder = convBuilder.FusedActivation(dml::FusedActivation::LeakyRelu(0.2f));
                    break;
                default: throw std::invalid_argument("todo");
            }
        }

        return convBuilder.Build();
    }

    dml::Expression ResidualDenseBlock_4C(std::string&& name, dml::Expression x, uint32_t gc)
    {
        //auto nameScope = m_graph.CreateNameScope(name);



        const uint32_t nf = gc;



        auto x1 = Conv2d("conv1", x, nf + 0 * gc, gc, 3, 1, 1, DML_OPERATOR_ACTIVATION_LEAKY_RELU);

        std::array<dml::Expression, 2> l1 = { x, x1 };
        auto x2 = Conv2d("conv2", dml::Join(l1, 1), nf + 1 * gc, gc, 3, 1, 1, DML_OPERATOR_ACTIVATION_LEAKY_RELU);



        std::array<dml::Expression, 3> l2 = { x, x1, x2 };
        auto x3 = Conv2d("conv3", dml::Join(l2, 1), nf + 2 * gc, gc, 3, 1, 1, DML_OPERATOR_ACTIVATION_LEAKY_RELU);

        std::array<dml::Expression, 4> l3 = { x, x1, x2, x3 };
        auto x4 = Conv2d("conv4", dml::Join(l3, 1), nf + 3 * gc, nf, 3, 1, 1);



        return x4 * 0.2f + x; // TODO: should be fused into a single op
    }
};
