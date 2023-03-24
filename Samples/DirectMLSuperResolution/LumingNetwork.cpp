#include "pch.h"
#include "LumingNetwork.h"
#include "LumingNetworkWeights.h"

DmlLumingNetwork::DmlLumingNetwork(IDMLDevice* device, DML_TENSOR_DATA_TYPE dataType, uint32_t lrHeight, uint32_t lrWidth)
  : DmlNetworkBuilder(L"FNS Candy", device)
{
    auto lr = Input(dml::TensorDesc(dataType, { 1, 3, lrHeight, lrWidth }));
    m_modelInputBufferSize = lr.GetOutputDesc().totalTensorSizeInBytes;

    PushNameScope("Pre-process upsample");
    auto hr = dml::Resample(lr, dml::TensorDimensions{ 1, 3, lrHeight * 2, lrWidth * 2 }, DML_INTERPOLATION_MODE_LINEAR);
    PopNameScope();

    constexpr uint32_t gc = 32;

    auto x1 = Conv2d("conv1", hr, 3, gc, 3, 2, 1);
    auto out = ResidualDenseBlock_4C("rdb1", x1, gc);
    out = ResidualDenseBlock_4C("rdb2", out, gc);
    out = out * 0.2f + x1; // TODO: should be fused into a single op
    out = Conv2d("conv_final", out, gc, 12, 3, 1, 1);
    out = dml::DepthToSpace(out, 2);

    m_outputs = { out };

    m_modelOutputBufferSize = out.GetOutputDesc().totalTensorSizeInBytes;
}

#define ConvertToVectorAndAssign(weights, name, weight_array) \
{ \
    int size = sizeof(weight_array)/sizeof(float); \
    WeightsType myvector; \
    myvector.assign(reinterpret_cast<float*>(weight_array), reinterpret_cast<float*>(weight_array) + size); \
    weights[name] = myvector; \
} \

void DmlLumingNetwork::PopulateWeightMap(WeightMapType& weights)
{
    ConvertToVectorAndAssign(weights, "conv1_weights", conv1_weights);
}

dml::Expression DmlLumingNetwork::Conv2d(
    std::string&& name,
    dml::Expression input,
    uint32_t inChannels,
    uint32_t outChannels,
    uint32_t kernelSize,
    uint32_t stride,
    uint32_t padding,
    std::optional<DML_OPERATOR_TYPE> activationType)
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

dml::Expression DmlLumingNetwork::ResidualDenseBlock_4C(std::string&& name, dml::Expression x, uint32_t gc)
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
