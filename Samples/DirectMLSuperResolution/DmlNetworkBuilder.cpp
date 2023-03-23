#include "pch.h"
#include "DmlNetworkBuilder.h"

DmlNetwork DmlNetworkBuilder::Compile(DML_EXECUTION_FLAGS flags)
{
    auto compiledGraph = m_graph.Compile(flags, m_outputs);

    if (!m_name.empty())
    {
        if (FAILED(compiledGraph->SetName(m_name.data())))
        {
            throw std::runtime_error("Error setting graph name");
        }
    }

    std::vector<DmlNetwork::TensorDesc> outputDescs;
    for (auto& output : m_outputs)
    {
        DmlNetwork::TensorDesc tensorDesc = {};
        tensorDesc.sizeInBytes = output.GetOutputDesc().totalTensorSizeInBytes;
        tensorDesc.flags = DML_TENSOR_FLAG_NONE;
        outputDescs.emplace_back(tensorDesc);
    }

    return DmlNetwork(
        compiledGraph.Get(),
        std::move(m_inputs),
        std::move(outputDescs)
    );
}

dml::NameScope DmlNetworkBuilder::NameScope(std::string&& name)
{
    return m_graph.CreateNameScope(name);
}

void DmlNetworkBuilder::PushNameScope(std::string&& name)
{
    m_graph.Impl()->PushName(name);
}

void DmlNetworkBuilder::PopNameScope()
{
    m_graph.Impl()->PopName();
}

dml::Expression DmlNetworkBuilder::Input(dml::TensorDesc desc, bool constantData)
{
    if (constantData)
    {
        desc.flags |= DML_TENSOR_FLAG_OWNED_BY_DML;
    }
    else if (desc.flags & DML_TENSOR_FLAG_OWNED_BY_DML)
    {
        throw std::invalid_argument("Only constant inputs should be flagged OWNED_BY_DML");
    }

    auto input = dml::InputTensor(m_graph, m_currentInputIndex++, desc);

    DmlNetwork::TensorDesc tensorDesc = {};
    tensorDesc.sizeInBytes = input.GetOutputDesc().totalTensorSizeInBytes;
    tensorDesc.flags = desc.flags;
    m_inputs.emplace_back(tensorDesc);

    return input;
}

dml::Expression DmlNetworkBuilder::Conv2d(dml::Expression input, uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize, bool includeBias)
{
    auto filter = ConstantInput(dml::TensorDesc(input.GetOutputDesc().dataType, { outChannels, inChannels, kernelSize, kernelSize }));

    std::optional<dml::Expression> bias;
    if (includeBias)
    {
        bias = ConstantInput(dml::TensorDesc(input.GetOutputDesc().dataType, { 1, outChannels, 1, 1 }));
    }

    std::array<uint32_t, 2> padding = { kernelSize / 2, kernelSize / 2 };
    return dml::ConvolutionBuilder(input, filter, bias).StartPadding(padding).EndPadding(padding).Build();
}