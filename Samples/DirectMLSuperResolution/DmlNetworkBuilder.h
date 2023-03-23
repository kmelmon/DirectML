#pragma once

#include "Common.h"
#include "DmlNetwork.h"

class DmlNetworkBuilder
{
public:
    DmlNetworkBuilder(std::wstring_view name, IDMLDevice* device) : m_graph(device), m_name(name) {}

    DmlNetwork Compile(DML_EXECUTION_FLAGS flags = DML_EXECUTION_FLAG_NONE);

protected:
    dml::NameScope NameScope(std::string&& name);
    void PushNameScope(std::string&& name);
    void PopNameScope();

    // Adds an input to the graph. The tensor data may be dynamic (bound at inference/execution) 
    // or constant (bound at initialization). Constant inputs are typically used for model parameters.
    dml::Expression Input(dml::TensorDesc desc, bool constantData = false);
    dml::Expression ConstantInput(dml::TensorDesc desc) { return Input(desc, true); }

    // Adds a conv2d layer with 'same' padding, similar to PyTorch Conv2d.
    dml::Expression Conv2d(dml::Expression input, uint32_t inChannels, uint32_t outChannels, uint32_t kernelSize, bool includeBias = true);

protected:
    dml::Graph m_graph;
    uint32_t m_currentInputIndex = 0;
    std::vector<DmlNetwork::TensorDesc> m_inputs;
    std::vector<dml::Expression> m_outputs;
    std::wstring m_name;
};