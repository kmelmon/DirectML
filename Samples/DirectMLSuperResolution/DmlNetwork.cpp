#include "pch.h"
#include "DmlNetwork.h"

using Microsoft::WRL::ComPtr;

DmlNetwork::DmlNetwork(IDMLCompiledOperator* op, std::vector<DmlNetwork::TensorDesc>&& inputDescs, std::vector<DmlNetwork::TensorDesc>&& outputDescs) :
    m_dmlGraph(op),
    m_inputDescs(std::move(inputDescs)),
    m_outputDescs(std::move(outputDescs))
{
    m_inputResources.resize(m_inputDescs.size());
    m_outputResources.resize(m_outputDescs.size());
}

//void Network::Initialize(DxDevice& device)
//{
//    ComPtr<IDMLOperatorInitializer> networkInitializer;
//    THROW_IF_FAILED(device.DML()->CreateOperatorInitializer(1, m_dmlGraph.GetAddressOf(), IID_PPV_ARGS(&networkInitializer)));
//
//    DML_BINDING_PROPERTIES initBindingProps = networkInitializer->GetBindingProperties();
//    DML_BINDING_PROPERTIES execBindingProps = m_dmlGraph->GetBindingProperties();
//
//    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc =
//    {
//        .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
//        // Not all networks require descriptors, but D3D cannot create an empty heap. Ideally reserve from existing heap.
//        .NumDescriptors = std::max(1u, initBindingProps.RequiredDescriptorCount),
//        .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
//    };
//
//    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
//    THROW_IF_FAILED(device.D3D()->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&descriptorHeap)));
//
//    DML_BINDING_TABLE_DESC bindingTableDesc =
//    {
//        .Dispatchable = networkInitializer.Get(),
//        .CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart(),
//        .GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart(),
//        .SizeInDescriptors = initBindingProps.RequiredDescriptorCount,
//    };
//
//    ComPtr<IDMLBindingTable> bindingTable;
//    THROW_IF_FAILED(device.DML()->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&bindingTable)));
//
//    std::vector<DML_BUFFER_BINDING> inputBufferBindings(m_inputDescs.size());
//    constexpr uint64_t bufferAlignment = DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT - 1ui64;
//    uint64_t constantBufferOffset = 0;
//
//    for (size_t i = 0; i < m_inputDescs.size(); i++)
//    {
//        auto& tensorDesc = m_inputDescs[i];
//
//        if (tensorDesc.flags & DML_TENSOR_FLAG_OWNED_BY_DML)
//        {
//            // Each constant input is suballocated from a single large buffer, which is bound here in Initialize().
//            inputBufferBindings[i].Offset = constantBufferOffset;
//            inputBufferBindings[i].SizeInBytes = tensorDesc.sizeInBytes;
//
//            uint64_t roundedSizeInBytes = (tensorDesc.sizeInBytes + bufferAlignment) & ~bufferAlignment;
//            constantBufferOffset += roundedSizeInBytes;
//        }
//        else
//        {
//            // Each dynamic input gets its own resource, which will be bound in Execute().
//            inputBufferBindings[i].Offset = 0;
//            inputBufferBindings[i].SizeInBytes = 0;
//            m_inputResources[i] = device.CreateDefaultBuffer(tensorDesc.sizeInBytes);
//        }
//    }
//
//    // Each network output is allocated as a separate resource.
//    for (size_t i = 0; i < m_outputResources.size(); i++)
//    {
//        m_outputResources[i] = device.CreateDefaultBuffer(m_outputDescs[i].sizeInBytes);
//    }
//
//    ComPtr<ID3D12Resource> constantInputStagingBuffer;
//    if (constantBufferOffset > 0)
//    {
//        constantInputStagingBuffer = device.CreateDefaultBuffer(constantBufferOffset);
//
//        for (size_t i = 0; i < m_inputDescs.size(); i++)
//        {
//            if (m_inputDescs[i].flags & DML_TENSOR_FLAG_OWNED_BY_DML)
//            {
//                inputBufferBindings[i].Buffer = constantInputStagingBuffer.Get();
//            }
//        }
//
//        DML_BUFFER_ARRAY_BINDING bufferArrayBindings = {};
//        bufferArrayBindings.BindingCount = inputBufferBindings.size();
//        bufferArrayBindings.Bindings = inputBufferBindings.data();
//
//        DML_BINDING_DESC bindingDesc = {};
//        bindingDesc.Desc = &bufferArrayBindings;
//        bindingDesc.Type = DML_BINDING_TYPE_BUFFER_ARRAY;
//
//        bindingTable->BindInputs(1, &bindingDesc);
//    }
//
//    ComPtr<ID3D12Resource> tempBuffer;
//    auto tempBufferSize = initBindingProps.TemporaryResourceSize;
//    if (tempBufferSize > 0)
//    {
//        tempBuffer = device.CreateDefaultBuffer(tempBufferSize);
//        DML_BUFFER_BINDING bufferBinding = { tempBuffer.Get(), 0, tempBufferSize };
//        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
//        bindingTable->BindTemporaryResource(&bindingDesc);
//    }
//
//    auto persistentBufferSize = execBindingProps.PersistentResourceSize;
//    if (persistentBufferSize > 0)
//    {
//        m_persistentResource = device.CreateDefaultBuffer(persistentBufferSize);
//        DML_BUFFER_BINDING bufferBinding = { m_persistentResource.Get(), 0, persistentBufferSize };
//        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
//        bindingTable->BindOutputs(1, &bindingDesc);
//    }
//
//    THROW_IF_FAILED(device.CommandList()->Reset(device.CommandAllocator(), nullptr));
//    device.CommandList()->SetDescriptorHeaps(1, descriptorHeap.GetAddressOf());
//    device.CommandRecorder()->RecordDispatch(device.CommandList(), networkInitializer.Get(), bindingTable.Get());
//    THROW_IF_FAILED(device.CommandList()->Close());
//
//    ID3D12CommandList* commandLists[] = { device.CommandList() };
//    device.CommandQueue()->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
//
//    device.WaitForGpu();
//}
//
//void Network::Execute(DxDevice& device)
//{
//    auto bindingProps = m_dmlGraph->GetBindingProperties();
//
//    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc =
//    {
//        .Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
//        // Not all networks require descriptors, but D3D cannot create an empty heap. Ideally reserve from existing heap.
//        .NumDescriptors = std::max(1u, bindingProps.RequiredDescriptorCount),
//        .Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
//    };
//
//    ComPtr<ID3D12DescriptorHeap> descriptorHeap;
//    THROW_IF_FAILED(device.D3D()->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&descriptorHeap)));
//
//    DML_BINDING_TABLE_DESC bindingTableDesc =
//    {
//        .Dispatchable = m_dmlGraph.Get(),
//        .CPUDescriptorHandle = descriptorHeap->GetCPUDescriptorHandleForHeapStart(),
//        .GPUDescriptorHandle = descriptorHeap->GetGPUDescriptorHandleForHeapStart(),
//        .SizeInDescriptors = bindingProps.RequiredDescriptorCount,
//    };
//
//    ComPtr<IDMLBindingTable> bindingTable;
//    THROW_IF_FAILED(device.DML()->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&bindingTable)));
//
//    // Bind inputs
//    if (!m_inputDescs.empty())
//    {
//        std::vector<DML_BUFFER_BINDING> buffers(m_inputDescs.size());
//        std::vector<DML_BINDING_DESC> bindings(m_inputDescs.size());
//
//        for (uint32_t i = 0; i < m_inputDescs.size(); i++)
//        {
//            if (!(m_inputDescs[i].flags & DML_TENSOR_FLAG_OWNED_BY_DML))
//            {
//                buffers[i].Buffer = m_inputResources[i].Get();
//                buffers[i].Offset = 0;
//                buffers[i].SizeInBytes = m_inputDescs[i].sizeInBytes;
//                bindings[i].Type = DML_BINDING_TYPE_BUFFER;
//                bindings[i].Desc = &buffers[i];
//            }
//            else
//            {
//                bindings[i].Type = DML_BINDING_TYPE_NONE;
//                bindings[i].Desc = nullptr;
//            }
//        }
//
//        bindingTable->BindInputs(static_cast<uint32_t>(bindings.size()), bindings.data());
//    }
//
//    // Bind outputs
//    if (!m_outputDescs.empty())
//    {
//        std::vector<DML_BUFFER_BINDING> buffers(m_outputDescs.size());
//        std::vector<DML_BINDING_DESC> bindings(m_outputDescs.size());
//
//        for (uint32_t i = 0; i < m_outputDescs.size(); i++)
//        {
//            buffers[i].Buffer = m_outputResources[i].Get();
//            buffers[i].Offset = 0;
//            buffers[i].SizeInBytes = m_outputDescs[i].sizeInBytes;
//            bindings[i].Type = DML_BINDING_TYPE_BUFFER;
//            bindings[i].Desc = &buffers[i];
//        }
//
//        bindingTable->BindOutputs(static_cast<uint32_t>(bindings.size()), bindings.data());
//    }
//
//    // Bind persistent resource
//    if (m_persistentResource)
//    {
//        DML_BUFFER_BINDING buffer = { .Buffer = m_persistentResource.Get(), .Offset = 0, .SizeInBytes = m_persistentResource->GetDesc().Width };
//        DML_BINDING_DESC binding = { .Type = DML_BINDING_TYPE_BUFFER, .Desc = &buffer };
//        bindingTable->BindPersistentResource(&binding);
//    }
//
//    // Bind temporary resource
//    ComPtr<ID3D12Resource> tempResource;
//    if (bindingProps.TemporaryResourceSize)
//    {
//        tempResource = device.CreateDefaultBuffer(bindingProps.TemporaryResourceSize);
//        DML_BUFFER_BINDING buffer = { .Buffer = tempResource.Get(), .Offset = 0, .SizeInBytes = bindingProps.TemporaryResourceSize };
//        DML_BINDING_DESC binding = { .Type = DML_BINDING_TYPE_BUFFER, .Desc = &buffer };
//        bindingTable->BindTemporaryResource(&binding);
//    }
//
//    THROW_IF_FAILED(device.CommandList()->Reset(device.CommandAllocator(), nullptr));
//    device.CommandList()->SetDescriptorHeaps(1, descriptorHeap.GetAddressOf());
//    device.CommandRecorder()->RecordDispatch(device.CommandList(), m_dmlGraph.Get(), bindingTable.Get());
//    THROW_IF_FAILED(device.CommandList()->Close());
//
//    ID3D12CommandList* commandLists[] = { device.CommandList() };
//    device.CommandQueue()->ExecuteCommandLists(ARRAYSIZE(commandLists), commandLists);
//
//    device.WaitForGpu();
//}