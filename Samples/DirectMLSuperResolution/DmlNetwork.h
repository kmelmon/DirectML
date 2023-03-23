#pragma once

//#include "Windows/AllowWindowsPlatformTypes.h"
#include <wrl/client.h>
#include <vector>
#pragma warning (disable : 4668)
#include "DirectML.h"
//#include "DxDevice.h"
//#include "Windows/HideWindowsPlatformTypes.h"

class DmlNetwork
{
public:
    struct TensorDesc
    {
        uint64_t sizeInBytes;
        DML_TENSOR_FLAGS flags;
    };

    DmlNetwork(
        // Compiled DML graph representing the network.
        IDMLCompiledOperator* op,

        // Tensor descriptions for resources bound to the DML graph input slots.
        std::vector<TensorDesc>&& inputDescs,

        // Tensor descriptions for resources bound to the DML graph output slots.
        std::vector<TensorDesc>&& outputDescs
    );

    // Allocates resources and initializes underlying DML graph. Should only be called once.
    // TODO: allow binding CPU data for initializers.
    //void Initialize(DxDevice& device);
    
    //void Execute(DxDevice& device);

private:
    Microsoft::WRL::ComPtr<IDMLCompiledOperator> m_dmlGraph;
    std::vector<TensorDesc> m_inputDescs;
    std::vector<TensorDesc> m_outputDescs;

    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> m_inputResources;
    Microsoft::WRL::ComPtr<ID3D12Resource> m_persistentResource;
    std::vector<Microsoft::WRL::ComPtr<ID3D12Resource>> m_outputResources;
};