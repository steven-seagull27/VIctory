// Standalone Windows SDK D3D12 interpolation readback. The shaders and geometry are original.
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <dxcapi.h>
#include <wrl/client.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstring>

using Microsoft::WRL::ComPtr;
using Bytes=std::vector<uint8_t>;
using Pixel=std::array<float,4>;
constexpr UINT W=128,H=96;
constexpr Pixel clear={-16,-17,-18,-19};
void ok(HRESULT hr,const char* where) { if(FAILED(hr)){std::fprintf(stderr,"%s: 0x%08lx\n",where,hr);throw std::runtime_error(where);} }
Bytes readFile(std::string path){std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error(path);return Bytes(std::istreambuf_iterator<char>(f),{});}
void validate(const char* path,const wchar_t* sdk){
    auto get=[&](const wchar_t* dll){
        std::wstring name=std::wstring(sdk)+L"\\"+dll;
        auto h=LoadLibraryExW(name.c_str(),nullptr,LOAD_WITH_ALTERED_SEARCH_PATH);
        if(!h)throw std::runtime_error("SDK library");
        auto fn=reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(h,"DxcCreateInstance"));
        if(!fn)throw std::runtime_error("SDK factory");
        return fn;
    };
    ComPtr<IDxcLibrary> lib;
    ok(get(L"dxcompiler.dll")(CLSID_DxcLibrary,IID_PPV_ARGS(&lib)),"DXC library");
    ComPtr<IDxcValidator> validator;
    ok(get(L"dxil.dll")(CLSID_DxcValidator,IID_PPV_ARGS(&validator)),"dxil.dll validator");
    auto data=readFile(path);ComPtr<IDxcBlobEncoding> blob;
    ok(lib->CreateBlobWithEncodingFromPinned(data.data(),static_cast<UINT>(data.size()),0,&blob),"blob");
    ComPtr<IDxcOperationResult> res;
    ok(validator->Validate(blob.Get(),DxcValidatorFlags_Default,&res),"Validate");
    HRESULT status;ok(res->GetStatus(&status),"validation status");
    ComPtr<IDxcBlobEncoding> msg;res->GetErrorBuffer(&msg);
    std::printf("validate %s 0x%08lx %s\n",path,status,msg&&msg->GetBufferSize()?static_cast<const char*>(msg->GetBufferPointer()):"");
}
struct Device {
    ComPtr<ID3D12Device> d;ComPtr<ID3D12CommandQueue> queue;ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12Fence> fence;UINT64 signal=0;
    explicit Device(bool warp){
        ComPtr<IDXGIFactory4> factory;ok(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"factory");
        ComPtr<IDXGIAdapter1> adapter;
        if(warp){ok(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)),"EnumWarpAdapter");}
        else {ok(factory->EnumAdapters1(0,&adapter),"EnumAdapters1");}
        DXGI_ADAPTER_DESC1 desc{};ok(adapter->GetDesc1(&desc),"GetDesc1");
        if(bool(desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)!=warp) throw std::runtime_error("adapter type");
        char name[256]{};WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,name,sizeof(name),nullptr,nullptr);
        std::printf("adapter %s vendor=%04x device=%04x software=%d\n",name,desc.VendorId,desc.DeviceId,warp);
        ok(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&d)),"device");
        D3D12_COMMAND_QUEUE_DESC qd{};ok(d->CreateCommandQueue(&qd,IID_PPV_ARGS(&queue)),"queue");
        ok(d->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"fence");
        D3D12_ROOT_SIGNATURE_DESC rd{};rd.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> blob;ok(D3D12SerializeRootSignature(&rd,D3D_ROOT_SIGNATURE_VERSION_1,&blob,nullptr),"serialize root");
        ok(d->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)),"root");
    }
    std::vector<Pixel> draw(const Bytes& vs,const Bytes& ps){
        const auto fmt=DXGI_FORMAT_R32G32B32A32_FLOAT;
        D3D12_GRAPHICS_PIPELINE_STATE_DESC p{};p.pRootSignature=root.Get();p.VS={vs.data(),vs.size()};p.PS={ps.data(),ps.size()};
        p.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;p.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;p.RasterizerState.DepthClipEnable=TRUE;
        auto& rt=p.BlendState.RenderTarget[0];rt.SrcBlend=rt.SrcBlendAlpha=D3D12_BLEND_ONE;rt.DestBlend=rt.DestBlendAlpha=D3D12_BLEND_ZERO;
        rt.BlendOp=rt.BlendOpAlpha=D3D12_BLEND_OP_ADD;rt.LogicOp=D3D12_LOGIC_OP_NOOP;rt.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
        p.SampleMask=UINT_MAX;p.DepthStencilState.DepthEnable=FALSE;p.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
        p.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;p.NumRenderTargets=1;p.RTVFormats[0]=fmt;p.SampleDesc.Count=1;
        ComPtr<ID3D12PipelineState> state;ok(d->CreateGraphicsPipelineState(&p,IID_PPV_ARGS(&state)),"graphics pipeline");
        D3D12_RESOURCE_DESC tex{};tex.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;tex.Width=W;tex.Height=H;tex.DepthOrArraySize=1;tex.MipLevels=1;tex.Format=fmt;tex.SampleDesc.Count=1;tex.Flags=D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE cv{};cv.Format=fmt;std::copy(clear.begin(),clear.end(),cv.Color);
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;heap.CreationNodeMask=heap.VisibleNodeMask=1;
        ComPtr<ID3D12Resource> target;ok(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&tex,D3D12_RESOURCE_STATE_RENDER_TARGET,&cv,IID_PPV_ARGS(&target)),"target");
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};UINT64 total=0;d->GetCopyableFootprints(&tex,0,1,0,&fp,nullptr,nullptr,&total);
        D3D12_RESOURCE_DESC buf{};buf.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;buf.Width=total;buf.Height=1;buf.DepthOrArraySize=1;buf.MipLevels=1;buf.SampleDesc.Count=1;buf.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        heap.Type=D3D12_HEAP_TYPE_READBACK;ComPtr<ID3D12Resource> read;ok(d->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buf,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&read)),"readback");
        D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.NumDescriptors=1;hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV;ComPtr<ID3D12DescriptorHeap> rtv;
        ok(d->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&rtv)),"RTV heap");auto view=rtv->GetCPUDescriptorHandleForHeapStart();d->CreateRenderTargetView(target.Get(),nullptr,view);
        ComPtr<ID3D12CommandAllocator> alloc;ok(d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&alloc)),"allocator");
        ComPtr<ID3D12GraphicsCommandList> list;ok(d->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,alloc.Get(),state.Get(),IID_PPV_ARGS(&list)),"list");
        D3D12_VIEWPORT vp{0,0,float(W),float(H),0,1};D3D12_RECT sc{3,5,121,88};list->SetGraphicsRootSignature(root.Get());list->RSSetViewports(1,&vp);list->RSSetScissorRects(1,&sc);
        list->ClearRenderTargetView(view,cv.Color,0,nullptr);list->OMSetRenderTargets(1,&view,FALSE,nullptr);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);list->DrawInstanced(3,1,0,0);
        D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Transition={target.Get(),0,D3D12_RESOURCE_STATE_RENDER_TARGET,D3D12_RESOURCE_STATE_COPY_SOURCE};list->ResourceBarrier(1,&b);
        D3D12_TEXTURE_COPY_LOCATION src{},dst{};src.pResource=target.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;dst.pResource=read.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=fp;
        list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);ok(list->Close(),"close");ID3D12CommandList* cmds[]={list.Get()};queue->ExecuteCommandLists(1,cmds);ok(queue->Signal(fence.Get(),++signal),"signal");
        HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!event)throw std::runtime_error("create event");
        ok(fence->SetEventOnCompletion(signal,event),"fence event");DWORD wait=WaitForSingleObject(event,8000);CloseHandle(event);if(wait!=WAIT_OBJECT_0)throw std::runtime_error("fence timeout");
        ok(d->GetDeviceRemovedReason(),"device status");D3D12_RANGE r{0,SIZE_T(total)};void* mapped=nullptr;ok(read->Map(0,&r,&mapped),"map");std::vector<Pixel> out(W*H);
        for(UINT y=0;y<H;++y)std::memcpy(&out[y*W],static_cast<const uint8_t*>(mapped)+fp.Offset+y*fp.Footprint.RowPitch,W*sizeof(Pixel));
        D3D12_RANGE none{0,0};read->Unmap(0,&none);return out;
    }
};
Pixel oracle(UINT x,UINT y,bool varying){
    double u=(x+.5)/W,v=(y+.5)/H;
    const double affine = -.5+.375*u+.6875*v;
    if(varying){const double z=1-u/2-v/2+u/4+v/8;u=u/(2*z);v=v/(4*z);}
    return {float(.125+.625*u+.03125*v),float(1.25-.25*u+.125*v),float(.875-.3125*u+.0625*v),float(affine)};
}
int wmain(int argc,wchar_t**argv){
    try{
        if(argc==3&&std::wstring(argv[1])==L"validate"){for(auto fn:{"ps-dxc.dxbc","ps-repeat.dxbc","ps-dedup.dxbc"})validate(fn,argv[2]);return 0;}
        if(argc!=2) return 2;auto which=std::wstring(argv[1]);if(which!=L"hardware"&&which!=L"warp")return 2;
        Device dev(which==L"warp");
        unsigned failure=0;
        for(bool varying:{false,true}){
            auto vs=readFile(varying?"vs-perspective.dxbc":"vs-flat.dxbc");std::vector<Pixel> reference;
            for(auto name:{"dxc","dedup","repeat"}){
                auto data=dev.draw(vs,readFile(std::string("ps-")+name+".dxbc"));unsigned covered=0,guard=0,guardErrors=0,cpuError=0,componentError=0,different=0;float maxErr=0;int sample=-1;
                for(UINT y=0;y<H;++y)for(UINT x=0;x<W;++x){
                    auto i=y*W+x;auto& a=data[i];
                    if(x<3||x>=121||y<5||y>=88){++guard;if(a!=clear)++guardErrors;continue;}
                    ++covered;auto expected=oracle(x,y,varying);bool bad=false;
                    for(UINT k=0;k<4;++k){float e=std::fabs(a[k]-expected[k]);maxErr=std::max(maxErr,e);if(!std::isfinite(a[k])||e>2e-5f){++componentError;bad=true;}}
                    if(bad){++cpuError;if(sample<0)sample=int(i);}
                    if(!reference.empty()&&std::memcmp(&a,&reference[i],sizeof(a)))++different;
                }
                std::printf("case=%s/%s covered=%u sentinel=%u sentinel_errors=%u cpu_pixel_errors=%u cpu_component_errors=%u bit_different_from_dxc=%u max_abs=%g\n",
                         varying?"perspective":"affine",name,covered,guard,guardErrors,cpuError,componentError,different,maxErr);
                if(sample>=0){auto a=data[sample],z=oracle(sample%W,sample/W,varying);std::printf("example=(%d,%d) actual=%g,%g,%g,%g expected=%g,%g,%g,%g\n",sample%W,sample/W,a[0],a[1],a[2],a[3],z[0],z[1],z[2],z[3]);}
                if(reference.empty())reference=data;
                if(guardErrors||cpuError||different)++failure;
            }
        }
        return failure?1:0;
    } catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}
}
