// Standalone latency/VRAM experiment. Not linked into the inference server.
// Input: little-endian files from the Rust example's export-gpu command.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CU(call) do { auto e=(call); if(e!=cudaSuccess) throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(e)); } while(0)
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t) {return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Query {
    uint32_t valid; float norm,aspect;
    uint8_t pixels[1024],desc[64];
    uint16_t coarse[9][64],fine[9][256];
};
static_assert(sizeof(Query)==6860);
struct Tables { const uint8_t *pixels,*desc; const uint16_t *coarse,*fine; const float *norm,*aspect; int n; };
__device__ uint32_t warp_sum(uint32_t x) {
    for(int i=16;i;i/=2) x+=__shfl_down_sync(0xffffffff,x,i);
    return __shfl_sync(0xffffffff,x,0);
}
__device__ float penalty(float a,float b) {return 0.1f*fminf(fabsf(logf(a/b)),1.0f);}
__device__ float pixel_score(const Query *q,Tables t,int index,int mask) {
    int lane=threadIdx.x%32;float best=0.0f;
    for(int a=0;a<9;++a) {
        if(!(mask&(1<<a))) continue;
        int dy=a/3-1,dx=a%3-1;uint32_t dot=0;
        for(int p=lane;p<1024;p+=32) {
            int y=p/32+dy,x=p%32+dx;
            if(x>=0&&x<32&&y>=0&&y<32) dot+=uint32_t(q->pixels[p])*uint32_t(t.pixels[index*1024+y*32+x]);
        }
        dot=warp_sum(dot);
        best=fmaxf(best,float(dot)/(q->norm*t.norm[index]));
    }
    return fmaxf(0.0f,best-penalty(q->aspect,t.aspect[index]));
}
__global__ void distances(Tables t,const Query *q,uint32_t *out) {
    int index=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(index>=t.n)return;
    uint32_t sum=0;
    for(int i=lane;i<64;i+=32) {int d=int(q->desc[i])-int(t.desc[index*64+i]);sum+=d*d;}
    sum=warp_sum(sum);if(lane==0)out[index]=sum;
}
__global__ void shortlist(Tables t,const Query *q,const uint32_t *indices,float *out,int count) {
    int k=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(k>=count)return;
    float score=pixel_score(q,t,indices[k],0x1ff);
    if(threadIdx.x%32==0)out[k]=score;
}
__global__ void verify(Tables t,const Query *q,float floor,float *out) {
    int index=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x%32;
    if(index>=t.n)return;
    float raw=floor+penalty(q->aspect,t.aspect[index])-0.00001f;
    uint32_t required=uint32_t(fmax(0.0,::floor(double(raw)*268435456.0)));
    int mask=0;
    for(int a=0;a<9;++a) {
        uint32_t dot=0;
        for(int i=lane;i<64;i+=32)dot+=uint32_t(q->coarse[a][i])*uint32_t(t.coarse[index*64+i]);
        if(warp_sum(dot)<required)continue;
        dot=0;
        for(int i=lane;i<256;i+=32)dot+=uint32_t(q->fine[a][i])*uint32_t(t.fine[index*256+i]);
        if(warp_sum(dot)>=required)mask|=1<<(8-a);
    }
    float score=mask?pixel_score(q,t,index,mask):-1.0f;
    if(lane==0)out[index]=score>=floor?score:-1.0f;
}

template<class T> std::vector<T> read(const std::string &path) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    if(!in)throw std::runtime_error("open "+path);
    auto size=in.tellg();if(size<0||size%sizeof(T))throw std::runtime_error("size "+path);
    std::vector<T> result(size/sizeof(T));in.seekg(0);in.read(reinterpret_cast<char*>(result.data()),size);
    if(!in)throw std::runtime_error("read "+path);return result;
}
size_t allocated=0,dict_bytes=0;
template<class T> T *device(size_t n) {T *p;CU(cudaMalloc(&p,n*sizeof(T)));allocated+=n*sizeof(T);return p;}
template<class T> T *upload(const std::vector<T>& v) {T *p=device<T>(v.size());CU(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
template<class T> T *pinned(size_t n) {T *p;CU(cudaMallocHost(&p,n*sizeof(T)));return p;}
struct Best {
    uint32_t a=0,b=0;float sa=-1.0f,sb=-1.0f;
    void add(uint32_t c,float s) {
        if(c==a)sa=std::max(sa,s);
        else if(c==b)sb=std::max(sb,s);
        else if(s>sa||(s==sa&&c<a)) {b=a;sb=sa;a=c;sa=s;}
        else if(s>sb||(s==sb&&c<b)) {b=c;sb=s;}
        if(sb>sa||(sb==sa&&b&&b<a)) {std::swap(a,b);std::swap(sa,sb);}
    }
    bool accepted()const{return b&&sa>=0.93f&&sa-sb>=0.04f;}
};
struct Matcher {
    Tables t;std::vector<uint32_t> labels;
    Query *hq,*dq;uint32_t *hdist,*ddist,*hindices,*dindices;float *hshort,*hfull,*dscores;
    std::vector<std::pair<uint32_t,uint32_t>> order;
    cudaStream_t stream;
    Matcher(const std::string &root) {
        auto norms=read<float>(root+"/norms.bin"),aspects=read<float>(root+"/aspects.bin");
        labels=read<uint32_t>(root+"/labels.bin");int n=norms.size();
        auto pix=read<uint8_t>(root+"/pixels.bin"),desc=read<uint8_t>(root+"/desc.bin");
        auto coarse=read<uint16_t>(root+"/coarse.bin"),fine=read<uint16_t>(root+"/fine.bin");
        if(n<256||aspects.size()!=size_t(n)||labels.size()!=size_t(n)||pix.size()!=size_t(n)*1024||desc.size()!=size_t(n)*64||coarse.size()!=size_t(n)*64||fine.size()!=size_t(n)*256)throw std::runtime_error("invalid dictionary dimensions");
        t={upload(pix),upload(desc),upload(coarse),upload(fine),upload(norms),upload(aspects),n};dict_bytes=allocated;
        hq=pinned<Query>(1);dq=device<Query>(1);hdist=pinned<uint32_t>(n);ddist=device<uint32_t>(n);
        hindices=pinned<uint32_t>(256);dindices=device<uint32_t>(256);hshort=pinned<float>(256);hfull=pinned<float>(n);dscores=device<float>(n);
        order.resize(n);CU(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    }
    Best match(const Query &q,bool &verified) {
        verified=false;if(!q.valid)return {};
        *hq=q;
        CU(cudaMemcpyAsync(dq,hq,sizeof(Query),cudaMemcpyHostToDevice,stream));
        distances<<<(t.n+7)/8,256,0,stream>>>(t,dq,ddist);CU(cudaGetLastError());
        CU(cudaMemcpyAsync(hdist,ddist,t.n*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));CU(cudaStreamSynchronize(stream));
        for(int i=0;i<t.n;++i)order[i]={hdist[i],uint32_t(i)};
        std::nth_element(order.begin(),order.begin()+256,order.end());
        for(int i=0;i<256;++i)hindices[i]=order[i].second;
        CU(cudaMemcpyAsync(dindices,hindices,256*sizeof(uint32_t),cudaMemcpyHostToDevice,stream));
        shortlist<<<32,256,0,stream>>>(t,dq,dindices,dscores,256);CU(cudaGetLastError());
        CU(cudaMemcpyAsync(hshort,dscores,256*sizeof(float),cudaMemcpyDeviceToHost,stream));CU(cudaStreamSynchronize(stream));
        Best result;for(int i=0;i<256;++i)result.add(labels[hindices[i]],hshort[i]);
        if(result.accepted()) {
            verified=true;float floor=result.sa-0.04f;
            verify<<<(t.n+7)/8,256,0,stream>>>(t,dq,floor,dscores);CU(cudaGetLastError());
            CU(cudaMemcpyAsync(hfull,dscores,t.n*sizeof(float),cudaMemcpyDeviceToHost,stream));CU(cudaStreamSynchronize(stream));
            for(int i=0;i<t.n;++i)if(hfull[i]>=floor)result.add(labels[i],hfull[i]);
        }
        return result;
    }
    ~Matcher() {
        cudaStreamDestroy(stream);
        for(auto p:{(void*)t.pixels,(void*)t.desc,(void*)t.coarse,(void*)t.fine,(void*)t.norm,(void*)t.aspect,(void*)dq,(void*)ddist,(void*)dindices,(void*)dscores})cudaFree(p);
        for(auto p:{(void*)hq,(void*)hdist,(void*)hindices,(void*)hshort,(void*)hfull})cudaFreeHost(p);
    }
};
int main(int argc,char**argv)try {
    if(argc!=3)throw std::runtime_error("usage: gpu_bench EXPORT_DIRECTORY REPEATS");
    int repeats=std::stoi(argv[2]);if(repeats<1||repeats>100)throw std::runtime_error("repeats 1..100");
    auto init=Clock::now();CU(cudaSetDevice(0));CU(cudaFree(nullptr));double init_ms=ms(init);
    size_t before,total;CU(cudaMemGetInfo(&before,&total));
    auto started=Clock::now();Matcher matcher(argv[1]);double load_ms=ms(started);
    auto queries=read<Query>(std::string(argv[1])+"/queries.bin");
    for(int r=0;r<2;++r)for(auto &q:queries){bool v;matcher.match(q,v);}
    CU(cudaDeviceSynchronize());size_t after,ignored;CU(cudaMemGetInfo(&after,&ignored));
    cudaDeviceProp prop{};CU(cudaGetDeviceProperties(&prop,0));
    printf("{\"event\":\"memory\",\"device\":\"%s\",\"templates\":%d,\"queries\":%zu,\"dictionary_bytes\":%zu,\"scratch_bytes\":%zu,\"allocated_bytes\":%zu,\"free_before\":%zu,\"free_after\":%zu,\"total_bytes\":%zu,\"init_ms\":%.6f,\"load_ms\":%.6f}\n",prop.name,matcher.t.n,queries.size(),dict_bytes,allocated-dict_bytes,allocated,before,after,total,init_ms,load_ms);fflush(stdout);
    for(int r=0;r<repeats;++r)for(size_t i=0;i<queries.size();++i) {
        bool verified;started=Clock::now();Best b=matcher.match(queries[i],verified);double wall=ms(started);
        printf("{\"event\":\"query\",\"round\":%d,\"index\":%zu,\"valid\":%s,\"accepted\":%s,\"verified\":%s,\"codepoint\":%u,\"score\":%.9g,\"runner_up\":%u,\"runner_score\":%.9g,\"wall_ms\":%.6f}\n",r,i,queries[i].valid?"true":"false",b.accepted()?"true":"false",verified?"true":"false",b.a,b.sa,b.b,b.sb,wall);
    }
    return 0;
} catch(const std::exception &e) {fprintf(stderr,"%s\n",e.what());return 1;}
