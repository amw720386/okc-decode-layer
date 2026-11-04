
// A3: the base Gluon lane mapping and floating-point operation order, across
// eight 128-thread CTAs. This source is embedded in the single-file candidate.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <stdexcept>
#include <stdint.h>
namespace cg = cooperative_groups;
using BF = __nv_bfloat16;
using TV = tvm::ffi::TensorView;

struct GlueArgs {
  const BF *branch, *post, *pre;
  const float *part, *stream, *alpha, *beta;
  BF *out;
  float *next, *router;
  int rows;
  const float* alpha2;
  float* tail;
};

__device__ __forceinline__ float ga(float x, float y) {
  float z; asm("add.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gm(float x, float y) {
  float z; asm("mul.rn.f32 %0,%1,%2;":"=f"(z):"f"(x),"f"(y)); return z;
}
__device__ __forceinline__ float gf(float x, float y, float a) {
  float z; asm("fma.rn.f32 %0,%1,%2,%3;":"=f"(z):"f"(x),"f"(y),"f"(a)); return z;
}
__device__ __forceinline__ float gb(float x) {
  BF z = __float2bfloat16_rn(x); return __bfloat162float(z);
}
__device__ __forceinline__ float gr(float x, float eps) {
  // start's div.full.f32 by 8192 + eps becomes this FFMA in ptxas SASS.
  float z = gf(x, 0.0001220703125f, eps);
  asm("rsqrt.approx.ftz.f32 %0,%1;":"=f"(z):"f"(z)); return z;
}
template<int V> __device__ __forceinline__ void glf(const float* p, float* x) {
  if constexpr (V == 4) {
    asm volatile("ld.global.v4.f32 {%0,%1,%2,%3},[%4];"
      :"=f"(x[0]),"=f"(x[1]),"=f"(x[2]),"=f"(x[3]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.v2.f32 {%0,%1},[%2];"
      :"=f"(x[0]),"=f"(x[1]):"l"(p):"memory");
  }
}
template<int V> __device__ __forceinline__ void glb(const BF* p, float* x) {
  unsigned u[V/2];
  if constexpr (V == 4) {
    asm volatile("ld.global.v2.b32 {%0,%1},[%2];":"=r"(u[0]),"=r"(u[1]):"l"(p):"memory");
  } else {
    asm volatile("ld.global.b32 %0,[%1];":"=r"(u[0]):"l"(p):"memory");
  }
  #pragma unroll
  for (int j=0;j<V/2;j++) {
    x[2*j]=__uint_as_float(u[j]<<16);
    x[2*j+1]=__uint_as_float(u[j]&0xffff0000u);
  }
}
template<int V> __device__ __forceinline__ void gsf(float* p, const float* x) {
  if constexpr (V == 4) {
    asm volatile("st.global.v4.f32 [%0],{%1,%2,%3,%4};"
      ::"l"(p),"f"(x[0]),"f"(x[1]),"f"(x[2]),"f"(x[3]):"memory");
  } else {
    asm volatile("st.global.v2.f32 [%0],{%1,%2};"
      ::"l"(p),"f"(x[0]),"f"(x[1]):"memory");
  }
}
template<int V> __device__ __forceinline__ void gsb(BF* p, const float* x) {
  unsigned u[V/2];
  #pragma unroll
  for(int j=0;j<V/2;j++) {
    asm("cvt.rn.bf16x2.f32 %0,%1,%2;":"=r"(u[j]):"f"(x[2*j+1]),"f"(x[2*j]));
  }
  if constexpr(V==4) {
    asm volatile("st.global.v2.b32 [%0],{%1,%2};"::"l"(p),"r"(u[0]),"r"(u[1]):"memory");
  } else {
    asm volatile("st.global.b32 [%0],%1;"::"l"(p),"r"(u[0]):"memory");
  }
}

template<int N> __device__ __forceinline__ float local_squares(const float (&x)[N]) {
  float q[N/2];
  #pragma unroll
  for(int j=0;j<N/2;j++) q[j]=gf(x[2*j],x[2*j],gm(x[2*j+1],x[2*j+1]));
  // Gluon contracts each even square into its adjacent odd square, then uses
  // the balanced in-thread tree. No sequential sum or reassociation is allowed.
  #pragma unroll
  for(int width=N/2;width>1;width/=2) {
    #pragma unroll
    for(int j=0;j<width/2;j++) q[j]=ga(q[2*j],q[2*j+1]);
  }
  return q[0];
}
__device__ __forceinline__ float butterfly(float x, int first) {
  #pragma unroll
  for(int d=first;d;d>>=1) x=ga(x,__shfl_xor_sync(0xffffffffu,x,d));
  return x;
}
template<int NW,int N>
__device__ __forceinline__ float cluster_squares(const float (&x)[N], float* sm,
                                                cg::cluster_group cluster, int rank) {
  constexpr int warps_per_cta=NW/8;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sum=butterfly(local_squares(x),16);
  if(lane==0 && warp<warps_per_cta) {
    #pragma unroll
    for(int dst=0;dst<8;dst++) {
      volatile float* p=cluster.map_shared_rank(sm,dst);
      p[rank*warps_per_cta+warp]=sum;
    }
  }
  cluster.sync();
  // Each warp independently reads the original 32 (or 16) warp partials and
  // performs the identical descending XOR butterfly. Separate r1/r2 buffers
  // prevent a fast CTA from overwriting a partial still read by a slow CTA.
  float all=sm[lane&(NW-1)];
  return butterfly(all,NW/2);
}

template<int NW,int N>
__device__ __forceinline__ float cluster_squares_pair(const float (&x)[N], const float (&y)[N],
    float* sm, float* smy, cg::cluster_group cluster, int rank, float& tail) {
  constexpr int warps_per_cta=NW/8;
  int tid=threadIdx.x, lane=tid&31, warp=tid>>5;
  float sx=butterfly(local_squares(x),16);
  float sy=butterfly(local_squares(y),16);
  if(lane==0 && warp<warps_per_cta) {
    #pragma unroll
    for(int dst=0;dst<8;dst++) {
      volatile float* px=cluster.map_shared_rank(sm,dst);
      volatile float* py=cluster.map_shared_rank(smy,dst);
      px[rank*warps_per_cta+warp]=sx;
      py[rank*warps_per_cta+warp]=sy;
    }
  }
  cluster.sync();
  tail=butterfly(smy[lane&(NW-1)],NW/2);
  return butterfly(sm[lane&(NW-1)],NW/2);
}

template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>
__global__ __launch_bounds__(128) void glue_cluster_boundary(GlueArgs a) {
  constexpr int logical_threads=NW*32;
  constexpr int per_cta=logical_threads/8;
  constexpr int N=8192/logical_threads;
  constexpr int repeats=N/V;
  cg::cluster_group cluster=cg::this_cluster();
  int rank=cluster.block_rank(), tid=threadIdx.x;
  int logical=rank*per_cta+(tid%per_cta);
  int row=blockIdx.x/8;
  bool active=tid<per_cta;
  float s[N],alpha[N],beta[N],pre[N],post[N],b[N],ns[N],a2[N],carry[N],masked[N];
  __shared__ float partials[3][NW];
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    glf<V>(a.stream+row*8192+col,s+r*V);
    glf<V>(a.alpha+col,alpha+r*V);
    glf<V>(a.beta+col,beta+r*V);
    glf<V>(a.alpha2+col,a2+r*V);
    glb<V>(a.pre+col,pre+r*V);
    if constexpr(POST) glb<V>(a.post+col,post+r*V);
    if constexpr(SLICED) {
      if(col>=1024) {
        #pragma unroll
        for(int j=0;j<V;j++) {
          int k=r*V+j;
          // Preserve the signed zero from the post weight as well as the
          // sliced boundary's alpha*stream FMA contraction.
          ns[k]=gf(alpha[k],s[k],gm(beta[k],gm(0.f,post[k])));
        }
        // carry is stored after both norms are formed.
      }
    }
  }
  // DSMEM destinations must exist before the first remote store. All work
  // above this barrier and the dependency wait is independent of the branch.
  cluster.sync();
  asm volatile("griddepcontrol.wait; griddepcontrol.launch_dependents;" ::: "memory");
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if constexpr(SLICED) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=0.f;
      if(col<1024) {
        #pragma unroll
        for(int sk=0;sk<8;sk++) {
          float part[V];glf<V>(a.part+(sk*a.rows+row)*1024+col,part);
          #pragma unroll
          for(int j=0;j<V;j++) b[r*V+j]=ga(b[r*V+j],part[j]);
        }
      }
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(b[r*V+j]);
    } else {
      glb<V>(a.branch+row*8192+col,b+r*V);
    }
    if constexpr(POST) {
      #pragma unroll
      for(int j=0;j<V;j++) b[r*V+j]=gb(gm(b[r*V+j],post[r*V+j]));
    }
  }
  float r1=gr(cluster_squares<NW>(b,partials[0],cluster,rank),1.e-8f);
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    if(!SLICED || col<1024) {
      #pragma unroll
      for(int j=0;j<V;j++) {
        int k=r*V+j;
        float bn=gb(gm(b[k],r1));
        if(SLICED || (PRELOADED_BASE && col<4096))
          ns[k]=gf(alpha[k],s[k],gm(beta[k],bn));
        else
          ns[k]=gf(beta[k],bn,gm(alpha[k],s[k]));
      }
      // carry is stored after both norms are formed.
    }
  }
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    #pragma unroll
    for(int j=0;j<V;j++) {
      int k=r*V+j;
      carry[k]=gm(a2[k],ns[k]);
      masked[k]=(col+j>=1024) ? carry[k] : 0.f;
    }
  }
  float tail;
  float r2=gr(cluster_squares_pair<NW>(ns,masked,partials[1],partials[2],cluster,rank,tail),1.e-5f);
  if(rank==0 && tid==0) a.tail[row]=tail;
  #pragma unroll
  for(int r=0;r<repeats;r++) {
    int col=logical*V+r*logical_threads*V;
    float h[V];
    #pragma unroll
    for(int j=0;j<V;j++) h[j]=gm(gm(ns[r*V+j],r2),pre[r*V+j]);
    if(active) {
      gsb<V>(a.out+row*8192+col,h);
      gsf<V>(a.next+row*8192+col,carry+r*V);
      if constexpr(ROUTER) gsf<V>(a.router+row*8192+col,h);
    }
  }
}

template<typename F> void glue_launch(GlueArgs a,F fn,cudaStream_t stream) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim=dim3(a.rows*8);cfg.blockDim=dim3(128);cfg.stream=stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id=cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim={8,1,1};
  attrs[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=attrs;cfg.numAttrs=2;
  auto e=cudaLaunchKernelEx(&cfg,fn,a);
  if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
// graft: boundary 1 (attention residual), BF16 [rows, 8192] branch, rows <= 16, optional FP32 router copy.
void b1(TV branch,TV stream,TV pre,TV alpha,TV beta,TV out,TV next,TV alpha2,TV tail,TV router,bool write_router) {
  GlueArgs a{(const BF*)branch.data_ptr(),(const BF*)pre.data_ptr(),(const BF*)pre.data_ptr(),
             (const float*)branch.data_ptr(),(const float*)stream.data_ptr(),
             (const float*)alpha.data_ptr(),(const float*)beta.data_ptr(),
             (BF*)out.data_ptr(),(float*)next.data_ptr(),(float*)router.data_ptr(),(int)stream.size(0),(const float*)alpha2.data_ptr(),(float*)tail.data_ptr()};
  if(a.rows<1 || a.rows>16) throw std::runtime_error("graft b1: rows must be 1..16");
  auto s=(cudaStream_t)TVMFFIEnvGetStream(stream.device().device_type,stream.device().device_id);
  if(write_router) {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,true,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,true,false>,s);
  } else {
    if(a.rows>1) glue_launch(a,glue_cluster_boundary<4,32,false,false,false,true>,s);
    else glue_launch(a,glue_cluster_boundary<4,32,false,false,false,false>,s);
  }
}
