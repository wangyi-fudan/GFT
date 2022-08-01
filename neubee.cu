#ifndef	neubee_included
#define	neubee_included
#include	<cuda_runtime.h>
#include	<cublas_v2.h>
#include	<algorithm>
#include	<iostream>
#include	<cstring>
#include	<cfloat>
#include	<cstdio>
#include	<vector>
#include	<cmath>
#include	<omp.h>
using	namespace	std;

namespace	neubee{
const	size_t	gpu_threads=16;
uint64_t	global_seed=time(NULL);
cublasHandle_t	handle;

static inline uint64_t wyrand(uint64_t *seed){
	*seed+=0xa0761d6478bd642full;
	__uint128_t	a=(__uint128_t)(*seed)*(*seed^0xe7037ed1a0b428dbull);
	return (uint64_t)(a>>64)^(uint64_t)a;
}
static inline double wy2u01(uint64_t r){ const double _wynorm=1.0/(1ull<<52); return (r>>12)*_wynorm;}
static inline double wy2gau(uint64_t r){ const double _wynorm=1.0/(1ull<<20); return ((r&0x1fffff)+((r>>21)&0x1fffff)+((r>>42)&0x1fffff))*_wynorm-3.0;}
static inline uint64_t wy2u0k(uint64_t r, uint64_t k){ 	__uint128_t	a=(__uint128_t)r*k; return	a>>64;	}

template<size_t	R,	size_t	C>
struct	Matrix{
	float	*data;
	Matrix(){	cudaMallocManaged(&data,	R*C*sizeof(float));	zero();	}
	~Matrix(){	cudaFree(data);	}
	size_t	size(void){	return	R*C;	}
	float*	operator()(size_t	c){	return	data+c*R;	}
	void	randomize(float	norm=1){	for(size_t	i=0;	i<R*C;	i++)	data[i]=norm*wy2gau(wyrand(&global_seed));	}
	void	zero(void){	cudaMemset(data,	0,	R*C*sizeof(float));	}
};

struct	IOFile{
	vector<float*>	ptr;
	vector<uint64_t>	siz;
	uint64_t	size=0;
	IOFile(){	cublasCreate(&handle);	}
	~IOFile(){	cublasDestroy(handle);	}
	void	bind(float	*p,	uint64_t	n){	ptr.push_back(p);	siz.push_back(n);	size+=n;	}
	void	save(const	char	*F){
		FILE	*f=fopen(F,"wb");
		for(size_t	i=0;	i<ptr.size();	i++){
			for(size_t	j=0;	j<siz[i];	j++)	fwrite(((uint16_t*)(ptr[i]+j))+1,2,1,f);
		}
		fclose(f);
	}
	bool	load(const	char	*F){
		FILE	*f=fopen(F,"rb");
		if(f==NULL)	return	false;
		for(size_t	i=0;	i<ptr.size();	i++){
			cudaMemset(ptr[i],0,siz[i]*sizeof(float));
			for(size_t	j=0;	j<siz[i];	j++)	if(fread(((uint16_t*)(ptr[i]+j))+1,2,1,f)!=1)	return	false;
		}	
		fclose(f);
		return	true;
	}
}iofile;

__global__	void	dasf(size_t	n,	float	*a,	float	*b){	size_t	id=blockIdx.x*blockDim.x+threadIdx.x;	a[id]=1.520866623f*sinf(b[id]);	}
__global__	void	dasb(size_t	n,	float	*a,	float	*b,	float	*c){	size_t	id=blockIdx.x*blockDim.x+threadIdx.x;	a[id]=1.520866623f*cosf(b[id])*c[id];	}
template<size_t	R,	size_t	C>
struct	af_sin{
	Matrix<R,C>	out,gra;
	void	forw(Matrix<R,C>	&inp){	dasf<<<R*C/gpu_threads,gpu_threads>>>(R*C,out.data,inp.data);	}
	void	back(Matrix<R,C>	&inp,	Matrix<R,C>	&gin){	dasb<<<R*C/gpu_threads,gpu_threads>>>(R*C,gra.data,inp.data,gin.data);	}
};


__global__	void	dsmf(size_t	R,	size_t	C,	float	*a,	float	*b){
	size_t	id=blockIdx.x*blockDim.x+threadIdx.x;
	float	*p=a+id*R,	*q=b+id*R,	sum=0;
	for(size_t	i=0;	i<R;	i++)	sum+=(p[i]=expf(q[i]));
	for(size_t	i=0;	i<R;	i++)	p[i]/=sum;
}

__global__	void	dsmb(size_t	R,	size_t	C,	float	*gin,	float	*out,	float	*gra){
	size_t	id=blockIdx.x*blockDim.x+threadIdx.x;
	float	*gi=gin+id*R,	*ou=out+id*R,	*gr=gra+id*R,	sum=0;
	for(size_t	i=0;	i<R;	i++)	sum+=ou[i]*gi[i];
	for(size_t	i=0;	i<R;	i++)	gr[i]=ou[i]*(gi[i]-sum);
}

template<size_t	R,	size_t	C>
struct	softmax{
	Matrix<R,C>	out,gra;
	void	forw(Matrix<R,C>	&inp){	dsmf<<<C/gpu_threads,gpu_threads>>>(R,C,out.data,inp.data);	}
	void	back(Matrix<R,C>	&inp,	Matrix<R,C>	&gin){	dsmb<<<C/gpu_threads,gpu_threads>>>(R,C,gin.data,out.data,gra.data);	}
};

__global__	void	dlnf(size_t	R,	size_t	C,	float	*inp,	float	*out,	float	*norm){
	size_t	id=blockIdx.x*blockDim.x+threadIdx.x;
	float	*in=inp+id*R,	*ou=out+id*R,	sum=0,	nor=0;
	for(size_t	i=0;	i<R;	i++)	sum+=in[i];
	sum/=R;
	for(size_t	i=0;	i<R;	i++){	float	o=in[i]-sum;	nor+=o*o;	}
	norm[id]=nor;	nor=sqrtf(R/nor);
	for(size_t	i=0;	i<R;	i++)	ou[i]=(in[i]-sum)*nor;
}

__global__	void	dlnb(size_t	R,	size_t	C,	float	*gin,	float	*out,	float	*gra,	float	*norm){
	size_t	id=blockIdx.x*blockDim.x+threadIdx.x;
	float	*gi=gin+id*R,	*ou=out+id*R,	*gr=gra+id*R,	mg=0,	s=sqrtf(R/norm[id]),	sum=0;
	for(size_t	i=0;	i<R;	i++)	mg+=ou[i]*gi[i];
	mg/=norm[id];
	for(size_t	i=0;	i<R;	i++)	sum+=(gr[i]=s*gi[i]-(mg/s)*ou[i]);
	sum/=R;
	for(size_t	i=0;	i<R;	i++)	gr[i]-=sum;
}

template<size_t	R,	size_t	C>
struct	layernorm{
	Matrix<C,1>	norm;
	Matrix<R,C>	out,gra;
	void	forw(Matrix<R,C>	&inp){	dlnf<<<C/gpu_threads,gpu_threads>>>(R,C,inp.data,out.data,norm.data);	}
	void	back(Matrix<R,C>	&inp,	Matrix<R,C>	&gin){	dlnb<<<C/gpu_threads, gpu_threads>>>(R,C,gin.data,out.data,gra.data,norm.data);	}
};

__global__	void	dlnf1(size_t	R,	size_t	C,	float	*inp,	float	*out,	float	*norm,	float	*c){
	size_t	id=blockIdx.x*blockDim.x+threadIdx.x;
	float	*in=inp+id*R,	*ou=out+id*R,	sum=0,	nor=0;
	for(size_t	i=0;	i<R;	i++)	sum+=in[i];
	sum/=R;
	for(size_t	i=0;	i<R;	i++){	float	o=in[i]-sum;	nor+=o*o;	}
	norm[id]=nor;	nor=sqrtf(R/nor)*(*c);
	for(size_t	i=0;	i<R;	i++)	ou[i]=(in[i]-sum)*nor;
}

__global__	void	dlnb1(size_t	R,	size_t	C,	float	*gin,	float	*out,	float	*gra,	float	*norm,	float	*c){
	size_t	id=blockIdx.x*blockDim.x+threadIdx.x;
	float	*gi=gin+id*R,	*ou=out+id*R,	*gr=gra+id*R,	mg=0,	s=sqrtf(R/norm[id])*(*c),	sum=0;
	for(size_t	i=0;	i<R;	i++)	mg+=ou[i]*gi[i];
	mg/=norm[id];
	for(size_t	i=0;	i<R;	i++)	sum+=(gr[i]=s*gi[i]-(mg/s)*ou[i]);
	sum/=R;
	for(size_t	i=0;	i<R;	i++)	gr[i]-=sum;
}

template<size_t	R,	size_t	C>
struct	layernormc{
	Matrix<1,1>	c;
	Matrix<C,1>	norm;
	Matrix<R,C>	out,gra;
	layernormc(){	iofile.bind(c.data,c.size());	c.data[0]=1;	}
	void	forw(Matrix<R,C>	&inp){	dlnf1<<<C/gpu_threads,gpu_threads>>>(R,C,inp.data,out.data,norm.data,c.data);	}
	void	back(Matrix<R,C>	&inp,	Matrix<R,C>	&gin){	
		float	g;	cublasSdot(handle,R*C,gin.data,1,out.data,1,&g);
		dlnb1<<<C/gpu_threads, gpu_threads>>>(R,C,gin.data,out.data,gra.data,norm.data,c.data);	
		float	z=logf(expf(c.data[0])-1)-g/sqrtf(R*C);
		c.data[0]=log1pf(expf(z));
	}
};

template<size_t	R0,	size_t	R1,	size_t	C>
struct	row2row{
	Matrix<R0,R1>	wei;
	Matrix<R1,C>	out;
	Matrix<R0,C>	gra;
	row2row(){	iofile.bind(wei.data,wei.size());	wei.randomize();	}
	void	forw(Matrix<R0,C>	&inp){		
		float	alf=1/sqrtf(R0),	bet=0;
		cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, R1, C, R0, &alf, wei.data, CUDA_R_32F,R0, inp.data, CUDA_R_32F,R0, &bet, out.data, CUDA_R_32F,R1,CUBLAS_COMPUTE_32F_FAST_16BF,CUBLAS_GEMM_DEFAULT); 
	}
	void	back(Matrix<R0,C>	&inp,	Matrix<R1,C>	&gin){
		float	alf=1/sqrtf(R0),	bet=0;
		cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, R0, C, R1, &alf, wei.data, CUDA_R_32F,R0, gin.data, CUDA_R_32F,R1, &bet, gra.data, CUDA_R_32F,R0,CUBLAS_COMPUTE_32F_FAST_16BF,CUBLAS_GEMM_DEFAULT);
		float	alf1=-1/sqrt(R0),	bet1=1;
		cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, R0, R1, C, &alf1, inp.data, CUDA_R_32F,R0, gin.data, CUDA_R_32F,R1, &bet1, wei.data, CUDA_R_32F,R0,CUBLAS_COMPUTE_32F_FAST_16BF,CUBLAS_GEMM_DEFAULT);
	}
};

__global__	void	dcsmf(size_t	R,	size_t	C,	float	*b,	float	*p){
	size_t	id=blockIdx.x*blockDim.x+threadIdx.x,	c=id%R,h=id/R;
	float	*q=b+id*R,	*p1=p+h*R+c;
	for(size_t	i=0;	i<=c;	i++)	q[i]=1/(1+expf(-q[i]-*(p1-i)));
	for(size_t	i=c+1;	i<R;	i++)	q[i]=0;
}

__global__	void	dcsmb(size_t	R,	size_t	C,	float	*gin,	float	*out){
	size_t	id=blockIdx.x*blockDim.x+threadIdx.x,	c=id%R;
	float	*gi=gin+id*R,	*ou=out+id*R;
	for(size_t	i=0;	i<=c;	i++)	gi[i]*=ou[i]*(1-ou[i]);
	for(size_t	i=c+1;	i<R;	i++)	gi[i]=0;
}

__global__	void	dsahb(size_t	C,	float	*a,	float	*p){
	size_t  id=blockIdx.x*blockDim.x+threadIdx.x,	h=id/C,	c=id%C;
	float	s=0,	*m=a+h*C*C;
	for(size_t	i=c;	i<C;	i++)	s+=m[i*C+(i-c)];
	p[id]-=s;
}

template<size_t	R,	size_t	C,	size_t	heads>
struct	attention{
	static	Matrix<C,C*heads>	da;
	static	Matrix<R,C>	dqk;
	static	Matrix<R,C>	dv;
	Matrix<C,heads>	pem;
	row2row<R,R,C>	qk;
	row2row<R,R,C>	v0;
	af_sin<R,C>	a0;
	row2row<R,R,C>	v;
	Matrix<C,C*heads>	a;
	Matrix<R,C>	out,	&gra=qk.gra;
	attention(){	iofile.bind(pem.data,pem.size());	for(size_t	h=0;	h<heads;	h++)	for(size_t	i=0;	i<C;	i++)	pem(h)[i]=-2*log1pf(i);	}
	void	forw(Matrix<R,C>	&inp){
		qk.forw(inp);	v0.forw(inp);	a0.forw(v0.out);	v.forw(a0.out);
		float	alf=1/sqrtf(R/heads/2),bet=0;
		cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, C, C, R/heads/2, &alf, qk.out.data+R/2, CUDA_R_32F,R, R/heads/2,	qk.out.data, CUDA_R_32F,R, R/heads/2, &bet, a.data, CUDA_R_32F,C, C*C,	heads,CUBLAS_COMPUTE_32F_FAST_16BF,CUBLAS_GEMM_DEFAULT);
		dcsmf<<<C*heads/gpu_threads,gpu_threads>>>(C,C*heads,a.data,pem.data);
		float	alf1=1,	bet1=0;
		cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, R/heads, C, C, &alf1, v.out.data, CUDA_R_32F,R, R/heads,a.data,	CUDA_R_32F,C, C*C,	&bet1, out.data, CUDA_R_32F,R,	R/heads,	heads,CUBLAS_COMPUTE_32F_FAST_16BF,CUBLAS_GEMM_DEFAULT);
	}
	void	back(Matrix<R,C>	&inp,	Matrix<R,C>	&gin){
		float	alf=1,bet=0;
		cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, R/heads, C, C, &alf, gin.data, CUDA_R_32F,R, R/heads, a.data, CUDA_R_32F,C, C*C,	&bet, dv.data, CUDA_R_32F,R,	R/heads,	heads,CUBLAS_COMPUTE_32F_FAST_16BF,CUBLAS_GEMM_DEFAULT);		
		cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, C, C, R/heads, &alf, v.out.data, CUDA_R_32F,R, R/heads,	gin.data, CUDA_R_32F,R, R/heads,	&bet, da.data, CUDA_R_32F,C,	C*C,	heads,CUBLAS_COMPUTE_32F_FAST_16BF,CUBLAS_GEMM_DEFAULT);
		float	alf1=1/sqrtf(R/heads/2),	bet1=0;
		dcsmb<<<C*heads/gpu_threads,gpu_threads>>>(C,C*heads,da.data,a.data);
		dsahb<<<heads*C/gpu_threads,gpu_threads>>>(C, da.data, pem.data);
		cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, R/heads/2, C, C, &alf1, qk.out.data+R/2, CUDA_R_32F,R, R/heads/2,	da.data, CUDA_R_32F,C, C*C,	&bet1, dqk.data, CUDA_R_32F,R,	R/heads/2,	heads,CUBLAS_COMPUTE_32F_FAST_16BF,CUBLAS_GEMM_DEFAULT);
		cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, R/heads/2, C, C, &alf1, qk.out.data, CUDA_R_32F,R, R/heads/2,	da.data, CUDA_R_32F,C, C*C,	&bet1, dqk.data+R/2, CUDA_R_32F,R,	R/heads/2,	heads,CUBLAS_COMPUTE_32F_FAST_16BF,CUBLAS_GEMM_DEFAULT);
		qk.back(inp,dqk);	v.back(a0.out,dv);	a0.back(v0.out,v.gra);	v0.back(inp,a0.gra);
		cublasSaxpy(handle,R*C,&alf,v0.gra.data,1,gra.data,1);		
	}		
};

template<size_t	R,	size_t	C,	size_t	heads>
Matrix<C,C*heads>	attention<R,C,heads>::da;
template<size_t	R,	size_t	C,	size_t	heads>
Matrix<R,C>	attention<R,C,heads>::dqk;
template<size_t	R,	size_t	C,	size_t	heads>
Matrix<R,C>	attention<R,C,heads>::dv;


template<size_t	R,	size_t	C,	size_t	heads>
struct	GFT{
	layernorm<R,C>	n0;
	attention<R,C,heads>	at;
	layernorm<R,C>	n1;
	row2row<R,R,C>	wo;
	Matrix<R,C>	out,gra;
	void	forw(Matrix<R,C>	&inp){
		n0.forw(inp);
		at.forw(n0.out);
		n1.forw(at.out);
		wo.forw(n1.out);		
		float	alf=1;
		cublasScopy(handle,R*C,wo.out.data,1,out.data,1);
		cublasSaxpy(handle,R*C,&alf,inp.data,1,out.data,1);
	}
	void	back(Matrix<R,C>	&inp,	Matrix<R,C>	&gin){
		wo.back(n1.out,gin);
		n1.back(at.out,wo.gra);
		at.back(n0.out,n1.gra);
		n0.back(inp,at.gra);
		float	alf=1;	
		cublasScopy(handle,R*C,n0.gra.data,1,gra.data,1);
		cublasSaxpy(handle,R*C,&alf,gin.data,1,gra.data,1);
	}
};
}
#endif
