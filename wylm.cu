#include	"neubee.cu"
using	namespace	neubee;

__global__	void	dembf(size_t	R,	size_t	C,	float	*inp,	uint8_t	*x){
	size_t	col=blockIdx.x*blockDim.x+threadIdx.x;	
	inp[col*R+x[col]]=1.0f;
}

__global__	void	dlossf(int	input,	int	output,	float	*a,	uint8_t	*x,	float	*y){
	float	loss=0;
	for(size_t	i=0;	i<input;	i++){
		loss-=logf(fmaxf(a[i*output+x[i+1]],FLT_MIN));
		a[i*output+x[i+1]]-=1;
	}
	*y=loss;
}
__global__	void	dlossb(int	n,	float	*a,	float	b){
	int	id=blockIdx.x*blockDim.x+threadIdx.x;
	if(id<n)	a[id]*=b;
}

template<size_t	input,	size_t	embedc,	size_t	heads,	size_t	depth,	size_t	output>
struct	wylm{
	uint8_t	*data;
	float	*ret;
	Matrix<output,input>	inp;
	row2row<output,embedc,input>	m0;
	GFT<embedc,input,heads>	tra[depth];
	row2row<embedc,output,input>	out;
	layernormc<output,input>	norm;
	softmax<output,input>	lf;
	wylm(){	cudaMallocManaged(&data,	input+1);	cudaMallocManaged(&ret,	sizeof(float));	for(size_t	i=0;	i<depth;	i++)	tra[i].wo.wei.randomize(1.0/(i+1));	}
	~wylm(){	cudaFree(data);	cudaFree(ret);	}
	float	train(uint8_t	*x,	float	eta){
		cudaMemcpy(data,x,input+1,cudaMemcpyHostToDevice);
		cudaMemset(inp.data,0,inp.size()*sizeof(float));
		dembf<<<input/gpu_threads,gpu_threads>>>(output,input,inp.data,data);
		m0.forw(inp);
		for(size_t	d=0;	d<depth;	d++)	tra[d].forw(d?tra[d-1].out:m0.out);
		out.forw(tra[depth-1].out);
		norm.forw(out.out);
		lf.forw(norm.out);
		dlossf<<<1,1>>>(input,output,lf.out.data,data,ret);
		dlossb<<<(input*output+gpu_threads-1)/gpu_threads,gpu_threads>>>(input*output,lf.out.data,eta);
		norm.back(out.out,lf.out);
		out.back(tra[depth-1].out,norm.gra);
		for(size_t	d=depth-1;	d<depth;	d--)	tra[d].back(d?tra[d-1].out:m0.out,d<depth-1?tra[d+1].gra:out.gra);
		m0.back(inp,tra[0].gra);
		cudaDeviceSynchronize();
		return	*ret;
	}
};

