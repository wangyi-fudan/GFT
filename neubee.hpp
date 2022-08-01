#ifndef	neubee_included
#define	neubee_included
#define	EIGEN_DONT_PARALLELIZE
#define	EIGEN_NO_DEBUG
#include	<Eigen/Eigen>
#include	<iostream>
#include	<cstring>
#include	<cfloat>
#include	<cstdio>
#include	<vector>
#include	<cmath>
using	namespace	std;

namespace	neubee{
uint64_t	global_seed=time(NULL);
uint32_t	global_para[16]={};

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
	Matrix(){	data=(float*)aligned_alloc(64,R*C*sizeof(float));	}
	~Matrix(){	free(data);	}
	size_t	size(void){	return	R*C;	}
	float*	operator()(size_t	c){	return	data+c*R;	}
	void	randomize(float	norm=1){	for(size_t	i=0;	i<R*C;	i++)	data[i]=norm*wy2gau(wyrand(&global_seed));	}
	void	zero(void){	memset(data,	0,	R*C*sizeof(float));	}
};

struct	IOFile{
	vector<float*>	ptr;
	vector<uint64_t>	siz;
	uint64_t	size=0;
	void	bind(float	*p,	size_t	n){	ptr.push_back(p);	siz.push_back(n);	size+=n;	}
	bool	load(const	char	*F){
		FILE	*f=fopen(F,"rb");
		if(f==NULL)	return	false;
		for(size_t	i=0;	i<ptr.size();	i++){
			memset(ptr[i],0,siz[i]*sizeof(float));
			for(size_t	j=0;	j<siz[i];	j++)	if(fread(((uint16_t*)(ptr[i]+j))+1,2,1,f)!=1)	return	false;
		}	
		fclose(f);
		return	true;
	}
}iofile;

template<size_t	R,	size_t	C>
struct	af_sin{
	Matrix<R,C>	out;
	void	forw(Matrix<R,C>	&inp){
		float	*p=out(0),	*q=inp(0);
		for(size_t	i=0;	i<R;	i++)	p[i]=1.520866623f*sinf(q[i]);
	}
};

template<size_t	R,	size_t	C>
struct	softmax{
	Matrix<R,C>	out;
	void	forw(Matrix<R,C>	&inp){
		Eigen::Map<Eigen::VectorXf,Eigen::Aligned64>	vi(inp(0),R),	vo(out(0),R);
		float	ma=vi.maxCoeff();
		vo=(vi.array()-ma).exp();
		vo/=vo.sum();
	}
};

template<size_t	R,	size_t	C>
struct	layernorm{
	Matrix<R,C>	out;
	void	forw(Matrix<R,C>	&inp){
		Eigen::Map<Eigen::VectorXf,Eigen::Aligned64>	vi(inp(0),R),	vo(out(0),R);
		vo.array()=vi.array()-vi.sum()/R;
		vo.array()*=sqrtf(R/vo.squaredNorm());
	}
};

template<size_t	R,	size_t	C>
struct	layernormc{
	Matrix<1,1>	c;
	Matrix<R,C>	out;
	layernormc(){	iofile.bind(c.data,c.size());	}
	void	forw(Matrix<R,C>	&inp){
		Eigen::Map<Eigen::VectorXf,Eigen::Aligned64>	vi(inp(0),R),	vo(out(0),R);
		vo.array()=vi.array()-vi.sum()/R;
		vo.array()*=c.data[0]*sqrtf(R/vo.squaredNorm());	
	}
};

template<size_t	R0,	size_t	R1,	size_t	C>
struct	row2row{
	Matrix<R0,R1>	wei;
	Matrix<R1,C>	out;
	row2row(){	iofile.bind(wei.data,wei.size());	wei.randomize();	}
	void	forw(Matrix<R0,C>	&inp){
		Eigen::Map<Eigen::MatrixXf,Eigen::Aligned64>	mwei(wei.data,R0,R1);
		Eigen::Map<Eigen::MatrixXf,Eigen::Aligned64>	minp(inp.data,R0,C);
		Eigen::Map<Eigen::MatrixXf,Eigen::Aligned64>	mout(out.data,R1,C);
		mout.col(0).noalias()=(1/sqrtf(R0))*(mwei.transpose()*minp.col(0));
	}
};

template<size_t	R0,	size_t	R1,	size_t	C>
struct	row2rowsin{
	Matrix<R0,R1>	wei;
	Matrix<R1,C>	out;
	row2rowsin(){	iofile.bind(wei.data,wei.size());	wei.randomize();	}
	void	forw(Matrix<R0,C>	&inp){
		Eigen::Map<Eigen::MatrixXf,Eigen::Aligned64>	mwei(wei.data,R0,R1);
		Eigen::Map<Eigen::VectorXf,Eigen::Aligned64>	minp(inp.data,R0);
		Eigen::Map<Eigen::VectorXf,Eigen::Aligned64>	mout(out.data,R1);
		mout.noalias()=(1/sqrtf(R0))*(mwei.transpose()*minp);
		float	*p=out(0);
		for(size_t	i=0;	i<R1;	i++)	p[i]=1.520866623f*sinf(p[i]);		
	}
};

template<size_t	R0,	size_t	R1,	size_t	C>
struct	row2row1{
	Matrix<R0,R1>	wei;
	Matrix<R1,C>	out;
	row2row1(){	iofile.bind(wei.data,wei.size());	wei.randomize();	}
	void	forw(Matrix<R0,1>	&inp,	size_t	col){
		Eigen::Map<Eigen::MatrixXf,Eigen::Aligned64>	mwei(wei.data,R0,R1);
		Eigen::Map<Eigen::MatrixXf,Eigen::Aligned64>	minp(inp.data,R0,1);
		Eigen::Map<Eigen::MatrixXf,Eigen::Aligned64>	mout(out.data,R1,C);
		mout.col(col).noalias()=(1/sqrtf(R0))*(mwei.transpose()*minp.col(0));
	}
};

template<size_t	R,	size_t	C,	size_t	heads>
struct	attention{
	static	Matrix<C,1>	a;
	Matrix<C,heads>	pem;
	row2row1<R,R,C>	qk;
	row2rowsin<R,R,1>	v0;
	row2row1<R,R,C>	v;
	Matrix<R,1>	out;	
	attention(){	iofile.bind(pem.data,pem.size());	}
	void	forw(Matrix<R,1>	&inp,	size_t	col){
		qk.forw(inp,col);	v0.forw(inp);	v.forw(v0.out,col);
		for(size_t	h=0;	h<heads;	h++){
			Eigen::Map<Eigen::MatrixXf,Eigen::Aligned64,Eigen::OuterStride<R>	>	mwk(qk.out.data+(heads+h)*R/heads/2,R/heads/2,C);
			Eigen::Map<Eigen::MatrixXf,Eigen::Aligned64,Eigen::OuterStride<R>	>	mwq(qk.out.data+h*R/heads/2,R/heads/2,C);
			Eigen::Map<Eigen::VectorXf,Eigen::Aligned64>	ma(a.data,C);
			ma.noalias()=(1/sqrtf(R/heads/2))*(mwk.transpose()*mwq.col(col));
			float	*p=pem(h);
			for(size_t	j=0;	j<C;	j++){	size_t	i=(j+1+col)%C;	a.data[i]=j<global_para[0]?0:1/(1+expf(-a.data[i]-p[C-1-j]));	}
			Eigen::Map<Eigen::MatrixXf,Eigen::Aligned64,Eigen::OuterStride<R>	>	mwv(v.out.data+h*R/heads,R/heads,C);
			Eigen::Map<Eigen::MatrixXf,Eigen::Aligned64,Eigen::OuterStride<R>	>	mj(out.data+h*R/heads,R/heads,1);
			mj.col(0).noalias()=mwv*ma;
		}		
	}
};

template<size_t	R,	size_t	C,	size_t	heads>
Matrix<C,1>	attention<R,C,heads>::a;

template<size_t	R,	size_t	C,	size_t	heads>
struct	GFT{
	layernorm<R,1>	n0;
	attention<R,C,heads>	at;
	layernorm<R,1>	n1;
	row2row<R,R,1>	wo;
	Matrix<R,1>	&out=wo.out;	
	void	forw(Matrix<R,1>	&inp,	size_t	col){
		n0.forw(inp);	
		at.forw(n0.out,col);
		n1.forw(at.out);
		wo.forw(n1.out);
		for(size_t	i=0;	i<R;	i++)	out(0)[i]+=inp(0)[i];	
	}
};
}
#endif
