#include	"neubee.hpp"
#include	<algorithm>
using	namespace	neubee;

struct	Sort{
	float	p;
	unsigned	i;
	bool	operator()(Sort	X,	Sort	Y){	return	X.p>Y.p;	}
};

template<size_t	input,	size_t	embedc,	size_t	heads,	size_t	depth,	size_t	output>
struct	wylm{
	float	diversity=1/M_SQRT2;
	size_t	curr=0;
	Matrix<output,1>	inp;
	row2row<output,embedc,1>	m0;
	GFT<embedc,input,heads>	tra[depth];
	row2row<embedc,output,1>	out;
	layernormc<output,1>	norm;	
	softmax<output,1>	lf;
	Sort	vs[output];
	size_t	sample(uint8_t	*x,	uint8_t	*p){
		global_para[0]=p+input-1>=x?p+input-1-x:0;
		memset(inp.data,0,inp.size()*sizeof(float));
		inp(0)[*x]=1;
		m0.forw(inp);
		for(size_t	d=0;	d<depth;	d++)	tra[d].forw(d?tra[d-1].out:m0.out,curr);
		out.forw(tra[depth-1].out);
		norm.forw(out.out);
		lf.forw(norm.out);
		double	ent0=0;
		for(size_t	i=0;	i<output;	i++){	vs[i].p=lf.out(0)[i];	vs[i].i=i;	if(vs[i].p>0)	ent0-=vs[i].p*logf(vs[i].p);	}
		sort(vs,vs+output,Sort());
		size_t	n=0;	double	sum=0,	plnp=0;
		do{	sum+=vs[n].p;	plnp+=vs[n].p>0?vs[n].p*logf(vs[n].p):0;	n++;	}while(n<output&&logf(sum)-plnp/sum<diversity*ent0);
		uint8_t	ret;
		do{
			double	ran=wy2u01(wyrand(&global_seed))*sum,	sum1=0;	
			for(size_t	i=0;	i<output;	i++){	sum1+=vs[i].p;	if(sum1>=ran){	ret=vs[i].i;	break;	}	}
		}while(ret=='\n');
		curr=(curr+1)%input;
		return	ret;
	}
	double	eval(uint8_t	*x,	uint8_t	*p,	uint8_t	t){
		global_para[0]=p+input-1>=x?p+input-1-x:0;
		memset(inp.data,0,inp.size()*sizeof(float));
		inp(0)[*x]=1;
		m0.forw(inp);
		for(size_t	d=0;	d<depth;	d++)	tra[d].forw(d?tra[d-1].out:m0.out,curr);
		out.forw(tra[depth-1].out);
		norm.forw(out.out);
		lf.forw(norm.out);
		curr=(curr+1)%input;
		return	lf.out(0)[t]>0?-log2f(lf.out(0)[t]):FLT_MAX;
	}
};

