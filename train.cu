#include	<stdint.h>
#include	<sys/mman.h>
#include	<sys/stat.h>
#include	<sys/time.h>
#include	<algorithm>
#include	<unistd.h>
#include	<unistd.h>
#include	<iostream>
#include	"wylm.cu"
#include	<fstream>
#include	<cstdlib>
#include	<fstream>
#include	<fcntl.h>
#include	<vector>
#include	<omp.h>
#include	"model.config"
const	uint64_t	fullbatch=1ull<<26;
wylm<context,embedc,nheads,depth,256>	model;

int	fd;
struct	stat	sb;
uint8_t	*data;
uint64_t	data_size,	seed=time(NULL),	training=0;

int	open_mmap(const	char	*F){
	fd=open(F,	O_RDONLY);	if(fd<0)	return	0;
	fstat(fd,	&sb);
	data=(uint8_t*)mmap(NULL,	sb.st_size,	PROT_READ,	MAP_SHARED,	fd,	0);
	if(data==MAP_FAILED)	return	0;
	data_size=sb.st_size;
	cerr<<F<<'\t'<<data_size<<'\n';
	return	data_size;
}

void	close_mmap(void){
	munmap(data,sb.st_size);	close(fd);
}

void	document(void){
	cerr<<"usage:	train [options] text\n";
	cerr<<"\t-i:	input model=NULL\n";
	cerr<<"\t-o:	output model=model\n";
	cerr<<"\t-s:	trained sample=0\n";
	exit(0);
}

int	main(int	ac,	char	**av){
	cerr<<"weights\t"<<iofile.size<<'\n';	
	string	out="model",	in;
	int	opt;
	while((opt=getopt(ac,	av,	"i:o:s:"))>=0){
		switch(opt){
		case	'i':	in=optarg;	break;
		case	'o':	out=optarg;	break;
		case	's':{	training=atoi(optarg);	training<<=20;	}	break;
		default:	document();
		}
	}
	if(ac<optind+1)	return	0;
	if(!open_mmap(av[optind]))	return	0;
	if(in.size())	iofile.load(in.c_str());
	double	loss=0,	beta=iofile.size;	timeval	beg,	end;
	gettimeofday(&beg,NULL);
	cerr.setf(ios::fixed);
	for(;;){
		training+=context;
		float	k=(double)training/sqrtf(context)/beta;
		uint64_t	s1=seed;	wyrand(&s1);	__builtin_prefetch(data+wyrand(&s1)%(data_size-context));
		loss+=model.train(data+wyrand(&seed)%(data_size-context),sqrtf((2*k+1)/(k+1)/(k+1)/context));
		if(!(training&(fullbatch-1))){
			gettimeofday(&end,NULL);
			string	fn=out+".model";	iofile.save(fn.c_str());
			cerr.precision(4);	cerr<<(training>>20)<<'\t'<<loss/fullbatch<<"\t"<<(end.tv_sec-beg.tv_sec+1e-6*(end.tv_usec-beg.tv_usec))<<'\n';
			gettimeofday(&beg,NULL);	loss=0;
		}
	}
	close_mmap();
	return	0;
}

