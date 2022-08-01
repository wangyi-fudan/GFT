#include	<unistd.h>
#include	"wylm.hpp"
#include	"model.config"
#include	<cstdio>
#include	<vector>
using	namespace	std;

void	document(void){
	fprintf(stderr,"gft\n王一@20220424\n");
	fprintf(stderr,"usage:	gft [options] \"quoted text\"\n");
	cerr<<"\t-d:	sampling diversity=0.7\n";
	cerr<<"\t-n:	maximum output bytes=1024\n";
	cerr<<"\t-m:	model file path=Chinese.model\n";
	exit(0);
}

int	main(int	ac,	char	**av){
	wylm<context,embedc,nheads,depth,256>	model;
	size_t	N=1024;	string	model_file="Chinese.model";
	int	opt;
	while((opt=getopt(ac,	av,	"d:n:m:"))>=0){
		switch(opt){
		case	'd':	model.diversity=atof(optarg);	break;
		case	'n':	N=atoi(optarg);	break;
		case	'm':	model_file=optarg;	break;
		default:	document();
		}
	}
	if(ac<optind+1){	document();	return	0;	}
	if(!iofile.load(model_file.c_str())){	fprintf(stderr,"fail to load %s\n",model_file.c_str());	return	0;	}
	vector<uint8_t>	s;	uint8_t	c='\n';
	for(char	*p=av[optind];	*p&&s.size()<N;	p++){
		s.push_back(*p);	c=model.sample(s.data()+s.size()-1,s.data());
		fputc(*p, stdout);	fflush(stdout);
	}
	while(s.size()<N){
		fputc(c, stdout);	fflush(stdout);
		s.push_back(c);	c=model.sample(s.data()+s.size()-1,s.data());
	}
	fputc('\n', stdout);	fflush(stdout);
	return	0;
}
