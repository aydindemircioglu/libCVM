#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include <limits.h>
#include <assert.h>


#include "svm.h"
#include "utility.h"

#define FILENAME_LEN 1024
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void exit_with_help()
{
	printf(
	"Usage: bvm-train [options] training_set_file [model_file]\n"
	"options:\n"
	"-s svm_type : set type of SVM (default 9)\n"
	"	 0 -- C-SVC\n"
	"	 1 -- nu-SVC\n"
	"	 2 -- one-class SVM\n"
	"	 3 -- epsilon-SVR\n"
	"	 4 -- nu-SVR\n"
	"	 5 -- CVDD (Core Vector Data Description for novelty detection)\n"
	"	      [Tsang, Kwok, Cheung, JMLR 2005]\n"
	"	 6 -- CVM (sqr. hinge-loss for classification) [Tsang, Kwok, Cheung, JMLR 2005]\n"
	"	 7 -- CVM-LS (sqr. eps.-insensitive loss for sparse least-squares classification)\n"
	"	      [Tsang, Kwok, Lai, ICML 2005]\n"
	"	 8 -- CVR [Tsang, Kwok, Lai, ICML 2005], [Tsang, Kwok, Zurada, TNN 2006]\n"
	"	 9 -- BVM [Tsang, Kocsor, Kwok, ICML 2007]\n"	
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_set_file)\n"
    "	5 -- laplacian: exp(-sqrt(gamma)*|u-v|)\n"
    "	6 -- normalized poly: ((gamma*u'*v+coef0)/sqrt((gamma*u'*u+coef0)*(gamma*v'*v+coef0)))^degree\n"
	"	7 -- inverse distance: 1/(sqrt(gamma)*|u-v|+1)\n"
	"	8 -- inverse square distance: 1/(gamma*|u-v|^2+1)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default -1, which sets 1/averaged distance between patterns)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"	
	"-c cost : set the regularization parameter C(= cost) of C-SVC, eps.-SVR, nu-SVR, BVM and CVDD/CVM,\n"
	"          and the regularization parameter C/(mu*m) in CVM-LS s.t. (mu = s_ratio/(cost*m))\n"
	"          (default 100 for BVM/CVDD/CVM/CVM-LS)\n"
	"-C s_ratio : set the scale parameter C = s_ratio*max|Y_i| in CVR, \n"
	"             and the scale parameter C = s_ratio in CVM-LS (same as the scale parameter in LASSO)\n"	
	"             (default 10000 for CVR/CVM-LS)\n"
	"-u mu_ratio : set the regularization parameter mu = mu_ratio*max|Y_i| in CVR (default = 0.02)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"	
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 200)\n"
	"-e epsilon : set tolerance of termination criterion\n"
	" (In CVM/BVM, default eps=-1 which sets eps according to the bound |f(x)-f(x)^*|; default 1e-3 for others)\n"
	"-f max #CVs : MAX number of Core Vectors in binary CVM and BVM (default 50000)\n"
	"-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n: n-fold cross validation mode\n"
	"-a size: sample size for probabilistic sampling (default 60)\n"
	);
	exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node  *x_space;
int cross_validation;
int nr_fold;
double reg_param   = 100.0;
double scale_param = 10000.0;

int main(int argc, char **argv)
{
	#ifdef WIN32
		// Send all reports to STDOUT
		_CrtSetReportMode( _CRT_WARN, _CRTDBG_MODE_FILE );
		_CrtSetReportFile( _CRT_WARN, _CRTDBG_FILE_STDOUT );
		_CrtSetReportMode( _CRT_ERROR, _CRTDBG_MODE_FILE );
		_CrtSetReportFile( _CRT_ERROR, _CRTDBG_FILE_STDOUT );
		_CrtSetReportMode( _CRT_ASSERT, _CRTDBG_MODE_FILE );
		_CrtSetReportFile( _CRT_ASSERT, _CRTDBG_FILE_STDOUT );

		// enable the options
		SET_CRT_DEBUG_FIELD( _CRTDBG_DELAY_FREE_MEM_DF );
		SET_CRT_DEBUG_FIELD( _CRTDBG_LEAK_CHECK_DF );
	#endif

	printf("int %d, short int %d, char %d, double %d, float %d, node %d\n",sizeof(int),sizeof(short int), sizeof(char), sizeof(double), sizeof(float), sizeof(svm_node));

	char input_file_name[FILENAME_LEN];    
	char model_file_name[FILENAME_LEN];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);
    read_problem(input_file_name);

	printf ("Finish reading input files!\n");

	error_msg = svm_check_parameter(&prob,&param);	

	#ifdef WIN32
		assert(_CrtCheckMemory());
	#endif

	if(error_msg)
	{
		fprintf(stderr,"Error: %s\n",error_msg);
		exit(1);
	}

    double duration;
	double start = getRunTime();
	if(cross_validation)
	{
		do_cross_validation();
	}
	else
	{
		printf("kernel: %d\n",param.kernel_type);
		model = svm_train(&prob,&param);
        double finish = getRunTime();	
        duration = (double)(finish - start);

    #ifdef WIN32
		assert(_CrtCheckMemory());
	#endif

		svm_save_model(model_file_name,model);
		svm_destroy_model(model);
	}
	
	printf("CPU Time = %f second\n", duration);
    FILE* fModel = fopen(model_file_name, "a+t");					// append mode
	fprintf(fModel, "CPU Time = %f second\n", duration);
	fclose(fModel);
	    
    svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);

	#ifdef WIN32
		assert(_CrtCheckMemory());
	#endif

    return 0;
}

void do_cross_validation()
{
	int i;
	int total_correct  = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);

	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}
	free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;

	// default values
	param.svm_type     = BVM;
	param.kernel_type  = RBF;
	param.degree       = 3;
	param.gamma        = -1;	
	param.coef0        = 0;
	param.nu           = 0.5;
	param.mu           = 0.02;
	param.cache_size   = 200;	
	param.C            = INVALID_C;
	param.eps          = 1e-3;
	param.p            = 0.1;
	param.shrinking    = 1;
	param.probability  = 0;
	param.nr_weight    = 0;
	param.weight_label = NULL;
	param.weight       = NULL;
	param.sample_size  = 60;
	param.num_basis    = 50000;
	cross_validation   = 0;
	bool epsIsSet      = false;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'f':
				param.num_basis = atoi(argv[i]);
				break;
			case 's':
				param.svm_type = atoi(argv[i]);
				if (!epsIsSet && (param.svm_type == CVDD || param.svm_type == CVM || param.svm_type == CVM_LS 
					|| param.svm_type == CVR || param.svm_type == BVM ))
					param.eps = -1;
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'u':
				param.mu = atof(argv[i]);
				break;			
			case 'c':
				reg_param = atof(argv[i]);
				break;
			case 'C':
				scale_param = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				epsIsSet = true;
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					fprintf(stderr,"n-fold cross validation: n must >= 2\n");
					exit_with_help();
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			case 'a':
				param.sample_size = atoi(argv[i]);
				break;
			default:
				fprintf(stderr,"unknown option\n");
				exit_with_help();
		}
	}

	// determine filenames

	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
	{
		strcpy(model_file_name,argv[i+1]);
	}
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}	
}

double CalRBFWidth()
{
	double sumDiagonal    = 0.0;
	double sumWholeKernel = 0.0;

	int inc = 1;
	int count = 0;
	int numData = prob.l;

	if (numData > 5000)
	{
		inc = (int)ceil(numData/5000.0);
	}

	for(int i=0; i<numData; i+=inc)
	{
		count++;

		for (int j=i; j<numData; j+=inc)
		{
			double dot = Kernel::dot(prob.x[i], prob.x[j]);
			if (j == i)
			{
				sumDiagonal    += dot;
				sumWholeKernel += (dot/2.0);
			}
			else sumWholeKernel += dot;
		}
	}

	return (sumDiagonal - (sumWholeKernel*2)/count)*(2.0/(count-1));
}

void count_pattern(FILE *fp, svm_problem &prob, int &elements, int &type, int &dim)
{
    int c;
    do
    {
	    c = fgetc(fp);
	    switch(c)
	    {
		    case '\n':
			    ++prob.l;
			    // fall through,
			    // count the '-1' element
			    if ((type == 1) && (dim == 0)) // dense format
			        dim = elements;				    
			    break;

		    case ':':
			    ++elements;
			    break;

		    case ',':
			    ++elements;
			    type = 1;
			    break;

		    default:
			    ;
	    }
    } while  (c != EOF);
    rewind(fp);
}

double labelMapping(double val, double *orgLab, double *newLab, int numMap)
{
    for (int i=0; i<numMap; i++)
    {
        if (val == orgLab[i])        
            return newLab[i];        
    }
    return val;
}

void load_pattern(FILE *fp, svm_problem &prob, int type, int dim, int bidx, int eidx, int &max_index, int &j, double *orgLab=NULL, double *newLab=NULL, int numMap=0)
{
    for(int i=bidx; i<eidx; i++)
	{
		double label;
		prob.x[i] = &x_space[j];
		if (type == 0) // sparse format
		{
			fscanf(fp,"%lf",&label);
			prob.y[i] = labelMapping(label,orgLab,newLab,numMap);
		}

		int elementsInRow = 0;
		while(1)
		{
			int c;
			do {
				c = getc(fp);				
				if(c=='\n') break;
			} while(isspace(c));
			if(c=='\n') break;

			ungetc(c,fp);

			if (type == 0) // sparse format
			{
#ifdef INT_FEAT
				int tmpindex;
				int tmpvalue;
				fscanf(fp,"%d:%d",&tmpindex,&tmpvalue);
                x_space[j].index = tmpindex;
			    x_space[j].value = tmpvalue;
#else
				fscanf(fp,"%d:%lf",&(x_space[j].index),&(x_space[j].value));
#endif
				++j;
			}
			else if ((type == 1) && (elementsInRow < dim)) // dense format, read a feature
			{
				x_space[j].index = elementsInRow;
				elementsInRow++;
#ifdef INT_FEAT
				int tmpvalue;
                fscanf(fp, "%d,", &tmpvalue);				
			    x_space[j].value = tmpvalue;
#else
				fscanf(fp, "%lf,", &(x_space[j].value));
#endif
				++j;
			}
			else if ((type == 1) && (elementsInRow >= dim)) // dense format, read the label
			{
                fscanf(fp,"%lf",&label);
				prob.y[i] = labelMapping(label,orgLab,newLab,numMap);
			}
		}	

		if(j>=1 && x_space[j-1].index > max_index)
			max_index = x_space[j-1].index;
		x_space[j++].index = -1;
	}
}

/*
  corrected solution with dense format extension
*/
void read_problem(const char *filename)
{
	int elements, max_index, i, j;
	int type, dim;

	FILE *fp = fopen(filename,"r");
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

    // Support multiple data files and label renaming map
    bool mfilemode = false;
	int ch1 = getc(fp);
	if (ch1 == '#')
	{
		int ch2 = getc(fp);
		if (ch2 == 'm')
            mfilemode = true;
		do { ch2 = getc(fp); } while (ch2 != '\n');
	}
    else 
        ungetc(ch1,fp);

    if (mfilemode == true)
    {
        int numFile;     
        int numMap;
        fscanf(fp,"%d %d\n", &numFile, &numMap); 
        printf("#files : %d",numFile);
        char** filenames = new char*[numFile];
        int*   numPats   = new int[numFile];
        int fidx;
        for (fidx = 0; fidx<numFile; fidx++)
        {
            filenames[fidx] = new char[FILENAME_LEN];
            fscanf(fp,"%s\n", filenames[fidx]);
            printf(", %s", filenames[fidx]);
        }
        printf("\n");
        double *orgLab = new double[numMap];
        double *newLab = new double[numMap];
        for (int midx = 0; midx<numMap; midx++)
        {            
            fscanf(fp,"%lf>%lf\n", orgLab+midx, newLab+midx);
            printf("%g>%g\n", orgLab[midx],newLab[midx]);
        }

   	    prob.l   = 0;
	    elements = 0;
	    type     = 0; // sparse format
	    dim      = 0;
        for (fidx = 0; fidx<numFile; fidx++)
        {
            FILE *fp2 = fopen(filenames[fidx],"r+t");
            if(fp2 == NULL)
		        fprintf(stderr,"can't open input file %s\n",filenames[fidx]);	
	        else
    	        count_pattern(fp2, prob, elements, type, dim);
            numPats[fidx] = prob.l;
            fclose(fp2);
        }

        prob.y  = Malloc(double,prob.l);
	    prob.x  = Malloc(struct svm_node *,prob.l);
	    x_space = Malloc(struct svm_node,elements + prob.l);

	    if (!prob.y || !prob.x || !x_space)
	    {
		    fprintf(stdout, "ERROR: not enough memory!\n");

		    prob.l = 0;
		    return;
	    }

	    max_index = 0;
	    j         = 0;
        for (fidx = 0; fidx<numFile; fidx++)
        {
            FILE *fp2 = fopen(filenames[fidx],"r+t");
            if(fp2 == NULL)
		        fprintf(stderr,"can't open input file %s\n",filenames[fidx]);	
	        else
                load_pattern(fp2, prob, type, dim, (fidx>0)?numPats[fidx-1]:0, numPats[fidx],  max_index, j, orgLab, newLab, numMap);            
            fclose(fp2);
        }
        for (fidx = 0; fidx<numFile; fidx++)
            delete [] filenames[fidx];
        delete [] filenames;
        delete [] numPats;
        delete [] orgLab;
        delete [] newLab;
    }
    else
    {
	    prob.l   = 0;
	    elements = 0;
	    type     = 0; // sparse format
	    dim      = 0;
        count_pattern(fp, prob, elements, type, dim);
    	
	    prob.y  = Malloc(double,prob.l);
	    prob.x  = Malloc(struct svm_node *,prob.l);
	    x_space = Malloc(struct svm_node,elements+prob.l);

	    if (!prob.y || !prob.x || !x_space)
	    {
		    fprintf(stdout, "ERROR: not enough memory!\n");

		    prob.l = 0;
		    return;
	    }

	    max_index = 0;
	    j         = 0;
        load_pattern(fp, prob, type, dim, 0, prob.l, max_index, j);     
    }
    fclose(fp);

	if ( param.svm_type == CVR )
	{
		param.C = (scale_param <= 0.0) ? 10000.0 : scale_param;
		if ( param.mu < 0.0 )
			param.mu = 0.02;
		
		double maxY = -INF, minY = INF;
		for (i=0; i<prob.l; i++)
		{
			maxY = max(maxY, prob.y[i]);
			minY = min(minY, prob.y[i]);
		}
		maxY     = max(maxY, -minY);
		param.C  = param.C *maxY;
		param.mu = param.mu*maxY;

		printf("MU %.16g, ", param.mu);
	}
	else if ( param.svm_type == CVM_LS )
	{
		param.C  = (scale_param <= 0.0) ? 10000.0 : scale_param;			
		param.mu = param.C/((reg_param < 0.0) ? 100.0 : reg_param)/prob.l;

		printf("MU %.16g, ", param.mu);
	}
	else // other SVM type		
	{
		param.C = (reg_param <= 0.0) ? 100.0 : param.C = reg_param;
	}

	if(param.gamma == 0.0)
		param.gamma = 1.0/max_index;
	else if (param.gamma < -0.5)
		param.gamma = 2.0/CalRBFWidth();

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}	

	switch(param.kernel_type)
	{
		case NORMAL_POLY:
		case POLY:
			printf("Degree %.16g, coef0 %.16g, ", param.degree, param.coef0);
			break;
		case RBF:
		case EXP:
		case INV_DIST:
		case INV_SQDIST:
			printf("Gamma %.16g, ", param.gamma);
			break;
		case SIGMOID:
			printf("Gamma %.16g, coef0 %.16g, ", param.gamma, param.coef0);
			break;						
	}
	printf("C = %.16g\n", param.C);
}

// read in a problem (in svmlight format)
/*
  original solution with memory bug
*/
/*
void read_problem(const char *filename)
{
	int elements, max_index, i, j;
	FILE *fp = fopen(filename,"r");
	
	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	while(1)
	{
		int c = fgetc(fp);
		switch(c)
		{
			case '\n':
				++prob.l;
				// fall through,
				// count the '-1' element
			case ':':
				++elements;
				break;
			case EOF:
				goto out;
			default:
				;
		}
	}
out:
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		double label;
		prob.x[i] = &x_space[j];
		fscanf(fp,"%lf",&label);
		prob.y[i] = label;

		while(1)
		{
			int c;
			do {
				c = getc(fp);
				if(c=='\n') goto out2;
			} while(isspace(c));
			ungetc(c,fp);
			fscanf(fp,"%d:%lf",&(x_space[j].index),&(x_space[j].value));
			++j;
		}	
out2:
		if(j>=1 && x_space[j-1].index > max_index)
			max_index = x_space[j-1].index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0)
		param.gamma = 1.0/max_index;
	else if (param.gamma < -0.5)
		param.gamma = 2.0/CalRBFWidth();

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
} 
*/
