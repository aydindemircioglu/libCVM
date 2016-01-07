#ifndef _LIBSVM_H
#define _LIBSVM_H

#ifdef __cplusplus
extern "C" {
#endif

//#define SHORT_INDEX 1
//#define INT_FEAT 1

#define INVALID_C -1		  // initial C

#ifdef SHORT_INDEX
    typedef short int INDEX_T;
#else
    typedef int INDEX_T;
#endif

#ifdef INT_FEAT
    typedef short int NODE_T;
#else
    typedef double NODE_T;
#endif


struct svm_node
{
	INDEX_T index;
	NODE_T  value;
};

struct svm_problem
{
	int l;
	int u;
	double *y;
	struct svm_node **x;

	struct SGraphStruct *graph;
};

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR, CVDD, CVM, CVM_LS, CVR, BVM };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED, EXP, NORMAL_POLY, INV_DIST, INV_SQDIST };   /* kernel_type */
enum { ONE_VS_ONE, ONE_VS_REST, C_AND_S };  /* mc_type */

struct svm_parameter
{
	int svm_type;
	int kernel_type;
	int degree;         /* for poly */
	double gamma;       /* for poly/rbf/sigmoid */
	double coef0;       /* for poly/sigmoid */

	/* these are for training only */
	double cache_size;  /* in MB */
	double eps;	        /* stopping criteria */
	double C;           /* for C_SVC, EPSILON_SVR, BVM and NU_SVR */
	int nr_weight;	    /* for C_SVC */
	int *weight_label;  /* for C_SVC */
	double* weight;     /* for C_SVC */
	double nu;          /* for NU_SVC, ONE_CLASS, and NU_SVR */
	double mu;          /* for CVR */
	double p;           /* for EPSILON_SVR */
    int mc_type;        /* for multiclass SVM/CVM/BVM */
	int shrinking;      /* use the shrinking heuristics */
	int probability;    /* do probability estimates */
	int sample_size;    /* size of probabilistic sampling in BVM */
	int num_basis;
	int knn;
	int weight_type;
};

//
// svm_model
//
struct svm_model
{
	svm_parameter param;	// parameter
	int nr_class;		// number of classes, = 2 in regression/one class svm
	int l;			// total #SV
	int u;
	svm_node **SV;		// SVs (SV[l])
	double **sv_coef;	// coefficients for SVs in decision functions (sv_coef[n-1][l])
	double *rho;		// constants in decision functions (rho[n*(n-1)/2])
	double *cNorm;		// center Norm of decison functions (rho[n*(n-1)/2])
	double *probA;          // pariwise probability information
	double *probB;

	// for classification only

	int *label;		// label of each class (label[n])
	int *nSV;		// number of SVs for each class (nSV[n])
				// nSV[0] + nSV[1] + ... + nSV[n-1] = l
	// XXX
	int free_sv;		// 1 if svm_model is created by svm_load_model
				// 0 if svm_model is created by svm_train
};


struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
double svm_get_svr_probability(const struct svm_model *model);

void svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict_rank(const struct svm_model *model, const struct svm_node *x);
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

void svm_destroy_model(struct svm_model *model);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

void displayInfoAboutModel(const struct svm_model* model);

#ifdef __cplusplus
}
#endif


#ifndef min
template <class T> inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> inline T max(T x,T y) { return (x>y)?x:y; }
#endif
namespace svmhelper
{
template <class T> inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
}
template <class S, class T> inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}

//#define INF HUGE_VAL
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void info(char *fmt,...);
void info_flush();

typedef float Qfloat;														// use single precision
typedef double Yfloat;														// use full precision
typedef float  Xfloat;														// use single precision
typedef float  Wfloat;														// use single precision
typedef double Afloat;														// use full precision
typedef signed char schar;													// for convenient


class QMatrix {
public:	
	virtual Qfloat *get_Q(int column, int len, int* indice = NULL) const = 0;
	virtual Qfloat *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~QMatrix() {}
};

class Kernel: public QMatrix {
public:
	Kernel(int l, svm_node * const * x, const svm_parameter& param);
	virtual ~Kernel();

	static double k_function(const svm_node *x, const svm_node *y,
				 const svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len, int* indice) const = 0;
	virtual Qfloat *get_QD() const = 0;
	virtual void swap_index(int i, int j) const;

	static double dot(const svm_node *px, const svm_node *py);
    static double distanceSq(const svm_node *x, const svm_node *y);

	bool IsSelfConst() const 
	{
		if (kernel_type == RBF || kernel_type == NORMAL_POLY || kernel_type == EXP || kernel_type == INV_DIST || kernel_type == INV_SQDIST)
			return true;
		else 
			return false;
	}

	static bool IsSelfConst(const svm_parameter& param)  
	{
		if (param.kernel_type == RBF || param.kernel_type == NORMAL_POLY || param.kernel_type == EXP || param.kernel_type == INV_DIST || param.kernel_type == INV_SQDIST)
			return true;
		else 
			return false;
	}

protected:

	double (Kernel::*kernel_function)(int i, int j) const;

private:
	const svm_node **x;
	double *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	double kernel_linear(int i, int j) const;
	double kernel_poly(int i, int j) const;
	double kernel_rbf(int i, int j) const;
	double kernel_sigmoid(int i, int j) const;
	double kernel_precomputed(int i, int j) const;
    double kernel_exp(int i, int j) const;
    double kernel_normalized_poly(int i, int j) const;
	double kernel_inv_sqdist(int i, int j) const;
	double kernel_inv_dist(int i, int j) const;	    
};



// Generalized SMO+SVMlight algorithm
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + b^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, b, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping criterion
//
// solution will be put in \alpha, objective value will be put in obj
//
class Solver {
public:
	Solver() {};
	virtual ~Solver() {};

	struct SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for Solver_NU
		double margin;
	};

	void Solve(int l, const QMatrix& Q, const double *b_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking);
protected:
	int active_size;
	schar *y;
	double *G;		// gradient of objective function
	enum { LOWER_BOUND, UPPER_BOUND, FREE };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double *alpha;
	const QMatrix *Q;
	const Qfloat *QD;
	double eps;
	double Cp,Cn;
	double *b;
	int *active_set;
	double *G_bar;		// gradient, if we treat free variables as 0
	int l;
	bool unshrinked;	// XXX

	virtual double get_C(int i)
	{
		return (y[i] > 0)? Cp : Cn;
	}
	void update_alpha_status(int i)
	{
		if(alpha[i] >= get_C(i))
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
	bool is_free(int i) { return alpha_status[i] == FREE; }
	void swap_index(int i, int j);
	void reconstruct_gradient();
	virtual int select_working_set(int &i, int &j);
	virtual int max_violating_pair(int &i, int &j);
	virtual double calculate_rho();
	virtual void do_shrinking();
};



#endif /* _LIBSVM_H */
