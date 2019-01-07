#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include "nrutil.h"

#define TINY 1.0e-20
#define NR_END 1
#define FREE_ARG char*
#define PI M_PI

#define EPSILON (1.0/24.0)
#define JSA 0.985497788

#define VRA 0.0330112265
#define BBETA (VRA/(1.0+VRA))

#define NUS 25.0
#define NUP 25.0
#define NUL 80.0

#define PE 10000.0
#define NA 0.0001

#define WI 10.62
#define DT 0.000001

#define MAXITER 10
#define M 10			// Define number of Chebyshev modes



/****************************************************************************************************/
/******* 	Scheme for solving linear system of equations (Numerical Recipes in C)	      *******/
/****************************************************************************************************/


void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
}

double *vector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;

	v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
	if (!v) nrerror("allocation failure in vector()");
	return v-nl+NR_END;
}

void free_vector(double *v, long nl, long nh)
/* free a double vector allocated with vector() */
{
	free((FREE_ARG) (v+nl-NR_END));
}

void ludcmp(double **a, int n, int *indx, double *d)
{

	int i, imax, j, k;
	double big, dum, sum, temp;
	double *vv;

	vv = vector(1,n);
	*d = 1.0;
	for (i = 0; i < n; i++){
		big = 0.0;
		for (j = 0; j < n; j++){
			if ((temp = fabs(a[i][j])) > big) big = temp;
		}
		if (big == 0.0) nrerror("Singular matrix in routine ludcmp");
		vv[i] = 1.0/big;
	}
	for (j = 0; j < n; j++){
		for (i = 0; i < j; i++){ 
			sum = a[i][j];
			for (k = 0; k < i; k++) {
				sum -= a[i][k]*a[k][j];
			}
			a[i][j] = sum;
		}
		big = 0.0;
		for (i = j; i < n; i++){
			sum = a[i][j];
			for (k = 0; k < j; k++){
				sum -= a[i][k]*a[k][j];
			}
			a[i][j] = sum;
			if ( (dum = vv[i]*fabs(sum)) >= big){
				big = dum;
				imax = i;
			}
		}
		if (j != imax) {
			for (k = 0; k < n; k++){
				dum = a[imax][k];
				a[imax][k] = a[j][k];
				a[j][k] = dum;
			}
			*d = -(*d);
			vv[imax] = vv[j];
		}
		indx[j] = imax;
		if (a[j][j] == 0.0) a[j][j] = TINY;
		if (j != n){
			dum = 1.0/(a[j][j]);
			for (i = j+1; i < n; i++) a[i][j] *= dum;
		}
	}
	free_vector(vv, 1, n);
}

void lubksb(double **a, int n, int *indx, double b[])
{
	int i, ii=0, ip, j;
	double sum;

	for (i = 0; i < n; i++){
		ip = indx[i];
		sum = b[ip];
		b[ip] = b[i];
		if (ii){
			for (j = ii; j <= i-1; j++) sum -= a[i][j]*b[j];
		}
		else if (sum) ii=i-1;
		b[i] = sum;
	}
	for (i = n-1; i >= 0; i--){
		sum = b[i];
		for (j = i+1; j < n; j++) sum -= a[i][j]*b[j];
			b[i] = sum/a[i][i];
	}
}


/***************************************************************************************************/
/*******			      Matrix & vector operations 			     *******/
/***************************************************************************************************/

void mat_mat_dot(double** result, double** _mat1, double** _mat2){

	int i, j, k;
	// Create output matrix
	double** result_TEMP = malloc((M) * sizeof(double *));
	for (i = 0; i < M; i++){
		result_TEMP[i] = malloc((M) * sizeof(double));
	}

	// Calculate dot product elements
	double sum;
	for (i = 0; i < M; i++){
		for (j = 0; j < M; j++){
			sum = 0.0;
			for (k = 0; k < M; k++){
				sum += _mat1[i][k]*_mat2[k][j];
			}
			result_TEMP[i][j] = sum;	
		}
	}

	for (i = 0; i < M; i++){
		for (j = 0; j < M; j++){
			result[i][j] = result_TEMP[i][j];
		}
	}

	for (i = 0; i < M; i++){
		free(result_TEMP[i]);
	}
	free(result_TEMP);
}


void mat_vec_dot(double* result, double** _mat1, double* _vec1){

	double* result_TEMP = malloc((M) * sizeof(double));	// Create temporary output vector

	int i, j;
	double sum;
	for (i = 0; i < M; i++){
		sum = 0.0;
		for (j = 0; j < M; j++){
			
			sum += _mat1[i][j]*_vec1[j];

		}
		result_TEMP[i] = sum;
	}
	for (i = 0; i < M; i++){
		result[i] = result_TEMP[i];
	}
	free(result_TEMP);
}


// Define function to add two square matrices
void add(double** result, double** _mat1, double** _mat2){

	// Create output matrix
	int i, j;
	for (i = 0; i < M; i++){
		for (j = 0; j < M; j++){
			result[i][j] = _mat1[i][j] + _mat2[i][j];
		}
	}
}


// Define function to multiply a matrix by a scalar
void scalar_mult_mat(double** result, double** _mat1, double _a){

	// Create output matrix
	int i, j;
	for (i = 0; i < M; i++){
		for (j = 0; j < M; j++){
			result[i][j] = _mat1[i][j] * _a;
		}
	}
}


// Define function to multiply a vector by a scalar
void scalar_mult_vec(double* result, double* _vec1, double _a){

	// Create output matrix
	int i;
	for (i = 0; i < M; i++){
		result[i] = _vec1[i] * _a;
	}
}

/***************************************************************************************************/
/*******		Function to update stress and temperature vectors 		     *******/
/*******										     *******/
/*******	_StressTemp argument is a 4*M length vector containing, respectively,        *******/
/*******	components of the stress tensor in the xx, xy, and yy direction, and         *******/
/*******	the temperature vector. Components of stress in each direction &             *******/
/*******	temperature represented by:						     *******/
/*******										     *******/
/*******		xx:   _StressTemp[0] -> _StressTemp[M-1]			     *******/
/*******		xy:   _StressTemp[M] -> _StressTemp[2M-1]			     *******/
/*******		yy:   _StressTemp[2M] -> _StressTemp[3M-1]			     *******/
/*******		temp: _StressTemp[3M] -> _StressTemp[4M-1]			     *******/
/***************************************************************************************************/

void EQS(double* _output, double* _StressTemp, double** D1, double** D2, double** II){

	int i, j;
	double d, A;

	double* _T = malloc((M) * sizeof(double));
	double* _Txy = malloc((M) * sizeof(double));
	double* _x1 = malloc((M) * sizeof(double));
	double* _rhsvec = malloc((M) * sizeof(double));
	double* _rhsvec_TEMP = malloc((M) * sizeof(double));
	double* T_dot = malloc((M) * sizeof(double));
	double* dTdt = malloc((M) * sizeof(double));	
	double** Es_p_l = malloc((3) * sizeof(double *));
	double** _Mat = malloc((M) * sizeof(double *));
	int* indx = malloc((M) * sizeof(int));

	// "Unpack" temperature vector from input vector
	for (i = 0; i < M; i++){
		_T[i] = _StressTemp[i+(3*M)];
		_Txy[i] = _StressTemp[i+M];
		_Mat[i] = malloc((M) * sizeof(double));
		_x1[i] = (1.0/_T[i]) - 1.0;				// Fill x1 vector (argument of exponentials)
	}

	for (i = 0; i < 3; i++){
		Es_p_l[i] = malloc((M) * sizeof(double));		// Declare matrix of Es, Ep and El values
	}


	mat_vec_dot( T_dot , D1 , _T );					// 1st time deriv. of temperature vector

	A = 0.0;
	for (i = 0; i < M; i++){
		A += pow(_T[i] , 2);
	}

	for (i = 0; i < M; i++){
		Es_p_l[0][i] = exp(NUS*_x1[i]);	  			// Es values
		Es_p_l[1][i] = exp(NUP*_x1[i]);	  			// Ep values
		Es_p_l[2][i] = exp(-NUL*_x1[i]);  			// El values

		T_dot[i] *= (-NUS/(pow(_T[i] , 2))); // Need to check if _T[i]*_T[i] is correct here, instead of A = (sum on i)_T[i]*_T[i]
		//T_dot[i] *= (-NUS/A);
	}

	for (i = 0; i < M; i++){					// Begin constructing LHS of Navier-Stokes equation
		for (j = 0; j < M; j++){
			_Mat[i][j] = II[i][j];
		}
		_Mat[i][i] = T_dot[i];
	}

	mat_mat_dot( _Mat , _Mat , D1 );
	add( _Mat , D2 , _Mat );
	scalar_mult_mat( _Mat , _Mat , BBETA );


	for (i = 0; i < M; i++){					// BCs for Navier-Stokes equation
		_Mat[0][i] = 0.0;
		_Mat[M-1][i] = 0.0;
	}

	mat_vec_dot( _rhsvec , D1 , _Txy );
	for (i = 0; i < M; i++){
		_rhsvec[i] *= (BBETA-1.0)/Es_p_l[0][i];
	}

	_Mat[0][0] = 1.0;						// BCs for Navier-Stokes equation
	_Mat[M-1][M-1] = 1.0;
	_rhsvec[0] = 0.0;
	_rhsvec[M-1] = 1.0;


	ludcmp(_Mat, M, indx, &d);
	lubksb(_Mat, M, indx, _rhsvec); 				// Solution given in _rhsvec

	for (i = 0; i < M; i++){
		_rhsvec_TEMP[i] = _rhsvec[i];
	}

	mat_vec_dot(_rhsvec , D1 , _rhsvec);				// 'gd' in python version
	mat_vec_dot(_rhsvec_TEMP , D2 , _T );
	scalar_mult_vec(dTdt , _rhsvec_TEMP , (1.0/PE) );

	A = 0.0;
	for (i = 0; i < M; i++){
		A += pow(_rhsvec[i] , 2);
	}

	for (i = 0; i < M; i++){
		dTdt[i] += (BBETA*NA/PE)*Es_p_l[0][i]*_rhsvec[i]*_rhsvec[i] + (1.0-BBETA)*(NA/PE)*_StressTemp[i+M]*_rhsvec[i]; // Need to check if _rhsvec[i]*_rhsvec[i] is correct here, instead of A = (sum on i)_rhsvec[i]*_rhsvec[i]
		//dTdt[i] += (BBETA*NA/PE)*Es_p_l[0][i]*A + (1.0-BBETA)*(NA/PE)*_StressTemp[i+M]*_rhsvec[i];

		_output[i] = (-1.0)*(_T[i]*Es_p_l[2][i]*_StressTemp[i]/WI) + ((1.0-JSA)*_StressTemp[i+M]*_rhsvec[i]) + (_StressTemp[i]*dTdt[i]/_T[i]);

		_output[i+(2*M)] = (-1.0)*(_T[i]*Es_p_l[2][i]*_StressTemp[i+(2*M)]/WI) - ((1.0+JSA)*_StressTemp[i+M]*_rhsvec[i]) + (_StressTemp[i+(2*M)]*dTdt[i]/_T[i]);

		_output[i+M] = (-1.0)*(_T[i]*Es_p_l[2][i]*_StressTemp[i+M]/WI); 
		_output[i+M] += (0.5*( (1.0-JSA)*_StressTemp[i+(2*M)] - (1.0+JSA)*_StressTemp[i] )*_rhsvec[i]); 
		_output[i+M] += (_StressTemp[i+M]*dTdt[i]/_T[i]) + (_T[i]*_rhsvec[i]*Es_p_l[1][i]*Es_p_l[2][i]);

		_output[i+(3*M)] = (BBETA*(NA/PE)*Es_p_l[0][i]*_rhsvec[i]*_rhsvec[i]);
		_output[i+(3*M)] += ((1.0-BBETA)*(NA/PE)*_StressTemp[i+M]*_rhsvec[i]);
	}

	// Deallocate arrays
	free(_T);
	free(_Txy);
	free(_x1);
	free(_rhsvec);
	free(_rhsvec_TEMP);
	free(T_dot);
	free(dTdt);
	free(indx);

	for (i = 0; i < M; i++){
		if (i < 3) free(Es_p_l[i]);
		free(_Mat[i]);
	}

	free(Es_p_l);
	free(_Mat);
}

/***************************************************************************************************/
/***************************************************************************************************/
/***************************************************************************************************/



int main()
{

	int i, j, k;
	char filename_trace[256];
	char filename_velocity[256];
	char filename_stress[256];
	char dirname1[256];
	char dirname2[256];
	char answer[2];
	double norm, d_lstar, d_lnplus1, d, A;


	double **II = malloc((M) * sizeof(double *));			// Identity	
	double *ygl = malloc((M) * sizeof(double));			// Vector for use in D1 calculation
	double **D1 = malloc((M) * sizeof(double *));			// 1st differential op.
	double **D2 = malloc((M) * sizeof(double *));			// 2nd differential op.
	double *cbar = malloc((M) * sizeof(double));			// Multipicative const. in D1 calculation

	double **Lstar = malloc((M) * sizeof(double *));		// Operator of T
	double **Lnplus1 = malloc((M) * sizeof(double *));		// Operator of T

	double *Temp = malloc((M) * sizeof(double));			
	double *r_vector = malloc((M) * sizeof(double));
	double **Mat = malloc((M) * sizeof(double *));
	double *rhsvec = malloc((M) * sizeof(double));
	double *T_dot = malloc((M) * sizeof(double));

	int *indx_Lstar = malloc((M) * sizeof(int));			// Index arrays for matrix inversion
	int *indx_Lnplus1 = malloc((M) * sizeof(int));
	int *indx_Mat = malloc((M) * sizeof(int));

	double *StressTemp = malloc((4*M) * sizeof(double));		// Arrays containing stress and temperature values in main loop
	double *StressStar = malloc((4*M) * sizeof(double));
	double *F_StressTemp = malloc((4*M) * sizeof(double));

	for (i = 0; i < M; i++){

		// Define & fill II
		II[i] = malloc((M) * sizeof(double));
		II[i][i] = 1.0;

		// Fill ygl
		ygl[i] = cos((PI*i)/(M-1));

		// Fill cbar
		cbar[i] = 1.0;

		// Define D1
		D1[i] = malloc((M) * sizeof(double));

		// Define D2
		D2[i] = malloc((M) * sizeof(double));

		// Define _Mat1 for calculating velocity field
		Mat[i] = malloc((M) * sizeof(double));

		// Define operators for T
		Lstar[i] = malloc((M) * sizeof(double));
		Lnplus1[i] = malloc((M) * sizeof(double));

		// Fill Temp vector for later on in main simulation
		StressTemp[i] = 0.0;
		StressTemp[M+i] = 0.0;
		StressTemp[(2*M)+i] = 0.0;
		StressTemp[(3*M)+i] = 1.0;
	}

	cbar[0] = 2.0;
	cbar[M-1] = 2.0;


	// Fill D1
	for (i = 0; i < M; i++){
		for (j = 0; j < M; j++){
			if (i != j){
				D1[i][j] = cbar[i]*(pow(-1.0 , (i+j)))/(cbar[j]*(ygl[i]-ygl[j]));
			}	
		}
	}	
	for (i = 1; i < M-1; i++){
		D1[i][i] = (-0.5*ygl[i])/(1.0-ygl[i]*ygl[i]);
	}

	D1[0][0] = (2.0*(M-1)*(M-1)+1.0)/6.0;
	D1[M-1][M-1] = -D1[0][0];

	//scalar_mult_mat(D1 , D1 , (2.0/EPSILON));
	scalar_mult_mat(D1 , D1 , 2.0);

	// Define 2nd differential op.
	mat_mat_dot(D2, D1, D1);

/*
	for (i = 0; i < M; i++){
		for (j = 0; j < M; j++){

			printf("%f ", D2[i][j]);

		}
		printf("\n");
	}
	exit(0);
*/

	/***************************************************************************/


	// Operators for T

	scalar_mult_mat(Lstar , D2 , (-0.25*DT/PE));
	scalar_mult_mat(Lnplus1 , D2 , (-0.5*DT/PE));
	add(Lstar, II, Lstar);
	add(Lnplus1, II, Lnplus1);

	for (i = 0; i < M; i++){
		Lstar[0][i] = 0.0;
		Lstar[M-1][i] = 0.0;
	
		Lnplus1[0][i] = 0.0;
		Lnplus1[M-1][i] = 0.0;
	}

	Lstar[0][0] = 1.0;
	Lstar[M-1][M-1] = 1.0;
	Lnplus1[0][0] = 1.0;
	Lnplus1[M-1][M-1] = 1.0;

	ludcmp(Lstar, M, indx_Lstar, &d_lstar);
	ludcmp(Lnplus1, M, indx_Lnplus1, &d_lnplus1);


	/************************************************************************/
	/************** 		Main code 		  ***************/
	/************************************************************************/


	printf( "\n#################################\n");
	printf( "####  Simulation  parameters ####\n" );
	printf( "#################################\n" );
	printf( "MaxIter\t\t=\t%d\n", MAXITER);
	printf( "DT\t\t=\t%f\n", DT );
	printf( "M\t\t=\t%d\n", M );
	printf( "JSa\t\t=\t%f\n", JSA );
	printf( "VRA\t\t=\t%f\n", VRA );
	printf( "NUs\t\t=\t%f\n", NUS );
	printf( "NUp\t\t=\t%f\n", NUP );
	printf( "NUl\t\t=\t%f\n", NUL );
	printf( "Peclet no.\t=\t%5.3f\n", PE );
	printf( "Na\t\t=\t%f\n", NA );
	printf( "Weisseman no.\t=\t%f\n", WI );
	printf( "#################################\n");


	// Create output files
	snprintf(dirname1, sizeof(dirname1), "./FIELDS_%d_%1.8f_%2.1f_%2.1f_%2.1f_%5.1f_%1.4f_%2.2f/", MAXITER, DT, NUS, NUP, NUL, PE, NA, WI);
	snprintf(dirname2, sizeof(dirname2), "./FIELDS_%d_%1.8f_%2.1f_%2.1f_%2.1f_%5.1f_%1.4f_%2.2f/VELOCITY/", MAXITER, DT, NUS, NUP, NUL, PE, NA, WI);
	snprintf(dirname2, sizeof(dirname2), "./FIELDS_%d_%1.8f_%2.1f_%2.1f_%2.1f_%5.1f_%1.4f_%2.2f/STRESS/", MAXITER, DT, NUS, NUP, NUL, PE, NA, WI);
	snprintf(filename_trace, sizeof(filename_trace), "./FIELDS_%d_%1.8f_%2.1f_%2.1f_%2.1f_%5.1f_%1.4f_%2.2f/trace.dat", MAXITER, DT, NUS, NUP, NUL, PE, NA, WI);

	FILE *f = NULL;
	DIR *dir = opendir(dirname1);

	if (dir){
		printf("\nFiles already exist. Overwrite existing files? (y/n)\n\n");
		fgets(answer , 2 , stdin);
		if (strcmp(answer, "n") == 0){
			printf("\nExiting...\n");
			exit(1);
		}
	}

	else if (errno == ENOENT) {
		fprintf(stderr, "\nCreating: ");
		fprintf(stderr, "%s\n\n", dirname1);
		mkdir(dirname1, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
		mkdir(dirname2, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	}

	f = fopen(filename_trace ,"w");

	if (f == NULL){
		printf("Error opening file. Exiting...\n");
		exit(1);
	}

	for (k = 0; k < MAXITER; k++){

		//fprintf(stderr, "k = %d\n", k);

		//*********************  Predictor step  **********************//

		EQS(F_StressTemp, StressTemp, D1, D2, II);

		for (i = 0; i < M; i++){
			StressStar[i] = StressTemp[i] + 0.5*DT*F_StressTemp[i];
			StressStar[M+i] = StressTemp[M+i] + 0.5*DT*F_StressTemp[M+i];
			StressStar[(2*M)+i] = StressTemp[(2*M)+i] + 0.5*DT*F_StressTemp[(2*M)+i];
			Temp[i] = StressTemp[(3*M)+i];
		}


		mat_vec_dot(r_vector , D2 , Temp );
		scalar_mult_vec(r_vector , r_vector , (0.25/PE)*DT );

		//****************************				// When to impose BCs ??
		for (i = 0; i < M; i++){
			r_vector[i] += Temp[i];
		}
		r_vector[0] = 1.0;
		r_vector[M-1] = 1.0;
		for (i = 0; i < M; i++){
			r_vector[i] += 0.5*DT*F_StressTemp[(3*M)+i];
		}
		//****************************

		//for (i = 0; i < M; i++){
		//	r_vector[i] += Temp[i] + 0.5*DT*F_StressTemp[(3*M)+i];
		//}
		//r_vector[0] = 1.0;
		//r_vector[M-1] = 1.0;					// When to impose BCs ??

		lubksb(Lstar , M , indx_Lstar , r_vector);

		for (i = 0; i < M; i++){
			StressStar[(3*M)+i] = r_vector[i];		// Put new "temp_star" values into StressStar array
		}

		//*********************  Corrector step  **********************//

		EQS(F_StressTemp, StressStar, D1, D2, II);

		for (i = 0; i < M; i++){
			StressTemp[i] += DT*F_StressTemp[i];
			StressTemp[i+M] += DT*F_StressTemp[i+M];
			StressTemp[i+(2*M)] += DT*F_StressTemp[i+(2*M)];
		}

		mat_vec_dot(r_vector , D2 , Temp );
		scalar_mult_vec(r_vector , r_vector , (0.5/PE)*DT );

		//****************************				// When to impose BCs ??
		for (i = 0; i < M; i++){
			r_vector[i] += Temp[i];
		}
		r_vector[0] = 1.0;
		r_vector[M-1] = 1.0;
		for (i = 0; i < M; i++){
			r_vector[i] += DT*F_StressTemp[(3*M)+i];
		}
		//****************************

		//for (i = 0; i < M; i++){				// When to impose BCs ??
		//	r_vector[i] += Temp[i] + DT*F_StressTemp[(3*M)+i];
		//}
		//r_vector[0] = 1.0;
		//r_vector[M-1] = 1.0;

		lubksb(Lnplus1 , M , indx_Lnplus1 , r_vector);

		for (i = 0; i < M; i++){
			StressTemp[(3*M)+i] = r_vector[i];		// Put new temperature values into StressTemp array
		}


		//*********************  Data acquisition  **********************//

		if (k%500 == 0){

			for (int i = 0; i < (4*M); ++i)
			{
				fprintf(stderr, "%f\n", StressTemp[i]);
			}
			exit(1);

			fprintf(stderr, "k = %d\n", k);

			// Calculate norm of Srt = [ StressTemp[M] : StressTemp[(2*M)-1] ]
			norm = 0.0;
			for (i = 0; i < M; i++){
				norm += StressTemp[i+M]*StressTemp[i+M];
			}
			norm = sqrt(norm);

			// Write norm value to file
			fprintf(f, "%f %40.38f\n", k*DT , norm);
			fflush(f);

			// Calculate velocity field
			mat_vec_dot(T_dot , D1 , r_vector);

			A = 0.0;
			for (i = 0; i < M; i++){
				A += pow(r_vector[i] , 2);
			}

			for (i = 0; i < M; i++){
				T_dot[i] *= (-NUS/(pow(r_vector[i] , 2)));	// Need to check if _T[i]*_T[i] is correct here, instead of A = (sum on i)_T[i]*_T[i]
				//T_dot[i] *= (-NUS/A);
			}

			Mat = II;
			for (i = 0; i < M; i++){
				Mat[i][i] = T_dot[i];
			}
			mat_mat_dot(Mat, Mat , D1 );
			add(Mat, D2 , Mat );
			scalar_mult_mat( Mat , Mat , BBETA );

			for (i = 0; i < M; i++){
				Mat[0][i] = 0.0;
				Mat[M-1][i] = 0.0;
				rhsvec[i] = StressTemp[i+M];			// Srt in Python version
			}

			mat_vec_dot(rhsvec , D1 , rhsvec );
			for (i = 0; i < M; i++){
				rhsvec[i] *= (BBETA-1.0)*exp((-1.0)*NUS*((1.0/r_vector[i]) - 1.0));
			}

			Mat[0][0] = 1.0;
			Mat[M-1][M-1] = 1.0;
			rhsvec[0] = 0.0;
			rhsvec[M-1] = 1.0;

			// Solve for the velocity field
			ludcmp(Mat, M, indx_Mat, &d);
			lubksb(Mat, M, indx_Mat, rhsvec);			// Result contained in rhsvec

			// Write velocity field data to file
			FILE *g = NULL;
			snprintf(filename_velocity, sizeof(filename_velocity), "./FIELDS_%d_%1.8f_%2.1f_%2.1f_%2.1f_%5.1f_%1.4f_%2.2f/VELOCITY/%d.dat", MAXITER, DT, NUS, NUP, NUL, PE, NA, WI, k);
			g = fopen(filename_velocity ,"w");
			if (g == NULL){
				printf("Error opening file. Exiting...\n");
				exit(1);
			}

			for (i = 0; i < M; i++){
				fprintf(g, "%2.10f %20.20f\n", ygl[i] , rhsvec[i]);
			}


			// Write xx stress field data to file
			FILE *xx = NULL;
			snprintf(filename_stress, sizeof(filename_stress), "./FIELDS_%d_%1.8f_%2.1f_%2.1f_%2.1f_%5.1f_%1.4f_%2.2f/STRESS/stress_xx_%d.dat", MAXITER, DT, NUS, NUP, NUL, PE, NA, WI, k);
			xx = fopen(filename_stress ,"w");
			if (xx == NULL){
				printf("Error opening file. Exiting...\n");
				exit(1);
			}

			for (i = 0; i < M; i++){
				fprintf(xx, "%2.10f %20.20f\n", ygl[i] , StressTemp[i]);
			}

			// Write xy stress field data to file
			FILE *xy = NULL;
			snprintf(filename_stress, sizeof(filename_stress), "./FIELDS_%d_%1.8f_%2.1f_%2.1f_%2.1f_%5.1f_%1.4f_%2.2f/STRESS/stress_xy_%d.dat", MAXITER, DT, NUS, NUP, NUL, PE, NA, WI, k);
			xy = fopen(filename_stress ,"w");
			if (xy == NULL){
				printf("Error opening file. Exiting...\n");
				exit(1);
			}

			for (i = 0; i < M; i++){
				fprintf(xy, "%2.10f %20.20f\n", ygl[i] , StressTemp[i+M]);
			}

			// Write yy stress field data to file
			FILE *yy = NULL;
			snprintf(filename_stress, sizeof(filename_stress), "./FIELDS_%d_%1.8f_%2.1f_%2.1f_%2.1f_%5.1f_%1.4f_%2.2f/STRESS/stress_yy_%d.dat", MAXITER, DT, NUS, NUP, NUL, PE, NA, WI, k);
			yy = fopen(filename_stress ,"w");
			if (yy == NULL){
				printf("Error opening file. Exiting...\n");
				exit(1);
			}

			for (i = 0; i < M; i++){
				fprintf(yy, "%2.10f %20.20f\n", ygl[i] , StressTemp[i+(2*M)]);
			}


			
			//if (k==5000){
			//	for (i = 0; i < M; i++){
			//		printf("%2.15f\n", rhsvec[i]);
			//	}
			//}
			
			fclose(g);
			fclose(xx);
			fclose(xy);
			fclose(yy);
			//exit(1);
		}
	}

	fclose(f);
	return 0;
}
