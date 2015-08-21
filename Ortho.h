
#include "armadillo"
#include "stdlib.h"

using namespace std;
using namespace arma;

//functions Ortho uses
void Ortho(mat W_d, mat C_d, int nComp, int maxIter, double perturbation_factor, double maxW, double maxB, string output_w, string output_c);
mat Perturb(mat T, double perturbation_factor, int nComp);
int CalculateIllegals(mat C);
mat FixIllegals(mat C);
double ScoreT(mat c, int nComp);
double CompareMat(mat a, mat b);
