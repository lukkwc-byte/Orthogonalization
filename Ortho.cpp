#include "Ortho.h"
#include <math.h>
#include <string>

main(void){
	mat W_d;
	mat C_d;
	mat C_o;
	mat W_o;
	mat W;
	mat C;
	string directory( "./Test_Files/");
	string weights[]={"Test_W1", "Test_W2","Test_W3","Test_W4","Test_W5","Test_W6","Test_W7","Test_W8","Test_W9","Test_W10"};
	string coeff[]={"Test_C1", "Test_C2","Test_C3","Test_C4","Test_C5","Test_C6","Test_C7","Test_C8","Test_C9","Test_C10"};
	string c[]={"c1","c2","c3","c4","c5","c6","c7","c8","c9","c10"};
	string w[]={"w1","w2","w3","w4","w5","w6","w7","w8","w9","w10"};
	double factors[]={0.001,0.01,0.1,1,10,100};	
	double maxW[]={0.01, 0.025,0.05,0.10,0.15,0.2};
	double maxB[]={0.01, 0.025,0.05,0.10,0.15,0.2};
	for (int i=0; i < 10; i++){
	
		W.load(directory+"W_real");
		C.load(directory+"C_real");
		cout<< "-------------------Trial:" << i << "----------------------\n";
		cout << "Loading Matrices\n";
		W_d.load(directory+weights[i]);
		C_d.load(directory+coeff[i]);
		arma_rng::set_seed(i);
		cout<<"Running Ortho\n\n";
		Ortho(W_d, C_d, 3, 3000, 2, 0.1, 0.01, directory+w[i], directory+c[i]);
		W_o.load("W_o");
		C_o.load("C_o");	

		cout << "\n\nAnalyzing Results\n";
		if(CompareMat(W_o,W)==CompareMat(W_d,W)){
			cout<< "Matrix unchanged \n";
		}
		else{
			cout<< "Comparing Ortho with Original: \n";
			cout<< "Weights: "<< CompareMat(W_o, W)<<"\n";
        		cout<< "Coeff: " << CompareMat(C_o, C) <<"\n";

        		cout<< "Comparing Deconv with Original:\n";
        		cout<< "Weights: "<< CompareMat(W_d, W) <<"\n";
        		cout<< "Coeff: " << CompareMat(C_d, C) <<"\n";
		}
		cout << "\n\n\n";
	}
	return 0;
}

void Ortho(mat W_d, mat C_d, int nComp, int maxIter, double perturbation_factor, double maxW, double maxB, string output_w, string output_c){
	
	//computational matrices and values	
	mat P;
	mat T=randu<mat>(nComp, nComp);
	mat T_n;
	mat C_n;
	mat C_T;
	mat C_T_current;
	mat W_T;
	int badBeta;
	int badW;
	int maxBadBeta;
	int maxBadW;
	
	maxBadBeta=C_d.n_cols*C_d.n_rows*maxW;
	maxBadW=W_d.n_cols*W_d.n_rows*maxB;

	P = W_d*C_d.t();		//Calculates product mat
	T.eye();	// initialize transformation to identify mat

	for (int i = 0; i < maxIter; i++){
		
		C_T_current = C_d * T;	//Calculate the transformed C mat using mat T
		T_n = Perturb(T, perturbation_factor, nComp);		//Randomly creates a transformation mat and calculates the new transformed C_T
		C_T = C_d*T_n;
		
		badBeta = CalculateIllegals(C_T);	//counts number of bad betas in C_T and if it exceeds the number of bad betas, replaces the bad ones
		cout<<"Bad Betas: "<< badBeta <<"\n";
		if (badBeta < maxBadBeta){			
			C_T = FixIllegals(C_T);
		}
		else { continue; }

		W_T = P*(pinv(C_T).t());	// calculate W_T	
		badW = CalculateIllegals(W_T);	//counts number of bad weights in W_T and if it exceeds the number of bad weights, replaces the bad ones
		cout<<"Bad Weights: "<<badW<<"\n";
		if (badW < maxBadW){
			W_T = FixIllegals(W_T);
		}
		else { continue; }
		float score1 = ScoreT(C_T_current, nComp);
		float score2 =ScoreT(C_T, nComp);
		cout << "Current: " << score1 << "	Transformed: " << score2 << endl;
		if (score2 < score1) {	//calculates the score of current T and the new one and chooses the best one
			cout<<"Replacing"<<endl;
			T = T_n;
		}
		cout << endl;
	}
	C_T_current = C_d * T;
	W_T=P*(pinv(C_T_current).t());
	C_T_current.save(output_c, raw_ascii);
	W_T.save(output_w, raw_ascii);
}

mat Perturb(mat T, double perturbation_factor, int nComp){

	/*
	Creates a randomly generated transformation matrix based on the one fed to it and the perturbation factor
	T:input transformation matrix
	perturbation_factor: how much you want the values to vary by
	nComp: number of cell components
	T_n: output transformation matrix
	*/

	mat temp;
	mat T_n;
	
	mat identity=randu<mat>(nComp, nComp);
	identity.eye();

	temp = identity + perturbation_factor*(randu(nComp, nComp) - 0.5);
	T_n=temp;
	for (int l = 0; l < nComp; l++){
		T_n(l, l) = 1 - (sum(temp.col(l)) - temp(l, l));
	}
	return T_n;
}

int CalculateIllegals(mat C){

	/*
	Counts the number of illegal entries in C. Illegal entries defined as >1 or <0.
	C:matrix
	Illegals: returns of illegal values
	*/

	int rows = C.n_rows-1;
	int cols = C.n_cols-1;
	int Illegals = 0;

	for (int j = 0; j < cols; j++){
		for (int i = 0; i < rows; i++){
			if (C(i, j) < 0 || C(i, j) > 1){
				Illegals++;
			}
		}
	}
	return Illegals;
}

mat FixIllegals(mat C){

	/*
	Fixes the illegal entries in C. Illegal entries defined as >1 or <0. If >1, then entry is set to 1. If <0 entry is set to 0.
	C:input matrix
	*/

	int rows = C.n_rows-1;
	int col = C.n_cols-1;
	int counter = 0;

	for (int j = 0; j < col; j++){
		for (int i = 0; i < rows; i++){
			if (C(i, j) < 0 || C(i, j) > 1){
				if (C(i, j) < 0){
					C(i, j) = 0;
				}
				else{
					C(i, j) = 1;
				}
			}
		}
	}
	return C;
}

double ScoreT(mat c, int nComp){

	/*
	Calculates using some quantitative metric
	C: transformed C matrix
	score: output. quantitative metric used to evaluate the quality of transformation matrix
	*/

	double score =0;
	
	//matrices used in scoring
	mat mid;
	mat zeropt1;
	mat zeropt9;

	mat column=randu<mat>(c.n_rows,1);
	column.ones();
	zeropt1 = 0.1*column;
	zeropt9 = 0.9*column;

	for (int j = 0; j < nComp; j++){
		score = score + norm(arma::min(abs(c.col(j) - zeropt1), abs(c.col(j) - zeropt9)));
	}
	return score;
}

double CompareMat(mat a, mat b){
	int rows=a.n_rows-1;
	int cols=a.n_cols-1;
	double minret=9999999999999;
	double sum=0;
	for (int i=0; i<cols; i++){
		for (int j=0; j<cols; j++){
			sum=0;
			for(int k=0; k<rows; k++){
				sum=sum+pow((a(k,i)-b(k,j)),2);
			}		
			if (sum<minret){
				minret=sum;
			}
		}		
	}
	return minret;
}
