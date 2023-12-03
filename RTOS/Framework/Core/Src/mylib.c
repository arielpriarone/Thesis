#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "wavelib.h"
#include "defines.h"

void printDoubleArray(double *array, int len){
	for (int i = 0; i < len; ++i) {
		printf("%e\t",array[i]);
	}
	return;
}

void printUint16_tArray(uint16_t *array, int len){
	for (int i = 0; i < len; ++i) {
		printf("%u\t",array[i]);
	}
	return;
}

void myprintf(const char* format, ...) {
    // Check the global flag
    if (VERBOSE) {
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        va_end(args);
    }
    return;
}

/**
 * Calculates the squared norm of an array.
 *
 * This function calculates the squared norm of the given array, which is the sum of the squares of all its elements.
 * The length parameter specifies the number of elements in the array.
 *
 * @param array The array for which to calculate the squared norm.
 * @param length The number of elements in the array.
 * @return The squared norm of the array.
 */
double norm2(double *array, int length) {
	double sum = 0.0;
	for (int i = 0; i < length; ++i) {
		sum += array[i] * array[i];
	}
	return sqrt(sum);
}


/**
 * Compute the power of each packet coefficient in the lowest level.
 *
 * @param inp The input array of coefficients.
 * @param length The length of the input array.
 * @param tree_depth The depth of the packet tree.
 * @param coefs The output array to store the computed powers.
 * @return A pointer to the array of computed powers.
 */
double *packetCoeff(double *inp, int length, int tree_depth, double *coefs) { // compute the power of each packet coefficient in the lowest level
	int N, len;
	int coef_len;
	coef_len = round(pow(2,tree_depth)); // the nodes in the lowest level are 2^depth

	wave_object obj;
	wtree_object wt;

	char *name = "db10";
	obj = wave_init(name); // Initialize the wavelet
	N = length;

	wt = wtree_init(obj, N, tree_depth); // Initialize the wavelet transform object
	setWTREEExtension(wt, "sym"); // Options are "per" and "sym". Symmetric is the default option

	wtree(wt, inp);
	// wtree_summary(wt); too much information - reenable if needed

	len = getWTREENodelength(wt, tree_depth); //because the lowest level is J
	myprintf(" \r\n %d", len);
	myprintf(" \r\n");

	double *oup = (double *)malloc(sizeof(double) * len);

	for(int node_index = 0; node_index < coef_len; node_index++){
		myprintf("Node [%d %d] Coefficients :  \r\n", tree_depth, node_index);
		getWTREECoeffs(wt, tree_depth, node_index, oup, len);
		coefs[node_index] = norm2(oup,len);
	}
	free(oup);
	free(inp);
	wave_free(obj);
	wtree_free(wt);

	return coefs;
}


double *featureExtractor(	uint16_t *time_array,			// time-domain snapshot
							int len_time_array,				// length of time-domain snapshot
							int tree_depth,					// depth of the wavelet decomposition tree
							double *out_features_array)		// output array of features
{
	// cast the input array to double
	double time_array_double[len_time_array];
    for (int i = 0; i < len_time_array; ++i) {
    	time_array_double[i] = (double)time_array[i];
    }

	myprintf("Time array converted to double: \r\n");
    for(int i = 0; i<len_time_array; i++){
        myprintf("%e\t",time_array_double[i]);
    }
	// compute the mean of the time-domain snapshot
	double mean = 0;
	for (int i = 0; i < len_time_array; ++i) {
		mean += time_array_double[i];
	}
	mean /= (double) len_time_array;
	// compute the energy of the time-domain snapshot
	double energy = 0;
	for (int i = 0; i < len_time_array; ++i) {
		energy += pow(time_array_double[i],2);
	}
	// compute the variance of the time-domain snapshot
	double variance = 0;
	double mean_energy = energy/len_time_array;
	variance = mean_energy - pow(mean,2);

	// assign the features to the output array
	out_features_array[0] = mean;
	out_features_array[1] = energy;
	out_features_array[2] = variance;

	// compute the wavelet decomposition of the time-domain snapshot
	int len = round(pow(2,tree_depth)); // the nodes in the lowest level are 2^depth
	double *coefs = (double *)malloc(sizeof(double) * len);
	coefs = packetCoeff(time_array_double, len_time_array, tree_depth, coefs);

	// assign the features to the output array
	for (int i = 0; i < len; ++i) {
		out_features_array[i+3] = coefs[i];
	}

	// free the memory
	free(coefs);

	return out_features_array;
}


