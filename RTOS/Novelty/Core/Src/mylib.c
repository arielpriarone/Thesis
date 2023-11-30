#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "wavelib.h"

double norm2(double *array, int length);
double *packetCoeff(double *inp, int length, int J, double *coefs);
double signal_power(double *array, int length);

int add_int(int *start, int *len){
	int result = 0;
	for (int var = 0; var < *len; ++var) {
		result += start[var];
	}
	return result;
}

void uint16ArrayToString(const uint16_t *uintArray, size_t length, char *result) {
	size_t i;
	for (i = 0; i < length - 1; ++i) {
		// sprintf(result, "%u\t", uintArray[i]);
		result += strlen(result);
	}
	sprintf(result, "%u", uintArray[i]);  // Last element without a tab
	return;
}

void printarray(const uint16_t *uintarray, size_t length){
	for (int var = 0; var < length; ++var) {
		printf("%u\r\n", uintarray[var]);
	}
	printf("\r\n");
	return;
}

void printFloatArray(double *inp, size_t length){
	for (int var = 0; var < length; ++var) {
		printf("%e \r\n", inp[var]);
	}
	printf(" \r\n");
	return;
}

void SnapReadyCallback(const uint16_t *uintarray, size_t length, int three_depth){
	// first convert the array to packet coef
	int out_len = round(pow(2,three_depth)); //the number of bottom level nodes
	double snap_time_domain[length]; // alloc the double array
	for (int i = 0; i < length; ++i) { // Cast each element from int to double
		snap_time_domain[i] = (double)uintarray[i];
	}
	double out_alloc[out_len];
	double *coefs = packetCoeff(snap_time_domain, (int)length, three_depth, out_alloc);

	// time domain features
	return;
}


double *packetCoeff(double *inp, int length, int J, double *coefs) { // compute the power of each packet coefficient in the lowest level
	int N, len;
	int coef_len;
	coef_len = round(pow(2,J)); // the nodes in the lowest level are 2^depth

	wave_object obj;
	wtree_object wt;

	char *name = "db10";
	obj = wave_init(name); // Initialize the wavelet
	N = length;

	wt = wtree_init(obj, N, J); // Initialize the wavelet transform object
	setWTREEExtension(wt, "sym"); // Options are "per" and "sym". Symmetric is the default option

	wtree(wt, inp);
	wtree_summary(wt);

	len = getWTREENodelength(wt, J); //because the lowest level is J
	printf(" \r\n %d", len);
	printf(" \r\n");

	double *oup = (double *)malloc(sizeof(double) * len);

	for(int node_index = 0; node_index < coef_len; node_index++){
		printf("Node [%d %d] Coefficients :  \r\n", J, node_index);
		getWTREECoeffs(wt, J, node_index, oup, len);
		coefs[node_index] = norm2(oup,len);
	}

	free(inp);
	wave_free(obj);
	wtree_free(wt);

	return coefs;
}

double norm2(double *array, int length) {
	double sum = 0.0;

	for (int i = 0; i < length; ++i) {
		sum += array[i] * array[i];
	}

	return sqrt(sum);
}

double signal_power(double *array, int length) {
	double sum = 0.0;

	for (int i = 0; i < length; ++i) {
		sum += array[i] * array[i];
	}

	return sqrt(sum);
}
