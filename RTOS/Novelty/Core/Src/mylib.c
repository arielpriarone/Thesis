#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "wavelib.h"

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
}

void printarray(const uint16_t *uintarray, size_t length){
	for (int var = 0; var < length; ++var) {
		printf("%u\r\n", uintarray[var]);
	}
	printf("\r\n");
}

SnapReadyCallback(const uint16_t *uintarray, size_t length){
	printf("Snapshot acquired: \r\n");
	printarray(uintarray, length);
}


void PacketTrasform(){
	int i, J, N, len;
	int X, Y;
	wave_object obj;
	wtree_object wt;
	double *inp, *oup;

	char *name = "db3";
	obj = wave_init(name);// Initialize the wavelet
	N = 147;
	inp = (double*)malloc(sizeof(double)* N);
	for (i = 1; i < N + 1; ++i) {
		inp[i - 1] = -0.25*i*i*i + 25 * i *i + 10 * i;
	}
	J = 3;

	wt = wtree_init(obj, N, J);// Initialize the wavelet transform object
	setWTREEExtension(wt, "sym");// Options are "per" and "sym". Symmetric is the default option

	wtree(wt, inp);
	wtree_summary(wt);
	X = 3;
	Y = 5;
	len = getWTREENodelength(wt, X);
	printf(" \r\n %d", len);
	printf(" \r\n");
	oup = (double*)malloc(sizeof(double)* len);

	printf("Node [%d %d] Coefficients :  \r\n",X,Y);
	getWTREECoeffs(wt, X, Y, oup, len);
	for (i = 0; i < len; ++i) {
		printf("%g ", oup[i]);
	}
	printf(" \r\n");

	free(inp);
	free(oup);
	wave_free(obj);
	wtree_free(wt);
	return;
}
