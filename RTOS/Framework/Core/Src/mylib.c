#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

printDoubleArray(double *array, int len){
	for (int i = 0; i < len; ++i) {
		printf("%e\t",array[i]);
	}
}

printUint16_tArray(uint16_t *array, int len){
	for (int i = 0; i < len; ++i) {
		printf("%u\t",array[i]);
	}
}

