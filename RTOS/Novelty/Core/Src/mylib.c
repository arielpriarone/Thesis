#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

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
		printf("%d\t", &uintarray[var]);
	}
	printf("\r\n");
}


