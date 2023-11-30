/*
 * mylib.h
 *
 *  Created on: Nov 27, 2023
 *      Author: Ariel Priarone
 */

#ifndef INC_MYLIB_H_
#define INC_MYLIB_H_





#endif /* INC_MYLIB_H_ */

int add_int(int *start, int *len);
void uint16ArrayToString(const uint16_t *uintArray, size_t length, char *result);
void printarray(const uint16_t *uintarray, size_t length);
void SnapReadyCallback(const uint16_t *uintarray, size_t length, int tree_depth);
double norm2(double *array, int length);
double *packetCoeff(double *inp, int length, int J, double *coefs);
