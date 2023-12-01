/*
 * mylib.h
 *
 *  Created on: Nov 27, 2023
 *      Author: Ariel Priarone
 */

#ifndef INC_MYLIB_H_
#define INC_MYLIB_H_





#endif /* INC_MYLIB_H_ */

void prinArray(double *array, int len);
void printUint16_tArray(uint16_t *array, int len);

double *featureExtractor(	uint16_t *time_array,			// time-domain snapshot
							int len_time_array,				// length of time-domain snapshot
							int tree_depth,					// depth of the wavelet decomposition tree
							double *out_features_array);		// output array of features

double *packetCoeff(double *inp, int length, int tree_depth, double *coefs); // compute the power of each packet coefficient in the lowest level

double norm2(double *array, int length);