#ifndef REFERENCE_CALC_H__
#define REFERENCE_CALC_H__

// for uchar4 struct
#include <cuda_runtime.h>

void referenceCalculation(const uchar4* const rgbaImage,
                          unsigned char *const greyImage,
                          size_t numRows,
                          size_t numCols);

#endif