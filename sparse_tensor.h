#pragma once
#include<vector>
#include "mem.h"

using namespace std;
////////////////////////////////////////////////////////////////////////////////
//! Defines the following sparse tensor formats
//
// COO - Coordinate
////////////////////////////////////////////////////////////////////////////////

template<typename IndexType>
struct tensor_shape
{
    typedef IndexType index_type;
    IndexType num_dims, num_nonzeros;
	std::vector<IndexType> dims;
    double time, gflops;
    int tag;
};


// COOrdinate tensor

template <typename IndexType, typename ValueType>
struct sptensor : public tensor_shape<IndexType> 
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    std::vector< std::vector<index_type> > subs;	// Subscripts of non-zero entries
	std::vector< value_type > vals;					// valu of non-zeros

};


