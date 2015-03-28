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


// COOrdinate tensor --  SPTENSOR

template <typename IndexType, typename ValueType>
struct sptensor : public tensor_shape<IndexType> 
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    std::vector< std::vector<index_type> > subs;	// Subscripts of non-zero entries
	std::vector< value_type > vals;					// valu of non-zeros

};

//  COOrdinate matricized tensor - SPTENMAT


template <typename IndexType, typename ValueType>
struct sptenmat 
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    double time, gflops;
    int tag;
    IndexType num_dims, num_nonzeros;
    std::vector< index_type> tsize;  				// dimension of tensors
    std::vector< int > rdims; 						// indices in tensor which mao to the row of corresponding matrix
    std::vector< int > cdims; 						// indices in tensor which mao to the row of corresponding matrix
    std::vector< std::vector<index_type> > subs;	// Subscripts of non-zero entries
	std::vector< value_type > vals;					// valu of non-zeros

};

