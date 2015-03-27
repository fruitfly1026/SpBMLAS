#include <iostream>
#include <stdio.h>
#include "cmdline.h"
#include "gallery.h"
#include "tests.h"
#include "config.h"
#include "timer.h"

using namespace std;

int main()
{

    sptensor<int,int> coo;

    coo = read_coo_tensor<int,int>("examples/tensor.mtx");

	cout<<coo.num_dims<<endl;
	cout<<coo.num_nonzeros<<endl;
	for (int i=0;i<coo.num_dims;i++)
		cout<<coo.dims[i]<<" ";
	cout<<endl;
	for (int i=0;i<coo.num_nonzeros;i++)
	{
		cout<<coo.vals[i]<<" ";
	}
	cout<<endl;
	
		
}
