#ifndef SGRAPH_H_kldfljkflajkfldjksljkfsdaklfdsakl
#define SGRAPH_H_kldfljkflajkfldjksljkfsdaklfdsakl

#include "svm.h"

//#define PRINT_GRAPH 1

//
// Define the structure of the index-value pair
//
struct IdxVal
{
    int idx;
    float value;
};

//
// Define the structure and methods of a sparse graph (associated array)
//
struct SGraphStruct
{
	int            numNode;
	unsigned char *numNeighbor;		
	int          **idx;
	Wfloat       **weight;	
	Wfloat        *degree;	
};

// binary search for adjacent node
int Edge2AdjIdx(SGraphStruct* graph, int sidx, int eidx);


// API for basic graph manipulations
void InitializeGraph (struct SGraphStruct& graph, int size);

void DestroyGraph (struct SGraphStruct& graph);

int ReadGraph(SGraphStruct &graph, int numNode, const char* graph_file_name, svm_parameter *param);

void WriteGraph(SGraphStruct &graph, const char* graph_file_name);


// API to compute edge weights 
void ComputeGraphWeight(SGraphStruct& graph, svm_parameter *param);

void ComputeNGraphWeight(SGraphStruct& graph, svm_parameter *param);

#endif //SGRAPH_H_kldfljkflajkfldjksljkfsdaklfdsakl
