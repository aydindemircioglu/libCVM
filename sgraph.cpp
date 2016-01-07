#include <ctype.h>
#include <algorithm>
#include <stdio.h>
#include <string.h> // memcpy

#include "utility.h"
#include "sgraph.h"


//
// Implement methods for the index-value pair
//
bool CompareLessVal(IdxVal a, IdxVal b)
{
    return (a.value < b.value);
}

bool CompareLessIdx(IdxVal a, IdxVal b)
{
    return (a.idx < b.idx);
}

//
// Implement methods of a sparse graph (associated array)
//

int Edge2AdjIdx(SGraphStruct* graph, int sidx, int eidx)
{	
	int lsIdx = 0;
	int rsIdx = graph->numNeighbor[sidx] - 1;	
	while (lsIdx <= rsIdx)
	{	
		int msIdx  = (lsIdx+rsIdx)/2;
		int tmpIdx = graph->idx[sidx][msIdx];
		if (tmpIdx == eidx)
			return msIdx;
		else if (tmpIdx > eidx)
			rsIdx = msIdx - 1;
		else
			lsIdx = msIdx + 1;			
	} 
	return -1;
}

int ReadGraph(SGraphStruct &graph, int numNode, const char* graph_file_name, svm_parameter *param)
{	
	graph.degree			   = NULL;
    graph.numNode			   = 0;
	int			 **idx		   = Malloc(int*,          numNode);
	Wfloat		 **weight	   = Malloc(Wfloat*,       numNode);
	unsigned char *numNeighbor = Malloc(unsigned char, numNode); 
	
	FILE *fp = fopen(graph_file_name,"r");
	if(fp == NULL)
	{
		fprintf(stderr,"Cannot open input file %s\n",graph_file_name);
		exit(1);
	}
		
	int elements = 0;
	while(1)
	{
		int c = fgetc(fp);
		if (c == '\n')
		{				
			numNeighbor[graph.numNode] = elements;
			elements				   = (elements < param->knn) ? elements : param->knn;
			weight[graph.numNode]      = new Wfloat[elements];
			idx[graph.numNode]         = new int[elements];
			elements                   = 0;
			graph.numNode ++;			
		}
		else if (c == ':')
			++elements;			
		else if (c == EOF)
			break;		
	}
	rewind(fp);

	graph.numNeighbor = new unsigned char[graph.numNode];
	graph.weight      = new Wfloat*		 [graph.numNode];
	graph.idx         = new int*		 [graph.numNode];
	double numEdge    = 0.0;
	for(int i=0; i<graph.numNode; i++)
	{
		int index;				
		fscanf(fp,"%d",&index);
		int numNB                = numNeighbor[i];
		graph.numNeighbor[index] = (numNB < param->knn) ? numNB : param->knn;
		graph.weight[index]      = weight[i];
		graph.idx[index]	     = idx[i];
		numEdge				    += graph.numNeighbor[index];

#ifdef PRINT_GRAPH
		printf("%d",index);
#endif

		for (int j=0; j<numNB; j++)
		{			
			int c;
			do
			{
				c = getc(fp);		
			} 
			while(isspace(c));
			ungetc(c, fp);											// put back the last "non-space"

			Wfloat buffer;
			int attrIdx;
			fscanf(fp, "%d:%f", &(attrIdx), &(buffer));			// sparse representation - (index and value)
			if (j < graph.numNeighbor[index])
			{
				graph.idx[index][j]    = attrIdx;
				graph.weight[index][j] = buffer;
			}

#ifdef PRINT_GRAPH
			printf(" %d:%f",attrIdx, buffer);
#endif
		}	

#ifdef PRINT_GRAPH
		printf("\n");
#endif
	}
	fclose(fp);

	numEdge = 0;
	for(int i=0;i<graph.numNode;i++)
	{		
		for (int j=0; j<graph.numNeighbor[i]; j++)
		{	
			int p_j      = graph.idx[i][j];
			int p_adjIdx = Edge2AdjIdx(&graph,p_j,i);
			if (p_adjIdx < 0)										
				numEdge ++;
			else numEdge += 0.5;								// avoid double counting of undirected edge
		}	
	}

	free(idx);
	free(weight);
	free(numNeighbor);
	return (int)numEdge;
}

void WriteGraph(SGraphStruct &graph, const char* graph_file_name)
{		
	FILE *fp = fopen(graph_file_name,"w+");
	if(fp == NULL)
	{
		fprintf(stderr,"can't open output file %s\n",graph_file_name);
		exit(1);
	}

	for(int i=0;i<graph.numNode;i++)
	{		
		fprintf(fp,"%d",i);
		int numNB = graph.numNeighbor[i];
		for (int j=0; j<numNB; j++)		
			fprintf(fp, " %d:%f", graph.idx[i][j], graph.weight[i][j]);			// sparse representation - (index and value)
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void ComputeGraphWeight(SGraphStruct& graph, svm_parameter *param)
{
	int numNode     = graph.numNode;
	int maxNeighbor = 0;
	int i;
	for (i = 0; i < numNode; i++)
		if (graph.numNeighbor[i] > maxNeighbor)
			maxNeighbor = graph.numNeighbor[i];
	IdxVal *resultlist = new IdxVal [maxNeighbor];
	for (i = 0; i < numNode; i++)
	{
		int j;
		int numNeighbor = graph.numNeighbor[i];
		for (j = 0; j < numNeighbor; j++)
		{
			resultlist[j].idx   = graph.idx[i][j];
			resultlist[j].value = graph.weight[i][j];
		}
		std::sort(resultlist, resultlist+numNeighbor, CompareLessIdx);
		for (j = 0; j < numNeighbor; j++)
		{
			graph.idx[i][j]    = resultlist[j].idx;
			graph.weight[i][j] = resultlist[j].value;
		}
	}
	delete [] resultlist;

	if (param->weight_type == 0)
	{	
		for (i = 0; i < numNode; i++)
		{			
			int numNeighbor = graph.numNeighbor[i];
			for (int j = 0; j < numNeighbor; j++)
				graph.weight[i][j] = 1;			
		}
	}
	else if (param->weight_type > 0)
	{
		for (i = 0; i < numNode; i++)
		{
			int numNeighbor = graph.numNeighbor[i];
			for (int j = 0; j < numNeighbor; j++)
			{
				double tmpW        = exp(-param->weight_type*graph.weight[i][j]);
				graph.weight[i][j] = (Wfloat)sqrt(tmpW);			
			}			
		}
	}
	else
	{
		double avgDist = 0.0;
		int numEdge    = 0;
		for (i = 0; i < numNode; i++)
		{		
			int numNeighbor = graph.numNeighbor[i];
			numEdge        += numNeighbor;
			for (int j = 0; j < numNeighbor; j++)			
				avgDist += graph.weight[i][j];			
		}
		avgDist /= numEdge;
		printf ("w width=%f\n", 1.0/avgDist);
		for (i = 0; i < numNode; i++)
		{
#ifdef PRINT_GRAPH
			printf("%d-> ",i);			
#endif
			int numNeighbor = graph.numNeighbor[i];
			for (int j = 0; j < numNeighbor; j++)
			{
				double tmpW        = exp(-graph.weight[i][j]/avgDist);
				graph.weight[i][j] = (Wfloat)sqrt(tmpW);				

#ifdef PRINT_GRAPH
				printf("%d:%.3g ",graph.idx[i][j],graph.weight[i][j]);
#endif
			}			
		}
	}	
}


void ComputeNGraphWeight(SGraphStruct& graph, svm_parameter *param)
{
	int numNode     = graph.numNode;
	int maxNeighbor = 0;
	int i;
	for (i = 0; i < numNode; i++)
		if (graph.numNeighbor[i] > maxNeighbor)
			maxNeighbor = graph.numNeighbor[i];
	IdxVal *resultlist = new IdxVal [maxNeighbor];
	for (i = 0; i < numNode; i++)
	{
		int j;
		int numNeighbor = graph.numNeighbor[i];
		for (j = 0; j < numNeighbor; j++)
		{
			resultlist[j].idx   = graph.idx[i][j];
			resultlist[j].value = graph.weight[i][j];
		}
		std::sort(resultlist, resultlist+numNeighbor, CompareLessIdx);
		for (j = 0; j < numNeighbor; j++)
		{
			graph.idx[i][j]    = resultlist[j].idx;
			graph.weight[i][j] = resultlist[j].value;
		}
	}
	delete [] resultlist;

	if (param->weight_type == 0)
	{			
		for (i = 0; i < numNode; i++)
		{			
			int numNeighbor = graph.numNeighbor[i];
			Wfloat degVal   = (Wfloat)sqrt((Wfloat)numNeighbor);
			graph.degree[i] = degVal;
			for (int j = 0; j < numNeighbor; j++)
				graph.weight[i][j] = 1;			
		}
	}
	else if (param->weight_type > 0)
	{
		for (i = 0; i < numNode; i++)
		{
			graph.degree[i] = 0.0;
			int numNeighbor = graph.numNeighbor[i];
			for (int j = 0; j < numNeighbor; j++)
			{
				double tmpW        = exp(-param->weight_type*graph.weight[i][j]);
				graph.weight[i][j] = (Wfloat)sqrt(tmpW);
				graph.degree[i]   += (Wfloat)tmpW;
			}
			graph.degree[i] = sqrt(graph.degree[i]);			
		}
	}
	else
	{
		double avgDist = 0.0;
		int numEdge    = 0;
		for (i = 0; i < numNode; i++)
		{		
			int numNeighbor = graph.numNeighbor[i];
			numEdge        += numNeighbor;
			for (int j = 0; j < numNeighbor; j++)			
				avgDist += graph.weight[i][j];			
		}
		avgDist /= numEdge;
		printf ("w width=%f\n", 1.0/avgDist);
		for (i = 0; i < numNode; i++)
		{
#ifdef PRINT_GRAPH
			printf("%d-> ",i);
#endif
			graph.degree[i] = 0.0;
			int numNeighbor = graph.numNeighbor[i];
			for (int j = 0; j < numNeighbor; j++)
			{
				double tmpW        = exp(-graph.weight[i][j]/avgDist);
				graph.weight[i][j] = (Wfloat)sqrt(tmpW);
				graph.degree[i]   += (Wfloat)tmpW;

#ifdef PRINT_GRAPH
				printf("%d:%.3g ",graph.idx[i][j],graph.weight[i][j]);
#endif
			}
			graph.degree[i] = sqrt(graph.degree[i]);

#ifdef PRINT_GRAPH
			printf("deg:%.3g\n",graph.degree[i]);
#endif
		}
	}	
}


void InitializeGraph (struct SGraphStruct& graph, int size)
{
	graph.numNode	  = size;
	graph.numNeighbor = new unsigned char[size];
	graph.idx         = new int*[size];
	graph.weight      = new Wfloat*[size];
	graph.degree      = NULL;
}


void DestroyGraph (struct SGraphStruct& graph)
{
	delete [] graph.numNeighbor;
	for (int i=0; i<graph.numNode; i++)
	{
		delete [] graph.idx[i];
		delete [] graph.weight[i];		
	}
	delete [] graph.idx;
	delete [] graph.weight;	
	graph.idx	 = NULL;
	graph.weight = NULL;
	if (graph.degree != NULL)
	{
		delete [] graph.degree;	
	    graph.degree = NULL;	
	}
}
