#pragma once

#ifdef __cplusplus
extern "C"
{
#endif
	// void Print();
#ifdef __cplusplus
}
#endif
static bool hnsw_auxiilary_init = true;
static int hnsw_single_top_k = 1200;
static int a3v_top_k = 0;
static double top_k_based_distance = 1.0;
#include <iostream>
#include <vector>
#include "m3v.h"
#include<sstream>
extern "C"
{
	#include <postgres.h>
	#include <utils/builtins.h>
	#include "storage/bufmgr.h"
}

#define DEFAULT_INDEX_TYPE true // memory index
#define CLOSE_QUERY_THRESHOLD 128
#define CLOSE_MIN_QUERY_THRESHOLD 0.1
#define CLOSE_MAX_QUERY_THRESHOLD 128
static ItemPointerData InvalidItemPointerData = {{0,0},InvalidOffsetNumber};
typedef uint32_t PageId;
#define FLOAT_SIZE sizeof(float)

std::string floatArrayToString(float* arr,int n);

template<class T>
std::vector<T> DeserializeVector(const std::string& filename) {
    std::vector<T> data;
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Cannot open the file for reading.\n";
        return data; // 返回空向量
    }

    T item;
    while (inFile.read((char*)&item, sizeof(T))) {
        data.push_back(item);
    }

    inFile.close();
    return data;
}

template<class T>
void SerializeVector(const std::vector<T>& data, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Cannot open the file for writing.\n";
        return;
    }

    for (const auto& item : data) {
        outFile.write((const char*)&item, sizeof(T));
    }

    outFile.close();
}

bool file_exists(const std::string& path);
OffsetNumber WriteA3vTupleToPage(Page page,Item item,Size size);
PageId CombineGetPageId(uint16 bi_hi,uint16 bi_lo);
BlockIdData SplitPageId(uint32_t page_id);
int a3vAllocateQueryId(Relation index);
std::string build_data_string(float* vec,int len);
void InsertNewQuery(IndexScanDesc scan,m3vScanOpaque so,int block_number);
Buffer m3vNewBuffer(Relation index, ForkNumber forkNum);
/*
 * Get the max number of connections in an upper layer for each element in the index
 */
bool A3vMemoryIndexType(Relation index);
float A3vCloseQueryThreshold(Relation index);
std::string extract_btree_filter(const char* source_text);
