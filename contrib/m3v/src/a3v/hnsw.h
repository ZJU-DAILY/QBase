#pragma once

#include<iostream>
#include "hnswlib.h"
#include<vector>
#include<queue>

extern "C"{
    #include "postgres.h"
    #include "storage/bufmgr.h"
	#include "access/tableam.h"
	#include "access/genam.h"
	#include "executor/executor.h"
}

const int multi_range_k = 50;
const int terminate_multi_top_k = 50;
const bool enable_hnsw_crack_init = false;
// const int check_thresold = 1.27;
// const float threshold_control = 0.5;
using ScorePair = std::pair<float,std::uint64_t>;
std::uint64_t GetNumberByItemPointerData(ItemPointer tid);
ItemPointerData GetItemPointerDataByNumber(hnswlib::labeltype label);
class MultiColumnHnsw{
	public:
		float l2_distance(std::vector<float> &data1,const float* query_point);
		bool  CheckFilter(ItemPointerData tid);
		float RankScore(hnswlib::labeltype label);
		MultiColumnHnsw(std::vector<std::shared_ptr<hnswlib::HierarchicalNSW<float>>> &hnsws_,std::vector<float*> &query_points_,
		int k_,bool &xs_inorder_scan_,float range_,std::vector<float> &weights_,int filter_amplication_k_,std::string relname_text_,
		std::string filter_text_,Relation heap_rel_,IndexScanDesc scan_):
		inRange(false),distanceThreshold(2.0),distanceQueueThreshold(20),
		hnsws(hnsws_),query_points(query_points_),k(k_),xs_inorder_scan(xs_inorder_scan_),range(range_),RangeTimes(0),weights(weights_),
		filter_amplication_k(filter_amplication_k_),filter_text(filter_text_),relname_text(relname_text_),heap_rel(heap_rel_),scan(scan_)
		{
			// slot = table_slot_create(heap_rel, NULL);
			// get result_iterator for now.
			for(int i = 0;i < hnsws.size();i++){
				hnsws_iterators.push_back(std::make_shared<hnswlib::ResultIterator<float>>(hnsws[i].get(), (const void*)query_points[i]));
			}
		}

		void SetEf(int ef_){
			for(int i = 0;i < hnsws.size();i++){
				hnsws[i]->setEf(ef_);
			}
		}
		
		bool GetNext();
		bool RangeNext();
		bool GetSingleNext();
		bool GetNewNext();
		bool WeightedRoundRobin();
		std::vector<std::shared_ptr<hnswlib::HierarchicalNSW<float>>> hnsws;
		// std::vector<std::shared_ptr<hnswlib::ResultIterator<float>>> hnsws_iterators;
		std::vector<std::shared_ptr<hnswlib::ResultIterator<float>>> hnsws_iterators;
		std::vector<float*> query_points;
		std::vector<float> weights;
		// <score,tid_number> s
		std::priority_queue<ScorePair> result_pq; // MaxHeap
		std::priority_queue<ScorePair,std::vector<ScorePair>,std::greater<ScorePair>> proc_pq; // MinHeap
		std::priority_queue<float,std::vector<float>,std::greater<float>> distanceQueue;
		std::unordered_set<std::uint64_t> seen_tid;
		ItemPointerData result_tid;
		hnswlib::labeltype label;
		float distance;
		int k;
		bool& xs_inorder_scan;
		float range;
		float distanceQueueThreshold{30};
		int	  distanceThreshold{2};
		bool  inRange{false};
		int   RangeTimes{0};
		int   filter_amplication_k;
		std::string filter_text;
		std::string relname_text;
		Relation 	heap_rel;
		IndexScanDesc scan;
		TupleTableSlot *slot;
		bool 		first_multi_column{true};
};
