#include "hnsw.h"
#include "simd_func.h"
#include "spi.h"
#include "util.h"

std::uint64_t GetNumberByItemPointerData(ItemPointer tid){
	std::int32_t blockId = ItemPointerGetBlockNumberNoCheck(tid);
    std::int32_t offset = ItemPointerGetOffsetNumberNoCheck(tid);
    std::uint64_t number = blockId;
    number = (number << 32) + offset;
	return number;
}

ItemPointerData GetItemPointerDataByNumber(hnswlib::labeltype label){
    ItemPointerData tid;
    BlockNumber blkno = (std::uint32_t) (label >> 32);
    OffsetNumber offset = (std::uint32_t) label;
    ItemPointerSet(&tid, blkno, offset);
    return tid;
}

float MultiColumnHnsw::l2_distance(std::vector<float> &data1,const float* query_point){
    // float res = 0.0;
    // for(int i = 0;i <data1.size();i++){
    //     float temp = (data1[i] -query_point[i]); temp = temp * temp;
    //     res += temp;
    // }
    // return res;
    int dim = (int)data1.size();
    return optimized_simd_distance_func(data1.data(),const_cast<float*>(query_point),dim);
}

float MultiColumnHnsw::RankScore(hnswlib::labeltype label){
    float score = 0.0;
    for(int i = 0;i < query_points.size();i++){
        std::vector<float> hit_data = hnsws[i]->getDataByLabel<float>(label);
        score += l2_distance(hit_data,query_points[i]) * weights[i];
    }
    return score;
}

bool MultiColumnHnsw::CheckFilter(ItemPointerData tid){
    scan->xs_heaptid = tid;
    index_fetch_heap(scan, slot);
    scan->ps_ExprContext->ecxt_scantuple = slot;
    return (scan->qual == NULL || ExecQual(scan->qual, scan->ps_ExprContext));
}

bool MultiColumnHnsw::RangeNext(){
    while(true){
        auto result = hnsws_iterators[0]->Next();
        result_tid = GetItemPointerDataByNumber(result->GetLabel());
        distance = result->GetDistance();
        if(!result->HasResult()) return false;
        if(!inRange){
            if(distanceQueue.size() < distanceQueueThreshold ||
            distanceQueue.top() > result->GetDistance()){
                if(distanceQueue.size() >= distanceQueueThreshold) distanceQueue.pop();
                distanceQueue.push(result->GetDistance());
            }else{
                return false;
            }
        }
        if(result->GetDistance() > range){
            RangeTimes++;
            if(inRange && RangeTimes >= distanceThreshold){
                return false;
            }else{
                continue;
            }
        }else{
            inRange = true;
        }
        return true;
    }
}

bool MultiColumnHnsw::GetSingleNext(){
    auto result = hnsws_iterators[0]->Next();
    // elog(INFO,"range: %d",range);
    if(result->HasResult()){
        result_tid = GetItemPointerDataByNumber(result->GetLabel());
        label = result->GetLabel();
        distance = result->GetDistance();
        if(!xs_inorder_scan){
            if (distanceQueue.size() == range &&
                distanceQueue.top() < result->GetDistance())
            {
                // elog(INFO,"xs_inorder_scan is true now.");
                xs_inorder_scan=true;
            }
            else
            {
                if (distanceQueue.size() >= range)
                {
                    distanceQueue.pop();
                }
                distanceQueue.push(result->GetDistance());
            }
        }
        return true;
    }
    // elog(LOG,"hnsw has no more.");
    return false;
}

bool MultiColumnHnsw::WeightedRoundRobin(){
    if(first_multi_column){
        first_multi_column = false;
        // check every column to get amplication_k, and do score computation.
        // we do a weighted iteration for MultiColumnHnsw Search
        std::vector<int> ks;
        float weights_sum = 0.0;
        int k_sum = std::max(filter_amplication_k,hnsw_single_top_k) * weights.size();
        for(int i = 0;i < weights.size();i++){
            weights_sum += weights[i];
        }
        for(int i = 0;i < weights.size();i++){
            ks.push_back((weights[i]/weights_sum) * k_sum);
        }
        for(int col = 0;col < query_points.size();col++){
            auto result = hnsws_iterators[col]->Next();
            int cur_nums = 0;
            while(result->HasResult() && cur_nums < ks[col]){ // 4000
                hnswlib::labeltype label = result->GetLabel();
                if(seen_tid.find(label) != seen_tid.end()){
                    result = hnsws_iterators[col]->Next();
                    continue;
                }
                seen_tid.insert(label);
                cur_nums++;
                result = hnsws_iterators[col]->Next();
            }
            // elog(INFO,"finish search here");
            if(!result->HasResult()){
                break;
            }
        }
        for (const hnswlib::labeltype& label : seen_tid) {
            float score = RankScore(label);
			proc_pq.push(std::make_pair(score, label));
        }
    }
    xs_inorder_scan = true;
    while(!proc_pq.empty()){
        result_tid =  GetItemPointerDataByNumber(proc_pq.top().second);proc_pq.pop();
        return true;
    }
    return false;
}

bool MultiColumnHnsw::GetNewNext(){
    if(first_multi_column){
        first_multi_column = false;
        // check every column to get amplication_k, and do score computation.
        for(int col = 0;col < query_points.size();col++){
            auto result = hnsws_iterators[col]->Next();
            int cur_nums = 0;
            while(result->HasResult() && cur_nums < std::max(filter_amplication_k,hnsw_single_top_k) ){ // 4000
                hnswlib::labeltype label = result->GetLabel();
                if(seen_tid.find(label) != seen_tid.end()){
                    result = hnsws_iterators[col]->Next();
                    continue;
                }
                seen_tid.insert(label);
                cur_nums++;
                result = hnsws_iterators[col]->Next();
            }
            // elog(INFO,"finish search here");
            if(!result->HasResult()){
                break;
            }
        }
        for (const hnswlib::labeltype& label : seen_tid) {
            float score = RankScore(label);
			proc_pq.push(std::make_pair(score, label));
        }
    }
    xs_inorder_scan = true;
    while(!proc_pq.empty()){
        result_tid =  GetItemPointerDataByNumber(proc_pq.top().second);proc_pq.pop();
        return true;
    }
    return false;
}

bool MultiColumnHnsw::GetNext(){
    float score = 0.0;
    bool finished = false;
    int  consecutive_drops = 0;
    ItemPointerData tid;
    HeapTupleData mytup;
    Buffer buf;
    double time_cost = 0.0;
    // auto begin_query = std::chrono::steady_clock::now();
    while(!finished){
        for(int col = 0; col < query_points.size();col++){
            const void* query_point = query_points[col];
            auto result = hnsws_iterators[col]->Next();
            if(!result->HasResult()){
                xs_inorder_scan = true;
                finished = true;break;
            }
            tid = GetItemPointerDataByNumber(result->GetLabel());
            // auto begin_query = std::chrono::steady_clock::now();
            // if(filter_text!=""){
            //     while(!CheckFilter(tid)&&result->HasResult()){
            //         result = hnsws_iterators[col]->Next();
            //         if(!result->HasResult()){
            //             xs_inorder_scan = true;finished = true;break;
            //         }
            //         tid = GetItemPointerDataByNumber(result->GetLabel());
            //     }
            // }
            // auto end_query = std::chrono::steady_clock::now();
            // elog(INFO,"filter_tuple time cost %.2f millseconds",std::chrono::duration<double, std::milli>(end_query - begin_query).count());
            hnswlib::labeltype label = result->GetLabel();
            distance = result->GetDistance();
            if(seen_tid.find(label) != seen_tid.end()){
                continue;
            }
            seen_tid.insert(label);
            consecutive_drops++;
            auto begin_query = std::chrono::steady_clock::now();
            float score = RankScore(label);
            auto end_query = std::chrono::steady_clock::now();
            elog(LOG,"RankScore time cost %.2f millseconds",std::chrono::duration<double, std::milli>(end_query - begin_query).count());
            if(proc_pq.size() < k || proc_pq.top().first > score){
				if(proc_pq.size() == k)
					proc_pq.pop();
				proc_pq.push(std::make_pair(score, label));
				consecutive_drops = 0;
			}
        }

        if(consecutive_drops >= terminate_multi_top_k){
            finished = true;break;
        }
    }
    // elog(INFO,"consecutive_drops: %d",consecutive_drops);
    // auto end_query = std::chrono::steady_clock::now();
    while(!proc_pq.empty()){
        result_pq.push(proc_pq.top());proc_pq.pop();
    }
    // elog(INFO,"GetNext time cost %.2f millseconds",std::chrono::duration<double, std::milli>(end_query - begin_query).count());
    if(result_pq.empty()){
        return false;
    }else{
        result_tid = GetItemPointerDataByNumber(result_pq.top().second);  
        // elog(LOG,"tid:(%d,%d)",ItemPointerGetBlockNumberNoCheck(&tid),ItemPointerGetOffsetNumberNoCheck(&tid));
        result_pq.pop();  
        return true;
    }
}
