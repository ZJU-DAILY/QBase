#include "a3v_async_server.h"
#include "init.h"
#include <iostream>
std::mutex mtx;
std::condition_variable cv;
std::queue<std::shared_ptr<Message>> channel;

// async a3v recieve server
void A3vAsyncRecieveServer() {
    while (true) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, []{ return !channel.empty(); }); 
        std::shared_ptr<Message> data = channel.front();
        channel.pop();
        lock.unlock(); 
        if(!memory_init.memory_indexes.count(data->path_key)){
            elog(INFO,"can't find path_key %s in A3vAsyncRecieveServer",data->path_key.c_str());
        }
        if(memory_init.memory_indexes[data->path_key].size() - 1 < data->a3v_id){
            elog(ERROR,"can't find a3v_id(overflow) %d in A3vAsyncRecieveServer",data->a3v_id);
        }
        // elog(LOG,"get a a3v message");
        // carefully open elog for async_server, it will affect the postgres clien results.
        auto a3v_index = memory_init.memory_indexes[data->path_key][data->a3v_id];
        if(a3v_index->query_records.load() < A3V_HINT_QUERY_RECORDS){
            if(data->query_type == KNN_QUERY_MESSAGE){
                // we use hnsw index to accelerate a3v init
                std::priority_queue<PQNode> result_pq;
                a3v_index->KnnCrackSearch(*data->weights,data->query_point->data(),data->k,result_pq,*data->dimensions,a3v_index->last_top_k_mean,true);
                a3v_index->query_records.fetch_add(1);
                a3v_index->last_top_k_mean =  (a3v_index->last_top_k_mean * (a3v_index->query_records -1) + result_pq.top().first)/(a3v_index->query_records);
                // elog(LOG,"a3v_index->last_top_k_mean: %.2lf", a3v_index->last_top_k_mean);
            }else if(data->query_type == RANGE_QUERY_MESSAGE){
                std::vector<PQNode> result_ids;
                a3v_index->RangeCrackSearch(*data->weights,data->query_point->data(),data->radius,result_ids,*data->dimensions);
                a3v_index->query_records.fetch_add(1);
            }else if(data->query_type == KNN_QUERY_HNSW_INIT_MESSAGE && !a3v_index->use_hnsw_init){
                // elog(LOG,"Perform A KnnAuxiliaryInit");
                // KNN HNSW INIT MESSAGE, we try twenty, for now, we just support single vector search
                a3v_index->use_hnsw_init = true;
                int k = data->k * 1000;
                int cur = 0;
                std::vector<PQNode> indexes;
                // data->send_hard_hnsws->SetEf(1024);
                // auto begin_query = std::chrono::steady_clock::now();
                while(data->send_hard_hnsws->GetSingleNext() && cur < k){
                    int label = data->send_hard_hnsws->label;
                    float distance = data->send_hard_hnsws->distance;
                    indexes.push_back({distance,label});
                    cur++;
                }
                // auto end_query1 = std::chrono::steady_clock::now();
		        // elog(LOG,"cost1 %.2f millseconds",std::chrono::duration<double, std::milli>(end_query1 - begin_query).count());
                // a3v_index->KnnAuxiliaryInit(*data->weights,data->query_point->data(),*data->dimensions,indexes);
                // auto end_query2 = std::chrono::steady_clock::now();
		        // elog(LOG,"cost2 %.2f millseconds",std::chrono::duration<double, std::milli>(end_query2 - end_query1).count());
                delete data->send_hard_hnsws;
                a3v_index->query_records.fetch_add(1);
            }
        }
        // elog(LOG,"finish a3v messgae processing");
    }
    //  elog(LOG,"core dump");
}

// async a3v send server
void A3vAsyncSendServer(std::shared_ptr<Message> message) {
    elog(INFO,"Send a message");
    // auto begin_query = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto weights = message->weights;
        channel.push(message); 
    }
    cv.notify_one();
    // auto end_query = std::chrono::steady_clock::now();
    // std::cout<<"time cost "<<std::chrono::duration<double, std::milli>(end_query - begin_query).count()<<"millseconds"<<std::endl;
}
