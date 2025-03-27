#pragma once

#include "a3v/m3v.h"
#include "hnswlib.h"
#include "init.h"
#include <tuple>
#include "a3v_async_server.h"
#include "memory_a3v.h"
extern "C"{
	#include "postgres.h"
	#include "access/relscan.h"
	#include "pgstat.h"
	#include "storage/bufmgr.h"
	#include "storage/lmgr.h"
	#include "utils/memutils.h"
	#include "utils/float.h"
}
#define SingleSearchType 1
#define MultiColumnSearchType 2
const int RANGE_QUERY_THRESHOLD_TIMES = 10;
#define MAX_KNN_DISTANCE get_float8_infinity()
#define KNN_QUERY(scan) (scan->orderByData != NULL && ((scan->orderByData->sk_flags & SK_ISNULL) == false))
#define RANGE_QUERY(scan) (scan->keyData != NULL && ((scan->keyData->sk_flags & SK_ISNULL) == false))
#define CRACKTHRESHOLD 128
#define VECTOR_FILTER_SELECTIVITY_THRESHOLD 0.03
#define VECTOR_FILTER_SELECTIVITY_HIGH_BOUND 0.6
#define MAX_K 4200
// https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=AB48C5606D99B76204CD6A51067CFB7F?doi=10.1.1.75.10014&rep=rep1&type=pdf
float8 getKDistance(pairingheap *heap, int k, int nn_size);
/**
 * postgres perf tutorial: https://www.youtube.com/watch?v=HghP4D72Noc
 */

/**
 * 1. page: current search page
 * 2. q: target vector
 * 3. k: order by a <=> q limit k;limit push down
 * 4. distance: distance of q and parent_page_entry
 *
 * For now, this is a bad implementation, we should add node when visiting inner node into nn, and then when we use this func,
 * remove the correlated node in nn. But this is a tricky implementation, so hold on for now.
 * q should be a vector**, it's a two dimension pointer array
 * 
 * we need two pq to do knn search.
 * 1. guide_pq. small heap
 * 2. result_pqs. big heap
 */
void getKNNRecurse(Relation index, BlockNumber blkno, Datum q, int k, float8 distance, pairingheap *p, pairingheap *nn, FmgrInfo *procinfo, Oid collation, int *nn_size,int columns)
{
	// elog(INFO, "get knn blkno: %d", blkno);
	Buffer buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	Page page = BufferGetPage(buf);
	OffsetNumber offsets = PageGetMaxOffsetNumber(page);
	// InternalPage
	if (PageType(m3vPageGetOpaque(page)->type) == M3V_INNER_PAGE_TYPE)
	{
		bool root_flag = is_root(m3vPageGetOpaque(page)->type);
		for (OffsetNumber offset = FirstOffsetNumber; offset <= offsets; offset = OffsetNumberNext(offset))
		{
			m3vElementTuple etup = (m3vElementTuple)PageGetItem(page, PageGetItemId(page, offset));
			if (!root_flag && ((abs(etup->distance_to_parent - distance)) - etup->radius <= getKDistance(nn, k, *nn_size)) || root_flag)
			{
				float8 target_to_parent_dist = GetPointerDistances(etup->vecs, reinterpret_cast<Vector**>(DatumGetPointer(q)), procinfo, collation,columns);
				float8 estimated_dist = max((target_to_parent_dist - etup->radius), 0);
				if (estimated_dist <= getKDistance(nn, k, *nn_size))
				{
					m3vKNNCandidate *candidate = static_cast<m3vKNNCandidate*>(palloc0(sizeof(m3vKNNCandidate)));
					// tid is invalid now.
					candidate->son_blkno = etup->son_page;
					candidate->distance = estimated_dist;
					candidate->target_parent_distance = target_to_parent_dist;
					// min heap
					pairingheap_add(p,reinterpret_cast<pairingheap_node*>(Createm3vPairingKNNNode(candidate)));
				}

				if (target_to_parent_dist + etup->radius <= getKDistance(nn, k, *nn_size))
				{
					// NN_Update task
					m3vKNNCandidate *candidate_nn = static_cast<m3vKNNCandidate*>(palloc0(sizeof(m3vKNNCandidate)));
					// not leaf, don't update tid
					candidate_nn->distance = target_to_parent_dist + etup->radius;
					candidate_nn->tid = NULL;
					// elog(INFO, "KNN Internal Distance: %f, addr: %p", candidate_nn->distance, candidate_nn->tid);
					// PrintVector("and again knn vec: ", &etup->vec);
					// max heap
					// pairingheap_add(nn, Createm3vPairingKNNNode(candidate_nn));
					// m3vPairingKNNNode *n = (m3vPairingKNNNode *)pairingheap_first(nn);
					// *nn_size = *nn_size + 1;
					// while (*nn_size > k)
					// {
					// 	pairingheap_remove_first(nn);
					// 	*nn_size = *nn_size - 1;
					// }
				}
			}
		}
	}
	else
	{
		// LeafPage
		for (OffsetNumber offset = FirstOffsetNumber; offset <= offsets; offset = OffsetNumberNext(offset))
		{
			m3vElementLeafTuple etup = (m3vElementLeafTuple)PageGetItem(page, PageGetItemId(page, offset));
			if ((abs(etup->distance_to_parent - distance)) <= getKDistance(nn, k, *nn_size))
			{
				float8 dist = GetPointerDistances(etup->vecs,reinterpret_cast<Vector**>(q), procinfo, collation,columns);
				if (dist <= getKDistance(nn, k, *nn_size))
				{
					// NN_Update task
					m3vKNNCandidate *candidate_nn = static_cast<m3vKNNCandidate*>(palloc0(sizeof(m3vKNNCandidate)));
					// not leaf, don't update tid
					// this is accurate distance
					candidate_nn->distance = dist;
					candidate_nn->tid = &etup->data_tid;
					// elog(INFO, "KNN Leaf Distance: %f, addr: %p", candidate_nn->distance, candidate_nn->tid);
					// max heap
					pairingheap_add(nn, reinterpret_cast<pairingheap_node*>(Createm3vPairingKNNNode(candidate_nn)));
					*nn_size = *nn_size + 1;
					while (*nn_size > k)
					{
						pairingheap_remove_first(nn);
						*nn_size = *nn_size - 1;
					}
				}
			}
		}
	}
	UnlockReleaseBuffer(buf);
}

float8 getKDistance(pairingheap *nn, int k, int nn_size)
{
	if (pairingheap_is_empty(nn) || nn_size < k)
	{
		return MAX_KNN_DISTANCE;
	}
	else
	{
		m3vPairingKNNNode *n = (m3vPairingKNNNode *)pairingheap_first(nn);
		return ((m3vPairingKNNNode *)pairingheap_first(nn))->inner->distance;
	}
}

/**
 * KNN Query
 */
static List *
GetKNNScanItems(IndexScanDesc scan, Datum q)
{
	m3vScanOpaque so = (m3vScanOpaque)scan->opaque;
	Relation index = scan->indexRelation;
	FmgrInfo *procinfo = so->procinfo;
	Oid collation = so->collation;
	List *w = NIL;
	m3vMetaPage metap;
	m3vElementTuple etup;
	Buffer buf;
	OffsetNumber offsets;
	Page page;
	/* Get MetaPageData */
	m3vMetaPageData meta = m3vGetMetaPageInfo(index);
	metap = &meta;
	int columns = metap->columns;
	BlockNumber root_block = metap->root;
	// limit push down
	int limits = scan->orderByData->KNNValues;
	if (limits <= 0)
	{
		// DebugEntirem3vTree(m3vGetMetaPageInfo(index).root, index, 0);
		elog(ERROR, "knn query should specify a valid k value");
	}
	// /* Just SImple Test Read Tuple Data */
	// buf = ReadBuffer(index, root_block);
	// LockBuffer(buf, BUFFER_LOCK_SHARE);
	// page = BufferGetPage(buf);
	// etup = (m3vElementTuple)PageGetItem(page, PageGetItemId(page, FirstOffsetNumber));
	// PrintVector("knn scan ", &etup->vec);
	// w = lappend(w, etup);
	// UnlockReleaseBuffer(buf);
	// DebugEntirem3vTree(metap->root, index, 0);

	/**
	 *	Olc Algorithm: Read And Release
	 */
	// min heap
	pairingheap *p = pairingheap_allocate(CompareKNNCandidatesMinHeap, NULL);
	// max heap (use nn to return)
	pairingheap *nn = pairingheap_allocate(CompareKNNCandidates, NULL);
	m3vKNNCandidate *candidate = static_cast<m3vKNNCandidate*>(palloc0(sizeof(m3vKNNCandidate)));
	// if son_blkno is InvalidBlockNumber, so it's a leaf
	candidate->son_blkno = root_block;
	int nn_size = 0;
	// DebugEntirem3vTree(root_block, index, 0);
	pairingheap_add(p, reinterpret_cast<pairingheap_node*>(Createm3vPairingKNNNode(candidate)));
	while (!pairingheap_is_empty(p))
	{
		m3vPairingKNNNode *node = (m3vPairingKNNNode *)pairingheap_remove_first(p);
		// prune data
		if (node->inner->distance > getKDistance(nn, limits, nn_size))
		{
			continue;
		}
		getKNNRecurse(index, node->inner->son_blkno, q, limits, node->inner->target_parent_distance, p, nn, procinfo, collation, &nn_size,columns);
	}

	while (!pairingheap_is_empty(nn))
	{
		m3vPairingKNNNode *node = (m3vPairingKNNNode *)pairingheap_remove_first(nn);
		w = lappend(w, node->inner->tid);
	}
	return w;
}

void range_query(Relation index, Datum q, List **w, BlockNumber blkno, float8 radius, float8 distance, FmgrInfo *procinfo, Oid collation,int columns)
{
	Buffer buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	Page page = BufferGetPage(buf);
	m3vPageOpaque opaque = m3vPageGetOpaque(page);
	OffsetNumber offsets = PageGetMaxOffsetNumber(page);
	if (PageType(opaque->type) == M3V_LEAF_PAGE_TYPE)
	{
		for (OffsetNumber offset = FirstOffsetNumber; offset <= offsets; offset = OffsetNumberNext(offset))
		{
			m3vElementLeafTuple etup = (m3vElementLeafTuple)PageGetItem(page, PageGetItemId(page, offset));
			if (abs(etup->distance_to_parent - distance) <= radius)
			{
				float8 dist = GetPointerDistances(etup->vecs, reinterpret_cast<Vector**>(q), procinfo, collation,columns);
				if (dist <= radius)
				{
					*w = lappend(*w, &etup->data_tid);
				}
			}
		}
	}
	else
	{
		if (is_root(opaque->type))
		{
			for (OffsetNumber offset = FirstOffsetNumber; offset <= offsets; offset = OffsetNumberNext(offset))
			{
				m3vElementTuple etup = (m3vElementTuple)PageGetItem(page, PageGetItemId(page, offset));
				float8 dist = GetPointerDistances(etup->vecs, reinterpret_cast<Vector**>(q), procinfo, collation,columns);
				if (etup->radius + radius >= dist)
				{
					range_query(index, q, w, etup->son_page, radius, dist, procinfo, collation,columns);
				}
			}
		}
		else
		{
			for (OffsetNumber offset = FirstOffsetNumber; offset <= offsets; offset = OffsetNumberNext(offset))
			{
				m3vElementTuple etup = (m3vElementTuple)PageGetItem(page, PageGetItemId(page, offset));
				PrintVectors("range scan vector: ", etup->vecs,columns);
				if (abs(etup->distance_to_parent - distance) <= radius + etup->radius)
				{
					// bug !!!!
					float8 dist = GetDistance(reinterpret_cast<Datum>(etup->vecs), q, procinfo, collation);
					if (dist <= radius + etup->radius)
					{
						range_query(index, q, w, etup->son_page, radius, dist, procinfo, collation,columns);
					}
				}
			}
		}
	}
	UnlockReleaseBuffer(buf);
}

/**
 * Range Query
 */
static List *
GetRangeScanItems(IndexScanDesc scan, Datum q, float8 radius,int columns)
{
	/*
	 * range query 查询算法流程:
	 *	从n出发找到所有距离targe小于r的点
	 * 	RangeQuery(Object n,Object target,double r,double target_parent_distance)
	 *   1. n如果是叶子节点
	 *		遍历每一个entry
	 *		if |distance_p(entry) - target_parent_distance| <= r
	 *			if distance(entry,target) <= r
	 *				add entry to result.
	 *	2. n如果是内部节点
	 *		2.1 root节点
	 *			遍历每一个entry
	 *			if R(entry) + r >= dist(entry,targe)
	 *				RangeQuery(entry.son,target,r,distance(entry,target));
	 *		2.2 非root节点
	 *			遍历每一个entry
	 *			if |distance_p(entry) - target_parent_distance| <= r + R(entry)
	 *				if distance(entry,target) <= R(entry) + r
	 *					RangeQuery(entry.son,target,r,distance(entry,target));
	 */
	List *w = NIL;
	m3vScanOpaque so = (m3vScanOpaque)scan->opaque;
	/* Get MetaPageData */
	Relation index = scan->indexRelation;
	m3vMetaPageData meta = m3vGetMetaPageInfo(index);
	m3vMetaPage metap = &meta;
	BlockNumber root_block = metap->root;
	FmgrInfo *procinfo = so->procinfo;
	Oid collation = so->collation;
	range_query(index, q, &w, root_block, radius, 0, procinfo, collation,metap->columns);
	return w;
}

static Datum
GetScanValue2(IndexScanDesc scan,float8 * weights,bool is_knn,int nums){

}

/*
 * Get scan value, value should be a Vector** 
 */
static Datum
GetScanValue(IndexScanDesc scan,Vector** values,float8 * weights,bool is_knn,int nums)
{
	// every time we need to query the nearest query point root, so we can get a3v tree.
	m3vScanOpaque so = (m3vScanOpaque)scan->opaque;
	// get multi vector columns
	if(is_knn){
		// scan->numberOfOrderBys == 1, it means it's a single metric query.
		for(int i = 0;i < scan->numberOfOrderBys; i++){
			weights[i] = DatumGetFloat8(scan->orderByData[i].w);
			values[i] = DatumGetVector(scan->orderByData[i].sk_argument);
			// elog(INFO,"w: %lf",weights[i]);
			PrintVector("vector knn: ",values[i]);
		}
	}else{
		for(int i = 0;i < scan->numberOfKeys; i++){
			weights[i] = DatumGetFloat8(scan->keyData[i].w);
			values[i] = DatumGetVector(scan->keyData[0].sk_argument);
			// elog(INFO,"w: %lf",DatumGetFloat8(weights[i]));
			PrintVector("vector range query: ",values[i]);
		}
		// elog(INFO,"radius: %lf",DatumGetFloat8(scan->keyData[0].query));
	}

	// if (is_knn&&scan->orderByData->sk_flags & SK_ISNULL)
	// 	value = PointerGetDatum(InitVector(GetDimensions(scan->indexRelation)));
	// else
	// {
	// 	no need to norm.
	// 	value = scan->orderByData->sk_argument;
	// 	// PrintVector("norm vector: ", DatumGetVector(value));
	// 	/* Value should not be compressed or toasted */
	// 	Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
	// 	Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

	// 	/* Fine if normalization fails */
	// 	if (so->normprocinfo != NULL)
	// 		m3vNormValue(so->normprocinfo, so->collation, &value, NULL);
	// }

	return PointerGetDatum(values);
}

/*
 * Get dimensions from metapage
 */
int GetDimensions(Relation index) {}

/*
 * Prepare for an index scan
 */
IndexScanDesc
m3vbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	m3vScanOpaque so;

	scan = RelationGetIndexScan(index, nkeys, norderbys);
	so = (m3vScanOpaque)palloc(sizeof(m3vScanOpaqueData));
	so->returned_nums = 0;
	so->inRange = false;
	so->range_next_times = 0;
	so->result_ids = new std::vector<PQNode>();
	so->result_ids->clear();
	so->result_ids->reserve(7000);
	so->weights = new std::vector<float>();
	so->weights->resize(3,1.0);
	so->first = true;
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "M3v scan temporary context",
									   ALLOCSET_DEFAULT_SIZES);
	m3vMetaPageData meta = m3vGetMetaPageInfo(index);
	m3vMetaPage metap = &meta;
	int columns = metap->columns;
	/* Set support functions */
	so->procinfo = index_getprocinfo(index, 1, M3V_DISTANCE_PROC);
	so->normprocinfo = m3vOptionalProcInfo(index, M3V_NORM_PROC);
	so->collation = index->rd_indcollation[0];
	so->columns = columns;
	scan->opaque = so;
	// load hnsw hard index
	bool load_hnsw_from_disk = memory_init.LoadHnswHardIndex(index, memory_init.GetDimensions(index),index->rd_att->natts);
	so->load_hnsw_from_disk = load_hnsw_from_disk;
	if(index->rd_att->natts > 1){
		so->search_type = MultiColumnSearchType;
	}else{
		so->search_type = SingleSearchType;
	}
	std::string index_file_threshold_path = build_memory_index_threshold_file_path(index);
	if(!memory_init.thresholds.count(index_file_threshold_path)){
		float threshold = 0.0;
		readFloatFromFile(index_file_threshold_path,threshold);
		memory_init.thresholds[index_file_threshold_path] = threshold;
		// elog(LOG,"Read Threshold: %.6lf",threshold);
	}
	/*
	 * Get a shared lock. This allows vacuum to ensure no in-flight scans
	 * before marking tuples as deleted.
	 */
	LockPage(scan->indexRelation, M3V_SCAN_LOCK, ShareLock);
	scan->xs_orderbyvals = (Datum *) palloc0(sizeof(Datum));
    scan->xs_orderbynulls = (bool *) palloc(sizeof(bool));
	scan->xs_inorder = false;
	return scan;
}

/*
 * Insert all matching tuples into a bitmap.
 */
int64
m3vgetbitmap(IndexScanDesc scan, TIDBitmap *tbm)
{
	elog(ERROR,"doesn't support bitmap scan for a3v index now");
	// int64		ntids = 0;
	// BlockNumber blkno = BLOOM_HEAD_BLKNO,
	// 			npages;
	// int			i;
	// BufferAccessStrategy bas;
	// BloomScanOpaque so = (BloomScanOpaque) scan->opaque;

	// if (so->sign == NULL)
	// {
	// 	/* New search: have to calculate search signature */
	// 	ScanKey		skey = scan->keyData;

	// 	so->sign = palloc0(sizeof(BloomSignatureWord) * so->state.opts.bloomLength);

	// 	for (i = 0; i < scan->numberOfKeys; i++)
	// 	{
	// 		/*
	// 		 * Assume bloom-indexable operators to be strict, so nothing could
	// 		 * be found for NULL key.
	// 		 */
	// 		if (skey->sk_flags & SK_ISNULL)
	// 		{
	// 			pfree(so->sign);
	// 			so->sign = NULL;
	// 			return 0;
	// 		}

	// 		/* Add next value to the signature */
	// 		signValue(&so->state, so->sign, skey->sk_argument,
	// 				  skey->sk_attno - 1);

	// 		skey++;
	// 	}
	// }

	// /*
	//  * We're going to read the whole index. This is why we use appropriate
	//  * buffer access strategy.
	//  */
	// bas = GetAccessStrategy(BAS_BULKREAD);
	// npages = RelationGetNumberOfBlocks(scan->indexRelation);

	// for (blkno = BLOOM_HEAD_BLKNO; blkno < npages; blkno++)
	// {
	// 	Buffer		buffer;
	// 	Page		page;

	// 	buffer = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM,
	// 								blkno, RBM_NORMAL, bas);

	// 	LockBuffer(buffer, BUFFER_LOCK_SHARE);
	// 	page = BufferGetPage(buffer);
	// 	TestForOldSnapshot(scan->xs_snapshot, scan->indexRelation, page);

	// 	if (!PageIsNew(page) && !BloomPageIsDeleted(page))
	// 	{
	// 		OffsetNumber offset,
	// 					maxOffset = BloomPageGetMaxOffset(page);

	// 		for (offset = 1; offset <= maxOffset; offset++)
	// 		{
	// 			BloomTuple *itup = BloomPageGetTuple(&so->state, page, offset);
	// 			bool		res = true;

	// 			/* Check index signature with scan signature */
	// 			for (i = 0; i < so->state.opts.bloomLength; i++)
	// 			{
	// 				if ((itup->sign[i] & so->sign[i]) != so->sign[i])
	// 				{
	// 					res = false;
	// 					break;
	// 				}
	// 			}

	// 			/* Add matching tuples to bitmap */
	// 			if (res)
	// 			{
	// 				tbm_add_tuples(tbm, &itup->heapPtr, 1, true);
	// 				ntids++;
	// 			}
	// 		}
	// 	}

	// 	UnlockReleaseBuffer(buffer);
	// 	CHECK_FOR_INTERRUPTS();
	// }
	// FreeAccessStrategy(bas);

	// return ntids;
	return 0;
}

/*
 * End a scan and release resources
 */
void m3vendscan(IndexScanDesc scan)
{
	// auto begin_query = std::chrono::steady_clock::now();
	m3vScanOpaque so = (m3vScanOpaque)scan->opaque;
	// elog(LOG,"Return Tuples: %d",so->returned_nums);
	/* Release shared lock */
	UnlockPage(scan->indexRelation, M3V_SCAN_LOCK, ShareLock);
	delete so->result_ids;
	delete so->weights;
	if(so->use_hard_hnsw&&so->hard_hnsws) delete so->hard_hnsws;
	// elog(LOG,"Delete Resources %d",so->returned_nums);
	MemoryContextDelete(so->tmpCtx);
	// elog(INFO,"range times %d",so->range_next_times);
	pfree(so);
	scan->opaque = NULL;
	// auto end_query = std::chrono::steady_clock::now();
	// elog(INFO,"release time cost %.2f millseconds",std::chrono::duration<double, std::milli>(end_query - begin_query).count());
}

std::string build_hnsw_index_file_path(Relation index){
	return std::string(PROJECT_ROOT_PATH) + "/" + std::string(RelationGetRelationName(index)) + "_hnsw.bin";
}

std::string build_hnsw_index_file_hard_path(Relation index,int idx){
	return std::string(PROJECT_ROOT_PATH) + "/" + std::string(RelationGetRelationName(index)) + "_hnsw." + std::to_string(idx);
}

std::string build_hnsw_index_file_hard_path_prefix(Relation index){
	return std::string(PROJECT_ROOT_PATH) + "/" + std::string(RelationGetRelationName(index)) + "_hnsw";
}

std::string build_memory_index_points_file_path(Relation index){
	std::string path = std::string(PROJECT_ROOT_PATH) + "/" + std::string(RelationGetRelationName(index)) + "_memory_points.bin";
	return path;
}

std::string build_memory_index_threshold_file_path(Relation index){
	std::string path = std::string(PROJECT_ROOT_PATH) + "/" + std::string(RelationGetRelationName(index)) + ".threshold";
	return path;
}

// store query ids std::vector. The ids should be tids to specify the entry position.
std::string build_a3v_index_forest_query_ids_file_path(Relation index){
	return std::string(PROJECT_ROOT_PATH) + "/" + std::string(RelationGetRelationName(index)) + "_a3v_forest_root_ids.bin";
}

/*
 * Fetch the next tuple in the given scan
 * https://www.postgresql.org/docs/current/parallel-plans.html#PARALLEL-SCANS
 */
/**
 * for knn, we need two priority_queue.
 * guide_pq: small heap
 * result: big heap
*/
void do_knn_search_a3v(m3vScanOpaque scan, Relation index,m3vScanOpaque result,ItemPointerData root_tid,std::vector<float> weights,float* vectors,std::vector<int> offsets,int k,std::vector<ItemPointerData> &p){
	
}

void do_range_search_a3v(m3vScanOpaque so,Relation index,m3vScanOpaque result,ItemPointerData root_tid,std::vector<float> weights,float* vectors,std::vector<int> offsets,float radius,std::vector<ItemPointerData> &p){
	
}

// multi-vector search:
// 1. single vector search, we should drop 
// 2. multi vector search, we should do similar search (do w1.0 search with crack and then do search without crack)
// todo!: multi-bacth similar query requests.
// todo!: planner prefilter and range filter.
bool MemoryA3vIndexGetTuple(IndexScanDesc scan,ItemPointerData& result_tid){
	m3vScanOpaque so = (m3vScanOpaque)(scan->opaque);
	Relation index = scan->indexRelation;
	Relation heap_rel =  scan->heapRelation;
	Vector* query_point;
	if(so->first){
		so->first = false;
		so->result_idx = 0;
		std::string filter_text = "";
		if(scan->sourceText){
			filter_text = extract_btree_filter(scan->sourceText);
			elog(INFO,"haha vector search filter is %s",filter_text.c_str());
		}
		// if(KNN_QUERY(scan)){
		// 	so->result_ids->clear();
		// }
		so->data_points = memory_init.GetDataPointsPointer(index);
		// 1. get query point
		const std::vector<int> dimensions =  memory_init.GetDimensions(scan->indexRelation);
		std::vector<float*> query_points;
		int sums = 0,offset = 0;
		for(int i = 0;i < dimensions.size();i++){
			sums += dimensions[i];
		}
		std::vector<float> query(sums,0.0f);
		for(int i = 0;i < dimensions.size();i++){
			if(KNN_QUERY(scan)){
				(*so->weights)[i] = DatumGetFloat8(scan->orderByData[i].w);
				elog(INFO,"w: %.3lf",DatumGetFloat8(scan->orderByData[i].w));
				query_point = DatumGetVector(scan->orderByData[i].sk_argument);
			}else{
				(*so->weights)[i] = DatumGetFloat8(scan->keyData[i].w);
				query_point = DatumGetVector(scan->keyData[i].sk_argument);
			}
			query_points.push_back(query_point->x);
			// elog(INFO,"check1");
			memcpy(query.data() + offset,query_point->x,sizeof(float) * dimensions[i]);
			// elog(INFO,"check2");
			offset += dimensions[i];
			// elog(INFO,"check3");
		}
		if(dimensions.size() == 1){
			(*so->weights)[0] = 1.0;
		}
		int a3v_label;
		// get memory_index from memory init.
		std::shared_ptr<MemoryA3v> a3v_index = memory_init.GetMultiVectorMemoryIndex(index,dimensions,query.data(),a3v_label);
		std::string index_file_path = build_memory_index_points_file_path(index);
		// judge hnsw or a3v
		// multi column/single column hnsw or a3v index
		// 1. check this is the first query for now, if so we should insert query into MetaHnsw, and push the query for the 
		// a3v index. Do HardHnsw Search.
		// 2. check query records for A3V Index (in the a3v threshold), if over A3V_HINT_QUERY_RECORDS = 7, do a3v index search.
		// 3. if not within the threshold, open a new a3v index and do HardHnsw Search
		
		// hack: first search is hardhnsw, but we can do a3v search A3V_HINT_QUERY_RECORDS times for the first time, but the real
		// result uses hard hnsw.
		memory_init.query_times++;
		// elog(INFO,"btree index selectivity %.6lf",scan->btree_index_selectivity);
		// scan->btree_index_selectivity
		// we should try to check the queryRecords
		// knn search
		if(KNN_QUERY(scan)){
			elog(INFO,"a3v btree selectivity: %.2lf",scan->btree_index_selectivity);
			if(dimensions.size() == 2){
				float w1 = (*so->weights)[0];
				float w2 = (*so->weights)[1];
				// elog(LOG,"use weighted-round-robin next");
				if(w1/w2 < 0.7 || w2/w1 < 0.7){
					so->use_get_new_next = false;
				}else{
					so->use_get_new_next = true;
				}
			}else if(dimensions.size() > 1){
				so->use_get_new_next = true;
			}
			int query_records = a3v_index->query_records.load();
			if(so->use_get_new_next && query_records >= A3V_HINT_QUERY_RECORDS && (scan->btree_index_selectivity > VECTOR_FILTER_SELECTIVITY_THRESHOLD || scan->btree_index_selectivity < 1.0e-10)){
				// elog(LOG,"use a3v index");
				so->use_hard_hnsw = false;
				// 61000
				if(scan->btree_index_selectivity > 1.0e-10){
					scan->orderByData->KNNValues = std::min(MAX_K,(int)(scan->orderByData->KNNValues/scan->btree_index_selectivity*alpha_amplication));
				}
				scan->orderByData->KNNValues = max(scan->orderByData->KNNValues,a3v_top_k);
				// elog(LOG,"a3v_top_k: %d",a3v_top_k);
				std::priority_queue<PQNode> result_pqs;
				a3v_index->KnnCrackSearch(*(so->weights),query.data(),scan->orderByData->KNNValues,result_pqs,dimensions,a3v_index->last_top_k_mean * alpha_amplication);
				a3v_index->query_records.fetch_add(1);
				// elog(INFO,"result_pqs.top().first: %.2lf",result_pqs.top().first);
				// a3v_index->last_top_k_mean =  (a3v_index->last_top_k_mean * (a3v_index->query_records - (A3V_HINT_QUERY_RECORDS-1)-1) + result_pqs.top().first)/(a3v_index->query_records - (A3V_HINT_QUERY_RECORDS-1));
				// a3v_index->last_top_k_mean = max(a3v_index->last_top_k_mean,top_k_based_distance);
				// a3v index 0.75, distance: 1.100, query_records: 2, alpha_amplication: 1.47
				// elog(LOG,"a3v index %.2lf, distance: %.2lf, query_records: %d, alpha_amplication: %.2lf",a3v_index->last_top_k_mean,result_pqs.top().first,(a3v_index->query_records - 1),alpha_amplication);
				// elog(INFO,"result_pqs's size %d",so->result_ids->size());
				while(!result_pqs.empty()){
					so->result_ids->push_back(result_pqs.top());result_pqs.pop();
				}
			}else{
				// elog(LOG,"use hnsw index");
				so->use_hard_hnsw = true;
				int filter_amplication_k = 0;
				if(scan->btree_index_selectivity > 1.0e-6){
					filter_amplication_k = (int)(scan->orderByData->KNNValues / scan->btree_index_selectivity);
				}else{
					filter_amplication_k = scan->orderByData->KNNValues;
				}
				std::string hard_hnsws_prefix_path = build_hnsw_index_file_hard_path_prefix(index);
				so->hard_hnsws = new MultiColumnHnsw(memory_init.hard_hnsws[hard_hnsws_prefix_path],query_points,scan->orderByData->KNNValues,scan->xs_inorder,130,*(so->weights),filter_amplication_k,std::string(RelationGetRelationName(heap_rel)),filter_text,scan->heapRelation,scan);
				so->result_idx = 0;so->result_ids->clear();
				bool has_next = false;
				if(so->search_type == SingleSearchType){
					has_next = so->hard_hnsws->GetSingleNext();
					scan->xs_orderbyvals[0] = Float4GetDatum(so->hard_hnsws->distance);
					result_tid = so->hard_hnsws->result_tid;
				}else{
					if(so->use_get_new_next){
						has_next = so->hard_hnsws->GetNewNext();	
					}else{
						has_next = so->hard_hnsws->WeightedRoundRobin();	
					}		
					scan->xs_orderbyvals[0] = Float4GetDatum(so->hard_hnsws->distance);
					result_tid = so->hard_hnsws->result_tid;
					elog(INFO,"hnsw search finished %d",memory_init.query_times);
				}
				// yield the query point to A3VReciveServer
				if(scan->btree_index_selectivity >= VECTOR_FILTER_SELECTIVITY_THRESHOLD){
					scan->orderByData->KNNValues = std::min(MAX_K,(int)(scan->orderByData->KNNValues/scan->btree_index_selectivity));
				}
				if(hnsw_auxiilary_init && a3v_index->query_records.load() == 0 && enable_hnsw_crack_init){
					elog(INFO,"send KNN_QUERY_HNSW_INIT_MESSAGE message");
					MultiColumnHnsw* send_hard_hnsws = new MultiColumnHnsw(memory_init.hard_hnsws[hard_hnsws_prefix_path],query_points,scan->orderByData->KNNValues,scan->xs_inorder,130,*(so->weights),filter_amplication_k,std::string(RelationGetRelationName(heap_rel)),filter_text,scan->heapRelation,scan);
					auto message = std::make_shared<Message>(KNN_QUERY_HNSW_INIT_MESSAGE,a3v_label,index_file_path,std::make_shared<std::vector<float>>(query),scan->orderByData->KNNValues,0.0,std::make_shared<std::vector<int>>(dimensions),std::make_shared<std::vector<float>>(*so->weights));
					message->send_hard_hnsws = send_hard_hnsws;
					A3vAsyncSendServer(message);
				}else{
					A3vAsyncSendServer(std::make_shared<Message>(KNN_QUERY_MESSAGE,a3v_label,index_file_path,std::make_shared<std::vector<float>>(query),scan->orderByData->KNNValues,0.0,std::make_shared<std::vector<int>>(dimensions),std::make_shared<std::vector<float>>(*so->weights)));
				}
				return has_next;
			}
		}else{
			so->range_next_times++;
			if(a3v_index->query_records.load() >= A3V_HINT_QUERY_RECORDS){
				// elog(LOG,"use a3v index");
				so->use_hard_hnsw = false;
				float8 radius = DatumGetFloat8(scan->keyData[0].query);
				a3v_index->RangeCrackSearch(*(so->weights),query.data(),radius,*so->result_ids,dimensions);
				// elog(LOG,"range crack successfully");
				a3v_index->query_records.fetch_add(1);
				// after search, we need to sort for tids, this is used to improve cache hits.
				sort(so->result_ids->begin(),so->result_ids->end());
			}else{
				elog(INFO,"use hnsw index");
				so->use_hard_hnsw = true;
				std::string hard_hnsws_prefix_path = build_hnsw_index_file_hard_path_prefix(index);
				float8 radius = DatumGetFloat8(scan->keyData[0].query);
				bool has_next = false;
				A3vAsyncSendServer(std::make_shared<Message>(RANGE_QUERY_MESSAGE,a3v_label,index_file_path,std::make_shared<std::vector<float>>(query),0,radius,std::make_shared<std::vector<int>>(dimensions),std::make_shared<std::vector<float>>(*so->weights)));
				if(so->search_type == SingleSearchType){
					so->hard_hnsws = new MultiColumnHnsw(memory_init.hard_hnsws[hard_hnsws_prefix_path],query_points,multi_range_k,scan->xs_inorder,radius,*(so->weights),0,std::string(RelationGetRelationName(heap_rel)),filter_text,scan->heapRelation,scan);
					has_next = so->hard_hnsws->RangeNext();
					scan->xs_orderbyvals[0] = Float4GetDatum(so->hard_hnsws->distance);
					// elog(INFO,"distance: %.6f",so->hard_hnsws->distance);
					if(!has_next){
						return false;
					}
					result_tid = so->hard_hnsws->result_tid;
				}else{
					so->hard_hnsws = new MultiColumnHnsw(memory_init.hard_hnsws[hard_hnsws_prefix_path],query_points,0,scan->xs_inorder,radius,*(so->weights),2000,std::string(RelationGetRelationName(heap_rel)),filter_text,scan->heapRelation,scan);
					has_next = so->hard_hnsws->GetNewNext();
					scan->xs_orderbyvals[0] = Float4GetDatum(so->hard_hnsws->distance);
					if(so->hard_hnsws->distance > radius || so->range_next_times > RANGE_QUERY_THRESHOLD_TIMES){
						return false;
					}
					result_tid = so->hard_hnsws->result_tid;
				}
				// yield the query point to A3VReciveServer
				return has_next;
			}
			// put result in so->result_ids, and set result_idx as zero.
		}
	}

	if(so->use_hard_hnsw){
		if(KNN_QUERY(scan)){
			if(so->search_type == SingleSearchType){
				bool has_next = so->hard_hnsws->GetSingleNext();
				scan->xs_orderbyvals[0] = Float4GetDatum(so->hard_hnsws->distance);
				result_tid = so->hard_hnsws->result_tid;
				return has_next;
			}else{
				// elog(INFO,"terminate check -> filter_amplication_k: %d ",so->hard_hnsws->filter_amplication_k);
				if(so->returned_nums >= so->hard_hnsws->filter_amplication_k) return false;
				bool has_next = false;
				if(so->use_get_new_next){
					has_next = so->hard_hnsws->GetNewNext();	
				}else{
					has_next = so->hard_hnsws->WeightedRoundRobin();	
				}
				scan->xs_orderbyvals[0] = Float4GetDatum(so->hard_hnsws->distance);
				result_tid = so->hard_hnsws->result_tid;
				return has_next;
			}
		}else{
			if(so->search_type == SingleSearchType){
				bool has_next = so->hard_hnsws->RangeNext();
				// scan->xs_orderbyvals[0] = Float4GetDatum(so->hard_hnsws->distance);
				// so->range_next_times++;
				result_tid = so->hard_hnsws->result_tid;
				return has_next;
			}else{
				bool has_next = so->hard_hnsws->GetNewNext();
				scan->xs_orderbyvals[0] = Float4GetDatum(so->hard_hnsws->distance);
				float8 radius = DatumGetFloat8(scan->keyData[0].query);
				if(so->hard_hnsws->distance > radius || so->range_next_times > RANGE_QUERY_THRESHOLD_TIMES){
					return false;
				}
				so->range_next_times++;
				result_tid = so->hard_hnsws->result_tid;
				return has_next;
			}
		}
	}else{
		if(so->result_idx < so->result_ids->size()){
			auto item = (*so->result_ids)[so->result_ids->size() - 1 - so->result_idx++];
			int idx = item.second;
			scan->xs_orderbyvals[0] = Float4GetDatum(item.first);
			result_tid = (*so->data_points)[idx].second;
			// elog(INFO,"a3v range return results");
			return true;
		}
		return false;
	}
}

void debug_weights(IndexScanDesc scan,bool is_knn){
	if(is_knn){
		// scan->numberOfOrderBys == 1, it means it's a single metric query.
		for(int i = 0;i < scan->numberOfOrderBys; i++){
			elog(INFO,"w: %lf",DatumGetFloat8(scan->orderByData[i].w));
			PrintVector("vector knn: ",DatumGetVector(scan->orderByData[i].sk_argument));
		}
	}else{
		for(int i = 0;i < scan->numberOfKeys; i++){
			// elog(INFO,"w: %lf",DatumGetFloat8(scan->keyData[i].w));
			PrintVector("vector range query: ",DatumGetVector(scan->keyData[i].sk_argument));
		}
		// elog(INFO,"radius: %lf",DatumGetFloat8(scan->keyData[0].query));
	}
}

/**
 * IndexPages Outline like below:
 * |--------------------------|
 * |		MetaPage		  |
 * |--------------------------|
 * |	   IndexPage0		  |
 * |--------------------------|
 * |	   IndexPage1		  |
 * |--------------------------|
 * |       ..........		  |
 * |--------------------------|
 * There is a problem, should we hold all entries in one and the same IndexPage as possible as we can? 
 * It'a trade-off, we think the similar queries are successive, so the most entries from the same a3v index
 * can be in the same index tree, and that's Okay.
 * About the index entry structure is like below:
 * |--------------------------------------------------------------------------|
 * | low | high | page_id0 | page_id1 | radius | query_id | offset0 | offset1 |
 * |  4B   4B       4B         4B        4B        4B         2B        2B	  |
 * |--------------------------------------------------------------------------|
 * low: this internal entry's left index in tids array
 * high: this internal entry's right index in tids array
 * page_id0: this internal entry's left entry's page id which tells us where it's.
 * page_id1: this internal entry's left entry's page id which tells us where it's.
 * radius: this internal entry's radius
 * // I think we must need this one, because we will do page cluster, so the ItemPointer of this internal query entry will change
 * // we use query id to
 * query id: this internal entry's query id key, we use this to get the true record from record cache. 
 * offset0: this internal entry's left entry's offset in its page.
 * offset1: this internal entry's right entry's offset in its page.
 * The whole size of one entry is 28 bytes. For leaf entry, it only needs `low` and `high`. But we don't
 * distinct them separately, we keep the same the format instead.
*/

bool m3vgettuple(IndexScanDesc scan, ScanDirection dir)
{
	/*
	 * We just support KnnQuery And RangeQuery
	 */
	// elog(INFO, "RANGE_QUERY: %d", RANGE_QUERY(scan));
	// elog(INFO, "KNN_QUERY: %d", KNN_QUERY(scan));
	if (!RANGE_QUERY(scan) && !KNN_QUERY(scan) || RANGE_QUERY(scan) && KNN_QUERY(scan))
	{
		elog(ERROR, "just support Knn Query and Range Query");
	}
	// elog(INFO,"A3V GetTuple");
	// elog(INFO,"A3vIndex Name: %s",RelationGetRelationName(scan->indexRelation));
	// debug_weights(scan,KNN_QUERY(scan));
	// return false;
	m3vScanOpaque so = (m3vScanOpaque)scan->opaque;
	if(A3vMemoryIndexType(scan->indexRelation)){
		ItemPointerData result_tid;
		// auto begin_query = std::chrono::steady_clock::now();
		bool has_next = MemoryA3vIndexGetTuple(scan,result_tid);
		// auto end_query = std::chrono::steady_clock::now();
		// elog(INFO,"get tuple time cost %.2f millseconds",std::chrono::duration<double, std::milli>(end_query - begin_query).count());
		#if PG_VERSION_NUM >= 120000
		scan->xs_heaptid = result_tid;
		#else
				scan->xs_ctup.t_self = hc->data_tid;
		#endif
        scan->xs_orderbynulls[0] = false;
        scan->xs_recheckorderby = false;
		so->returned_nums++;
		return has_next;
	}
	
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);
	// if it's the first time to get a3v root.
	if(so->first){
		m3vMetaPageData metaPage =  m3vGetMetaPageInfo(scan->indexRelation);
		int dim =0;
		for(int i = 0;i < metaPage.columns;i++){
			dim += metaPage.dimentions[i]; // Dimension of the elements
		}
		so->total_len  =dim;
		so->columns = metaPage.columns;
		hnswlib::HierarchicalNSW<float>* alg_hnsw;
		if(metaPage.simliar_query_root_nums == 0){
			// it's the first time to do query operation, we should insert this one to hnsw index.
			int max_elements = 10000;   // Maximum number of elements, should be known beforehand
			int M = 16;                 // Tightly connected with internal dimensionality of the data
										// strongly affects the memory consumption
			int ef_construction = 200;  // Controls index search speed/build speed tradeoff

			// Initing index
			hnswlib::L2Space space(dim);
			alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
			std::string path = build_hnsw_index_file_path(scan->indexRelation);
			init.InsertHnswIndex(path,alg_hnsw);
			float vector[dim];
			int offset = 0;
			// we don't distinct knn query and range query, so we can use one and the same a3v tree index
			// for the similar query point whatever knn query or range query.
			for(int i = 0;i < metaPage.columns;i++){
				if(RANGE_QUERY(scan)){
					// Range Query
					Vector* range_scan_point = DatumGetVector(scan->keyData[i].sk_argument);
					memcpy(vector + offset,range_scan_point->x,sizeof(float) * metaPage.dimentions[i]);
				}else{
					// KNN Query
					Vector* range_scan_point = DatumGetVector(scan->orderByData[i].sk_argument);
					memcpy(vector + offset,range_scan_point->x,sizeof(float) * metaPage.dimentions[i]);
				}
				offset += metaPage.dimentions[i];
			}
			// add the first vector query point into hnsw;
			alg_hnsw->addPoint(vector,++metaPage.simliar_query_root_nums);
			// commit the meta page info modity.
			a3vUpdateMetaPage(scan->indexRelation,metaPage.simliar_query_root_nums,metaPage.tuple_nums,MAIN_FORKNUM);
			// trigger a3v tree build algorithm, and then we insert the root entry postion in buffer page
			// into init's tids, we will 

			// attentation: every query point data in the query a3v tree will be saved in rocksdb, and we will retrive it by record cache.
		}else{
			// so we should get hnsw index from `init`, if we can't find it out, we should load it from disk.
			// try get root a3v index from hnsw index, and after find out the root, we should give the root entry postion in buffer page
			// into m3vScanOpaque
			alg_hnsw = init.LoadHnswIndex(scan->indexRelation,dim);
		}
		so->alg_hnsw = alg_hnsw;
		so->tuple_nums = metaPage.tuple_nums;
	}
	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	// first we should get query point from the hnsw index to find the closest a3v index tree.
	if(so->first){
		int offset = 0;
		float* data = so->query_point;
		const std::vector<int>& dimensions = init.GetDimensions(scan->indexRelation);
		for(int i = 0;i < so->columns;i++){
			if(RANGE_QUERY(scan)){
				// Range Query
				Vector* range_scan_point = DatumGetVector(scan->keyData[i].sk_argument);
				(*so->weights)[i] = DatumGetFloat8(scan->keyData[i].w);
				memcpy(data + offset,range_scan_point->x,sizeof(float) * dimensions[i]);
			}else{
				// KNN Query
				Vector* knn_scan_point = DatumGetVector(scan->orderByData[i].sk_argument);
				(*so->weights)[i] = DatumGetFloat8(scan->orderByData[i].w);
				memcpy(data + offset,knn_scan_point->x,sizeof(float) * dimensions[i]);
			}
			offset += dimensions[i];
		}
		std::priority_queue<std::pair<float, hnswlib::labeltype>> result = so->alg_hnsw->searchKnn(so->query_point,1);
		auto root_point = result.top();
		hnswlib::labeltype label = root_point.second;
		// open a new root a3v index.
		if(root_point.first > A3vCloseQueryThreshold(scan->indexRelation)){
			int index_pages = RelationGetNumberOfBlocks(scan->indexRelation);
			// we need to new the first page now,and insert it into hnsw index.
			// there is only one meta page.
			if(index_pages == 1){
				InsertNewQuery(scan,so,INVALID_BLOCK_NUMBER);
			}else{
				// Get Last Page and try to insert tuple.If it's fill, we need to new a page.
				InsertNewQuery(scan,so,index_pages);
			}
		}else{
			so->root_tid = init.GetRootTidAtIndex(std::string(RelationGetRelationName(scan->indexRelation)),label);
		}
	}

	if (KNN_QUERY(scan))
	{
		if (so->first)
		{
			Datum value;

			/* Count index scan for stats */
			pgstat_count_index_scan(scan->indexRelation);

			/* Safety check */
			if (scan->orderByData == NULL)
				elog(ERROR, "cannot scan m3v index without order");

			/* Get scan value */
			float8* weights = static_cast<float8*>(palloc0(sizeof(float8) * scan->numberOfOrderBys));
			Vector** values = static_cast<Vector**>(palloc0(sizeof(Vector* )* scan->numberOfOrderBys));
			// value = GetScanValue(scan,values,weights,true,scan->numberOfOrderBys);
			// value = GetScanValue2(scan,values,weights,true,scan->numberOfOrderBys);
			// so->w = GetKNNScanItems(scan, value);
			int k = scan->orderByData->KNNValues;
			// /* Release shared lock */
			// UnlockPage(scan->indexRelation, M3V_SCAN_LOCK, ShareLock);

			so->first = false;
		}
	}
	else
	{
		if (so->first)
		{
			float8 radius = DatumGetFloat8(scan->keyData[0].query);
			float8* weights = static_cast<float8*>(palloc0(sizeof(float8) * scan->numberOfKeys));
			// maybe bug!!!!
			Vector** values = static_cast<Vector**>(palloc0(sizeof(Vector* )* scan->numberOfKeys));
			Datum q = GetScanValue(scan,values,weights,false,scan->numberOfKeys);
			// let's optmize this after meta index and vector separate idea.
			// so->w = GetRangeScanItems(scan, q, radius,so->columns);
			so->w = NULL;
			so->first = false;
		}
		// elog(ERROR, "just support KNN Query for Now!");
	}

	while (list_length(so->w) > 0)
	{
		ItemPointer tid = static_cast<ItemPointer>(llast(so->w));
		list_delete_last(so->w);
#if PG_VERSION_NUM >= 120000
		scan->xs_heaptid = *tid;
#else
		scan->xs_ctup.t_self = hc->data_tid;
#endif
		/*
		 * Typically, an index scan must maintain a pin on the index page
		 * holding the item last returned by amgettuple. However, this is not
		 * needed with the current vacuum strategy, which ensures scans do not
		 * visit tuples in danger of being marked as deleted.because here will mark
		 * deleted by index selft.
		 * https://www.postgresql.org/docs/current/index-locking.html
		 */

		scan->xs_recheckorderby = false;
		return true;
	}

	MemoryContextSwitchTo(oldCtx);
	return false;
}

/*
 * Start or restart an index scan. Init OrderBy Keys And Condition Keys
 */
void m3vrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	m3vScanOpaque so = (m3vScanOpaque)scan->opaque;

	so->first = true;
	MemoryContextReset(so->tmpCtx);

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}