/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.action.search;

import org.apache.lucene.index.Term;
import org.apache.lucene.search.CollectionStatistics;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermStatistics;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.join.ScoreMode;
import org.apache.lucene.util.SetOnce;
import org.elasticsearch.client.internal.Client;
import org.elasticsearch.common.lucene.Lucene;
import org.elasticsearch.common.lucene.search.TopDocsAndMaxScore;
import org.elasticsearch.index.query.MatchNoneQueryBuilder;
import org.elasticsearch.index.query.NestedQueryBuilder;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.search.SearchPhaseResult;
import org.elasticsearch.search.SearchShardTarget;
import org.elasticsearch.search.builder.SearchSourceBuilder;
import org.elasticsearch.search.builder.SubSearchSourceBuilder;
import org.elasticsearch.search.dfs.AggregatedDfs;
import org.elasticsearch.search.dfs.DfsKnnRescoreInfo;
import org.elasticsearch.search.dfs.DfsKnnResults;
import org.elasticsearch.search.dfs.DfsSearchResult;
import org.elasticsearch.search.internal.ShardSearchRequest;
import org.elasticsearch.search.query.QuerySearchRequest;
import org.elasticsearch.search.query.QuerySearchResult;
import org.elasticsearch.search.vectors.KnnScoreDocQueryBuilder;
import org.elasticsearch.search.vectors.KnnSearchBuilder;
import org.elasticsearch.transport.Transport;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This search phase fans out to every shards to execute a distributed search with a pre-collected distributed frequencies for all
 * search terms used in the actual search query. This phase is very similar to the default query-then-fetch search phase, but it doesn't
 * retry on another shard if any of the shards are failing. Failures are treated as shard failures and are counted as a non-successful
 * operation.
 * @see CountedCollector#onFailure(int, SearchShardTarget, Exception)
 */
class DfsQueryPhase extends SearchPhase {

    public static final String NAME = "dfs_query";

    private final SearchPhaseResults<SearchPhaseResult> queryResult;
    private final Client client;
    private final AbstractSearchAsyncAction<?> context;
    private final SearchProgressListener progressListener;
    private long phaseStartTimeInNanos;

    DfsQueryPhase(SearchPhaseResults<SearchPhaseResult> queryResult, Client client, AbstractSearchAsyncAction<?> context) {
        super(NAME);
        this.progressListener = context.getTask().getProgressListener();
        this.queryResult = queryResult;
        this.client = client;
        this.context = context;
    }

    // protected for testing
    protected SearchPhase nextPhase(AggregatedDfs dfs) {
        return SearchQueryThenFetchAsyncAction.nextPhase(client, context, queryResult, dfs);
    }

    @SuppressWarnings("unchecked")
    @Override
    protected void run() {
        phaseStartTimeInNanos = System.nanoTime();
        List<DfsSearchResult> searchResults = (List<DfsSearchResult>) context.results.getAtomicArray().asList();
        AggregatedDfs dfs = aggregateDfs(searchResults);
        // TODO we can potentially also consume the actual per shard results from the initial phase here in the aggregateDfs
        // to free up memory early
        final CountedCollector<SearchPhaseResult> counter = new CountedCollector<>(
            queryResult,
            searchResults.size(),
            () -> onFinish(dfs),
            context
        );

        List<DfsKnnResults> knnResults = mergeKnnResults(context.getRequest(), searchResults);
        for (final DfsSearchResult dfsResult : searchResults) {
            final SearchShardTarget shardTarget = dfsResult.getSearchShardTarget();
            final int shardIndex = dfsResult.getShardIndex();
            List<DfsKnnRescoreInfo> knnRescoreInfos = buildKnnRescoreInfos(knnResults, dfsResult.getShardSearchRequest());
            ShardSearchRequest rewrittenRequest = rewriteShardSearchRequest(knnResults, dfsResult.getShardSearchRequest());
            QuerySearchRequest querySearchRequest = new QuerySearchRequest(
                context.getOriginalIndices(shardIndex),
                dfsResult.getContextId(),
                rewrittenRequest,
                dfs,
                knnRescoreInfos
            );
            final Transport.Connection connection;
            try {
                connection = context.getConnection(shardTarget.getClusterAlias(), shardTarget.getNodeId());
            } catch (Exception e) {
                shardFailure(e, querySearchRequest, shardIndex, shardTarget, counter);
                continue;
            }
            context.getSearchTransport()
                .sendExecuteQuery(connection, querySearchRequest, context.getTask(), new SearchActionListener<>(shardTarget, shardIndex) {

                    @Override
                    protected void innerOnResponse(QuerySearchResult response) {
                        try {
                            response.setSearchProfileDfsPhaseResult(dfsResult.searchProfileDfsPhaseResult());
                            counter.onResult(response);
                        } catch (Exception e) {
                            context.onPhaseFailure(NAME, "", e);
                        }
                    }

                    @Override
                    public void onFailure(Exception exception) {
                        try {
                            shardFailure(exception, querySearchRequest, shardIndex, shardTarget, counter);
                        } finally {
                            if (context.isPartOfPointInTime(querySearchRequest.contextId()) == false) {
                                // the query might not have been executed at all (for example because thread pool rejected
                                // execution) and the search context that was created in dfs phase might not be released.
                                // release it again to be in the safe side
                                context.sendReleaseSearchContext(querySearchRequest.contextId(), connection);
                            }
                        }
                    }
                });
        }
    }

    private void onFinish(AggregatedDfs dfs) {
        context.getSearchResponseMetrics()
            .recordSearchPhaseDuration(getName(), System.nanoTime() - phaseStartTimeInNanos, context.getSearchRequestAttributes());
        mergeKnnRescoreResultsIntoTopDocs();
        context.executeNextPhase(NAME, () -> nextPhase(dfs));
    }

    /**
     * Merges KNN float-precision rescore results from all shards into each shard's TopDocs.
     * For each optimized KNN search:
     *   1. Collect float-scored TopDocs from all shards
     *   2. Merge to global top-k
     *   3. Add surviving docs to each shard's TopDocs (disjunctive combination with BM25)
     */
    private void mergeKnnRescoreResultsIntoTopDocs() {
        List<SearchPhaseResult> results = queryResult.getAtomicArray().asList();
        if (results.isEmpty()) {
            return;
        }
        // Determine how many KNN rescore results each shard has
        int numKnnSearches = 0;
        for (SearchPhaseResult result : results) {
            QuerySearchResult qsr = result.queryResult();
            if (qsr != null && qsr.isNull() == false && qsr.knnRescoreResults() != null) {
                numKnnSearches = qsr.knnRescoreResults().size();
                break;
            }
        }
        if (numKnnSearches == 0) {
            return;
        }

        // Collect and merge per KNN search
        SearchRequest searchRequest = context.getRequest();
        List<DfsKnnResults> globalKnnTopK = new ArrayList<>(numKnnSearches);

        List<Integer> kValues = new ArrayList<>(numKnnSearches);
        List<Float> boosts = new ArrayList<>(numKnnSearches);
        List<String> queryNames = new ArrayList<>(numKnnSearches);
        if (searchRequest.source() != null) {
            for (KnnSearchBuilder knn : searchRequest.source().knnSearch()) {
                if (knn.isOptimizedRescoring()) {
                    kValues.add(knn.k());
                    boosts.add(knn.boost());
                    queryNames.add(knn.queryName());
                }
            }
        }

        for (int knnIdx = 0; knnIdx < numKnnSearches; knnIdx++) {
            List<TopDocs> perShardTopDocs = new ArrayList<>();
            for (SearchPhaseResult result : results) {
                QuerySearchResult qsr = result.queryResult();
                if (qsr == null || qsr.isNull() || qsr.knnRescoreResults() == null || knnIdx >= qsr.knnRescoreResults().size()) {
                    continue;
                }
                TopDocs shardTopDocs = qsr.knnRescoreResults().get(knnIdx).topDocs;
                SearchPhaseController.setShardIndex(shardTopDocs, qsr.getShardIndex());
                perShardTopDocs.add(shardTopDocs);
            }
            if (perShardTopDocs.isEmpty()) {
                continue;
            }
            int k = knnIdx < kValues.size() ? kValues.get(knnIdx) : 10;
            TopDocs merged = TopDocs.merge(k, perShardTopDocs.toArray(new TopDocs[0]));
            globalKnnTopK.add(new DfsKnnResults(null, merged.scoreDocs, null, k));
        }

        // Inject KNN top-k docs into each shard's TopDocs
        for (SearchPhaseResult result : results) {
            QuerySearchResult qsr = result.queryResult();
            if (qsr == null || qsr.isNull() || qsr.hasConsumedTopDocs()) {
                continue;
            }
            TopDocsAndMaxScore existing = qsr.topDocs();
            ScoreDoc[] bm25Docs = existing.topDocs.scoreDocs;
            int shardIndex = qsr.getShardIndex();

            // Collect all KNN docs assigned to this shard from the global top-k
            Map<Integer, Float> knnDocScores = new HashMap<>();
            for (int knnIdx = 0; knnIdx < globalKnnTopK.size(); knnIdx++) {
                float boost = knnIdx < boosts.size() ? boosts.get(knnIdx) : 1.0f;
                for (ScoreDoc sd : globalKnnTopK.get(knnIdx).scoreDocs()) {
                    if (sd.shardIndex == shardIndex) {
                        knnDocScores.merge(sd.doc, sd.score * boost, Float::sum);
                    }
                }
            }

            if (knnDocScores.isEmpty() && bm25Docs.length == 0) {
                continue;
            }

            // Disjunctive merge: combine BM25 and KNN scores
            Map<Integer, Float> combinedScores = new HashMap<>();
            for (ScoreDoc sd : bm25Docs) {
                combinedScores.put(sd.doc, sd.score);
            }
            for (var entry : knnDocScores.entrySet()) {
                combinedScores.merge(entry.getKey(), entry.getValue(), Float::sum);
            }

            ScoreDoc[] combinedDocs = new ScoreDoc[combinedScores.size()];
            int idx = 0;
            float maxScore = Float.NEGATIVE_INFINITY;
            for (var entry : combinedScores.entrySet()) {
                float score = entry.getValue();
                combinedDocs[idx++] = new ScoreDoc(entry.getKey(), score);
                maxScore = Math.max(maxScore, score);
            }
            // Sort by score descending, then by doc ascending for tie-breaking
            Arrays.sort(combinedDocs, (a, b) -> {
                int cmp = Float.compare(b.score, a.score);
                return cmp != 0 ? cmp : Integer.compare(a.doc, b.doc);
            });

            TotalHits totalHits = new TotalHits(combinedDocs.length, TotalHits.Relation.EQUAL_TO);
            TopDocs newTopDocs = new TopDocs(totalHits, combinedDocs);
            qsr.topDocs(new TopDocsAndMaxScore(newTopDocs, maxScore == Float.NEGATIVE_INFINITY ? Float.NaN : maxScore), null);
        }
    }

    private void shardFailure(
        Exception exception,
        QuerySearchRequest querySearchRequest,
        int shardIndex,
        SearchShardTarget shardTarget,
        CountedCollector<SearchPhaseResult> counter
    ) {
        context.getLogger().debug(() -> "[" + querySearchRequest.contextId() + "] Failed to execute query phase", exception);
        progressListener.notifyQueryFailure(shardIndex, shardTarget, exception);
        counter.onFailure(shardIndex, shardTarget, exception);
    }

    // package private for testing
    ShardSearchRequest rewriteShardSearchRequest(List<DfsKnnResults> knnResults, ShardSearchRequest request) {
        SearchSourceBuilder source = request.source();
        if (source == null || source.knnSearch().isEmpty()) {
            return request;
        }

        List<SubSearchSourceBuilder> subSearchSourceBuilders = new ArrayList<>(source.subSearches());

        int i = 0;
        for (DfsKnnResults dfsKnnResults : knnResults) {
            boolean optimizedRescoring = dfsKnnResults.oversample() != null;
            if (optimizedRescoring) {
                // KNN rescore will be handled as a separate side channel via DfsKnnRescoreInfo,
                // not as a sub-search in the combined query
                i++;
                continue;
            }
            List<ScoreDoc> scoreDocs = new ArrayList<>();
            for (ScoreDoc scoreDoc : dfsKnnResults.scoreDocs()) {
                if (scoreDoc.shardIndex == request.shardRequestIndex()) {
                    scoreDocs.add(scoreDoc);
                }
            }
            scoreDocs.sort(Comparator.comparingInt(scoreDoc -> scoreDoc.doc));
            String nestedPath = dfsKnnResults.getNestedPath();
            QueryBuilder query = new KnnScoreDocQueryBuilder(
                scoreDocs.toArray(Lucene.EMPTY_SCORE_DOCS),
                source.knnSearch().get(i).getField(),
                source.knnSearch().get(i).getQueryVector(),
                source.knnSearch().get(i).getSimilarity(),
                source.knnSearch().get(i).getFilterQueries(),
                dfsKnnResults.oversample(),
                dfsKnnResults.k()
            ).boost(source.knnSearch().get(i).boost()).queryName(source.knnSearch().get(i).queryName());
            if (nestedPath != null) {
                query = new NestedQueryBuilder(nestedPath, query, ScoreMode.Max).innerHit(source.knnSearch().get(i).innerHit());
            }
            subSearchSourceBuilders.add(new SubSearchSourceBuilder(query));
            i++;
        }
        // If all KNN searches are optimized and there are no other sub-searches,
        // add a match_none so the query phase produces empty BM25 TopDocs
        if (subSearchSourceBuilders.isEmpty()) {
            subSearchSourceBuilders.add(new SubSearchSourceBuilder(new MatchNoneQueryBuilder()));
        }
        source = source.shallowCopy().subSearches(subSearchSourceBuilders).knnSearch(List.of());
        request.source(source);

        return request;
    }

    /**
     * Builds per-shard KNN rescore instructions for KNN searches using optimized rescoring.
     * These are sent as a side channel alongside the query phase request.
     */
    private static List<DfsKnnRescoreInfo> buildKnnRescoreInfos(List<DfsKnnResults> knnResults, ShardSearchRequest request) {
        if (knnResults == null) {
            return List.of();
        }
        SearchSourceBuilder source = request.source();
        if (source == null || source.knnSearch().isEmpty()) {
            return List.of();
        }
        List<DfsKnnRescoreInfo> rescoreInfos = new ArrayList<>();
        for (int i = 0; i < knnResults.size(); i++) {
            DfsKnnResults dfsKnnResults = knnResults.get(i);
            if (dfsKnnResults.oversample() == null) {
                continue;
            }
            List<ScoreDoc> shardDocs = new ArrayList<>();
            for (ScoreDoc scoreDoc : dfsKnnResults.scoreDocs()) {
                if (scoreDoc.shardIndex == request.shardRequestIndex()) {
                    shardDocs.add(scoreDoc);
                }
            }
            shardDocs.sort(Comparator.comparingInt(scoreDoc -> scoreDoc.doc));
            KnnSearchBuilder knnSearch = source.knnSearch().get(i);
            // Always include entry (even if empty) to maintain consistent positional ordering across shards
            rescoreInfos.add(
                new DfsKnnRescoreInfo(
                    shardDocs.toArray(Lucene.EMPTY_SCORE_DOCS),
                    knnSearch.getField(),
                    knnSearch.getQueryVector(),
                    dfsKnnResults.k(),
                    knnSearch.boost(),
                    knnSearch.queryName()
                )
            );
        }
        return rescoreInfos;
    }

    private static List<DfsKnnResults> mergeKnnResults(SearchRequest request, List<DfsSearchResult> dfsSearchResults) {
        if (request.hasKnnSearch() == false) {
            return null;
        }
        SearchSourceBuilder source = request.source();
        List<List<TopDocs>> topDocsLists = new ArrayList<>(source.knnSearch().size());
        List<SetOnce<String>> nestedPath = new ArrayList<>(source.knnSearch().size());
        for (int i = 0; i < source.knnSearch().size(); i++) {
            topDocsLists.add(new ArrayList<>());
            nestedPath.add(new SetOnce<>());
        }
        Float[] oversampling = new Float[source.knnSearch().size()];
        Integer[] k = new Integer[source.knnSearch().size()];
        for (DfsSearchResult dfsSearchResult : dfsSearchResults) {
            if (dfsSearchResult.knnResults() != null) {
                for (int i = 0; i < dfsSearchResult.knnResults().size(); i++) {
                    DfsKnnResults knnResults = dfsSearchResult.knnResults().get(i);
                    ScoreDoc[] scoreDocs = knnResults.scoreDocs();
                    TotalHits totalHits = new TotalHits(scoreDocs.length, TotalHits.Relation.EQUAL_TO);
                    TopDocs shardTopDocs = new TopDocs(totalHits, scoreDocs);
                    SearchPhaseController.setShardIndex(shardTopDocs, dfsSearchResult.getShardIndex());
                    topDocsLists.get(i).add(shardTopDocs);
                    nestedPath.get(i).trySet(knnResults.getNestedPath());
                    // A knn search will spawn across multiple shards, and it is possible that some may be on older nodes that
                    // do not support lazy oversampling. So, we want to read from the responses whether we should allow oversampling or not.
                    // There are two options here:
                    // * either the value is null indicating an older node
                    // * or the value is consistent amongst all other nodes (it is picked up by the search request or the index settings)
                    if (oversampling[i] == null) {
                        oversampling[i] = knnResults.oversample() != null ? knnResults.oversample() : null;
                        k[i] = knnResults.k() != null ? knnResults.k() : null;
                    }
                }
            }
        }

        List<DfsKnnResults> mergedResults = new ArrayList<>(source.knnSearch().size());
        for (int i = 0; i < source.knnSearch().size(); i++) {
            int localK = k[i] != null ? k[i] : source.knnSearch().get(i).k();
            int resultsToKeep = localK;
            if (oversampling[i] != null && oversampling[i] >= 1) {
                resultsToKeep = (int) Math.ceil(oversampling[i] * localK);
            }
            TopDocs mergedTopDocs = TopDocs.merge(resultsToKeep, topDocsLists.get(i).toArray(new TopDocs[0]));
            // When no shard sent KNN results (e.g. all on older nodes), nestedPath was never set
            String path = topDocsLists.get(i).isEmpty() ? null : nestedPath.get(i).get();
            mergedResults.add(new DfsKnnResults(path, mergedTopDocs.scoreDocs, oversampling[i], localK));
        }
        return mergedResults;
    }

    private static AggregatedDfs aggregateDfs(Collection<DfsSearchResult> results) {
        Map<Term, TermStatistics> termStatistics = new HashMap<>();
        Map<String, CollectionStatistics> fieldStatistics = new HashMap<>();
        long aggMaxDoc = 0;
        for (DfsSearchResult lEntry : results) {
            final Term[] terms = lEntry.terms();
            final TermStatistics[] stats = lEntry.termStatistics();
            assert terms.length == stats.length;
            for (int i = 0; i < terms.length; i++) {
                assert terms[i] != null;
                if (stats[i] == null) {
                    continue;
                }
                TermStatistics existing = termStatistics.get(terms[i]);
                if (existing != null) {
                    assert terms[i].bytes().equals(existing.term());
                    termStatistics.put(
                        terms[i],
                        new TermStatistics(
                            existing.term(),
                            existing.docFreq() + stats[i].docFreq(),
                            existing.totalTermFreq() + stats[i].totalTermFreq()
                        )
                    );
                } else {
                    termStatistics.put(terms[i], stats[i]);
                }

            }

            assert lEntry.fieldStatistics().containsKey(null) == false;
            for (var entry : lEntry.fieldStatistics().entrySet()) {
                String key = entry.getKey();
                CollectionStatistics value = entry.getValue();
                if (value == null) {
                    continue;
                }
                assert key != null;
                CollectionStatistics existing = fieldStatistics.get(key);
                if (existing != null) {
                    CollectionStatistics merged = new CollectionStatistics(
                        key,
                        existing.maxDoc() + value.maxDoc(),
                        existing.docCount() + value.docCount(),
                        existing.sumTotalTermFreq() + value.sumTotalTermFreq(),
                        existing.sumDocFreq() + value.sumDocFreq()
                    );
                    fieldStatistics.put(key, merged);
                } else {
                    fieldStatistics.put(key, value);
                }
            }
            aggMaxDoc += lEntry.maxDoc();
        }
        return new AggregatedDfs(termStatistics, fieldStatistics, aggMaxDoc);
    }
}
