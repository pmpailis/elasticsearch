/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.elasticsearch.search.profile.query.QueryProfiler;

import java.io.IOException;

public class ESKnnByteVectorQuery extends KnnByteVectorQuery implements QueryProfilerProvider {
    private final int kParam;
    private long vectorOpsCount;
    private final boolean earlyTermination;

    // Post-filtering fields (set when post-filtering is active)
    private Weight postFilterWeight;
    private int postFilterOriginalK;
    private int postFilterOversampledK;

    public ESKnnByteVectorQuery(String field, byte[] target, int k, int numCands, Query filter, KnnSearchStrategy strategy) {
        this(field, target, k, numCands, filter, strategy, false);
    }

    public ESKnnByteVectorQuery(
        String field,
        byte[] target,
        int k,
        int numCands,
        Query filter,
        KnnSearchStrategy strategy,
        boolean earlyTermination
    ) {
        super(field, target, numCands, filter, strategy);
        this.kParam = k;
        this.earlyTermination = earlyTermination;
    }

    @Override
    public Query rewrite(IndexSearcher indexSearcher) throws IOException {
        if (filter != null && postFilterWeight == null) {
            BooleanQuery booleanQuery = new BooleanQuery.Builder().add(filter, BooleanClause.Occur.FILTER)
                .add(new FieldExistsQuery(field), BooleanClause.Occur.FILTER)
                .build();
            Query rewritten = indexSearcher.rewrite(booleanQuery);
            if (rewritten.getClass() != MatchNoDocsQuery.class) {
                Weight filterWeight = indexSearcher.createWeight(rewritten, ScoreMode.COMPLETE_NO_SCORES, 1f);
                float selectivity = computeSelectivity(filterWeight, indexSearcher);
                if (selectivity > 0.7f) {
                    float overSamplingFactor = Math.max(1.2f / selectivity, 1.1f);
                    int oversampledK = (int) Math.ceil(kParam * overSamplingFactor);
                    ESKnnByteVectorQuery pfQuery = new ESKnnByteVectorQuery(
                        field,
                        getTargetCopy(),
                        kParam,
                        k,
                        null,
                        searchStrategy,
                        earlyTermination
                    );
                    pfQuery.postFilterWeight = filterWeight;
                    pfQuery.postFilterOriginalK = kParam;
                    pfQuery.postFilterOversampledK = oversampledK;
                    Query result = pfQuery.rewrite(indexSearcher);
                    this.vectorOpsCount = pfQuery.vectorOpsCount;
                    return result;
                }
            }
        }
        return super.rewrite(indexSearcher);
    }

    private float computeSelectivity(Weight filterWeight, IndexSearcher indexSearcher) throws IOException {
        long totalVectors = 0;
        long filterCost = 0;
        for (LeafReaderContext leafCtx : indexSearcher.getIndexReader().leaves()) {
            ByteVectorValues bvv = leafCtx.reader().getByteVectorValues(field);
            if (bvv != null) {
                totalVectors += bvv.size();
            }
            ScorerSupplier ss = filterWeight.scorerSupplier(leafCtx);
            if (ss != null) {
                filterCost += ss.cost();
            }
        }
        return totalVectors > 0 ? Math.min(1f, (float) filterCost / totalVectors) : 0f;
    }

    @Override
    protected TopDocs mergeLeafResults(TopDocs[] perLeafResults) {
        // if k param is set, we get only top k results from each shard
        TopDocs topK = TopDocs.merge(kParam, perLeafResults);
        vectorOpsCount = topK.totalHits.value();
        return topK;
    }

    @Override
    public void profile(QueryProfiler queryProfiler) {
        queryProfiler.addVectorOpsCount(vectorOpsCount);
    }

    public Integer kParam() {
        return kParam;
    }

    public KnnSearchStrategy getStrategy() {
        return searchStrategy;
    }

    @Override
    protected KnnCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
        if (postFilterWeight != null) {
            KnnCollectorManager mgr = new PostFilteringHnswCollectorManager(
                postFilterOversampledK,
                postFilterOriginalK,
                postFilterWeight
            );
            return earlyTermination ? PatienceCollectorManager.wrap(mgr) : mgr;
        }
        KnnCollectorManager base = super.getKnnCollectorManager(k, searcher);
        return earlyTermination ? PatienceCollectorManager.wrap(base) : base;
    }
}
