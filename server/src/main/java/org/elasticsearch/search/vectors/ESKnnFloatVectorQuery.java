/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.apache.lucene.util.FixedBitSet;
import org.elasticsearch.search.profile.query.QueryProfiler;

import java.io.IOException;

public class ESKnnFloatVectorQuery extends KnnFloatVectorQuery implements QueryProfilerProvider, PostFilterableKnnQuery {
    private final int kParam;
    private long vectorOpsCount;
    private final boolean earlyTermination;
    private final boolean skipPostFilter;
    private final FixedBitSet seenDocs;

    // Captured by mergeLeafResults for query-level post-filtering
    private TopDocs capturedMergedResults;

    public ESKnnFloatVectorQuery(String field, float[] target, int k, int numCands, Query filter, KnnSearchStrategy strategy) {
        this(field, target, k, numCands, filter, strategy, false);
    }

    public ESKnnFloatVectorQuery(
        String field,
        float[] target,
        int k,
        int numCands,
        Query filter,
        KnnSearchStrategy strategy,
        boolean earlyTermination
    ) {
        this(field, target, k, numCands, filter, strategy, earlyTermination, false, null);
    }

    ESKnnFloatVectorQuery(
        String field,
        float[] target,
        int k,
        int numCands,
        Query filter,
        KnnSearchStrategy strategy,
        boolean earlyTermination,
        boolean skipPostFilter,
        FixedBitSet seenDocs
    ) {
        super(field, target, numCands, filter, strategy);
        this.kParam = k;
        this.earlyTermination = earlyTermination;
        this.skipPostFilter = skipPostFilter;
        this.seenDocs = seenDocs;
    }

    @Override
    public Query rewrite(IndexSearcher indexSearcher) throws IOException {
        if (skipPostFilter == false && filter != null) {
            BooleanQuery booleanQuery = new BooleanQuery.Builder().add(filter, BooleanClause.Occur.FILTER)
                .add(new FieldExistsQuery(field), BooleanClause.Occur.FILTER)
                .build();
            Query rewritten = indexSearcher.rewrite(booleanQuery);
            if (rewritten.getClass() != MatchNoDocsQuery.class) {
                Weight filterWeight = indexSearcher.createWeight(rewritten, ScoreMode.COMPLETE_NO_SCORES, 1f);
                float selectivity = computeSelectivity(filterWeight, indexSearcher);
                if (selectivity > 0.7f) {
                    return postFilterRewrite(indexSearcher, filterWeight, selectivity);
                }
            }
        }
        return super.rewrite(indexSearcher);
    }

    /**
     * Post-filtering rewrite: creates a filter-less delegate wrapped in PostFilterAwareKnnQuery
     * which handles filtering and retry with doc-ID exclusion.
     */
    private Query postFilterRewrite(IndexSearcher searcher, Weight filterWeight, float selectivity) throws IOException {
        int scaledNumCands = (int) Math.ceil(k / selectivity);
        ESKnnFloatVectorQuery delegate = new ESKnnFloatVectorQuery(
            field,
            getTargetCopy(),
            scaledNumCands,
            scaledNumCands,
            null,
            searchStrategy,
            earlyTermination,
            true,
            null
        );
        return new PostFilterAwareKnnQuery(delegate, filterWeight, kParam, searcher.getIndexReader(), ops -> this.vectorOpsCount = ops);
    }

    private float computeSelectivity(Weight filterWeight, IndexSearcher indexSearcher) throws IOException {
        long totalVectors = 0;
        long filterCost = 0;
        for (LeafReaderContext leafCtx : indexSearcher.getIndexReader().leaves()) {
            FloatVectorValues fvv = leafCtx.reader().getFloatVectorValues(field);
            if (fvv != null) {
                totalVectors += fvv.size();
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
        TopDocs topK = TopDocs.merge(kParam, perLeafResults);
        this.capturedMergedResults = topK;
        vectorOpsCount = topK.totalHits.value();
        return topK;
    }

    @Override
    public void profile(QueryProfiler queryProfiler) {
        queryProfiler.addVectorOpsCount(vectorOpsCount);
    }

    // --- PostFilterableKnnQuery implementation ---

    @Override
    public TopDocs capturedResults() {
        return capturedMergedResults;
    }

    @Override
    public PostFilterableKnnQuery createRetryQuery(IndexReader reader) {
        int maxDoc = reader.maxDoc();
        FixedBitSet newSeenDocs = new FixedBitSet(Math.max(maxDoc, 1));
        if (seenDocs != null) {
            newSeenDocs.or(seenDocs);
        }
        // Add all raw docs from this round to the seen set
        if (capturedMergedResults != null) {
            for (ScoreDoc sd : capturedMergedResults.scoreDocs) {
                if (sd.doc >= 0 && sd.doc < maxDoc) {
                    newSeenDocs.set(sd.doc);
                }
            }
        }
        return new ESKnnFloatVectorQuery(
            field,
            getTargetCopy(),
            kParam,
            k, // super.k = numCands from Lucene
            new PostFilterAwareKnnQuery.ExcludeDocsQuery(newSeenDocs, reader),
            searchStrategy,
            earlyTermination,
            true,
            newSeenDocs
        );
    }

    @Override
    public long vectorOpsCount() {
        return vectorOpsCount;
    }

    // --- Accessors ---

    public int kParam() {
        return kParam;
    }

    public KnnSearchStrategy getStrategy() {
        return searchStrategy;
    }

    @Override
    protected KnnCollectorManager getKnnCollectorManager(int k, IndexSearcher searcher) {
        KnnCollectorManager base = super.getKnnCollectorManager(k, searcher);
        return earlyTermination ? PatienceCollectorManager.wrap(base) : base;
    }
}
