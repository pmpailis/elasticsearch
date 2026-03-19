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
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnByteVectorQuery;
import org.apache.lucene.search.MatchNoDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.search.knn.KnnSearchStrategy;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class ESKnnByteVectorQuery extends KnnByteVectorQuery implements QueryProfilerProvider {
    private final int kParam;
    private long vectorOpsCount;
    private final boolean earlyTermination;

    // Captured by mergeLeafResults for query-level post-filtering
    private TopDocs capturedMergedResults;

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
        if (filter != null) {
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
     * Post-filtering rewrite: runs a standard KNN search (no filter, full Lucene Phase 1 + Phase 2)
     * via an inner query, then applies the filter to the merged results at the query level.
     */
    private Query postFilterRewrite(IndexSearcher searcher, Weight filterWeight, float selectivity) throws IOException {
        // Scale numCands by 1/selectivity to match ACORN's exploration behavior:
        // ACORN's queue fills only with filtered docs, so at selectivity 0.7 it takes ~1.43x more
        // visits to fill. We replicate this by inflating the search depth proportionally.
        int scaledNumCands = (int) Math.ceil(k / selectivity);
        ESKnnByteVectorQuery inner = new ESKnnByteVectorQuery(
            field,
            getTargetCopy(),
            scaledNumCands, // kParam: keep all results from scaled search for filtering
            scaledNumCands, // numCands: scaled search depth to match ACORN exploration
            null,           // no filter — post-filter after merge
            searchStrategy,
            earlyTermination
        );
        inner.rewrite(searcher);
        this.vectorOpsCount = inner.vectorOpsCount;

        TopDocs raw = inner.capturedMergedResults;
        if (raw == null || raw.scoreDocs.length == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }

        // Post-filter the merged results
        ScoreDoc[] filtered = applyFilter(raw.scoreDocs, filterWeight, searcher);
        int count = Math.min(kParam, filtered.length);
        if (count == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }
        return new KnnScoreDocQuery(Arrays.copyOf(filtered, count), searcher.getIndexReader());
    }

    /**
     * Applies the filter to ScoreDocs with global doc IDs. Groups docs by leaf for efficient
     * filter iterator advancement, then returns passing docs sorted by score descending.
     */
    private static ScoreDoc[] applyFilter(ScoreDoc[] scoreDocs, Weight filterWeight, IndexSearcher searcher) throws IOException {
        List<LeafReaderContext> leaves = searcher.getIndexReader().leaves();

        // Group docs by leaf ordinal
        @SuppressWarnings({ "unchecked", "rawtypes" })
        List<ScoreDoc>[] byLeaf = new List[leaves.size()];
        for (ScoreDoc sd : scoreDocs) {
            int leafOrd = ReaderUtil.subIndex(sd.doc, leaves);
            if (byLeaf[leafOrd] == null) {
                byLeaf[leafOrd] = new ArrayList<>();
            }
            byLeaf[leafOrd].add(sd);
        }

        List<ScoreDoc> passing = new ArrayList<>();
        for (int leafOrd = 0; leafOrd < leaves.size(); leafOrd++) {
            if (byLeaf[leafOrd] == null) continue;
            LeafReaderContext ctx = leaves.get(leafOrd);
            ScorerSupplier ss = filterWeight.scorerSupplier(ctx);
            if (ss == null) continue;

            DocIdSetIterator filterIter = ss.get(NO_MORE_DOCS).iterator();
            // Sort by local doc ID for efficient filter advancing
            List<ScoreDoc> leafDocs = byLeaf[leafOrd];
            leafDocs.sort(Comparator.comparingInt(sd -> sd.doc));

            int filterDoc = -1;
            for (ScoreDoc sd : leafDocs) {
                int localDoc = sd.doc - ctx.docBase;
                if (filterDoc < localDoc) {
                    filterDoc = filterIter.advance(localDoc);
                }
                if (filterDoc == localDoc) {
                    passing.add(sd);
                }
                if (filterDoc == NO_MORE_DOCS) break;
            }
        }

        // Sort by score descending
        passing.sort((a, b) -> Float.compare(b.score, a.score));
        return passing.toArray(new ScoreDoc[0]);
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
        TopDocs topK = TopDocs.merge(kParam, perLeafResults);
        this.capturedMergedResults = topK;
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
        KnnCollectorManager base = super.getKnnCollectorManager(k, searcher);
        return earlyTermination ? PatienceCollectorManager.wrap(base) : base;
    }
}
