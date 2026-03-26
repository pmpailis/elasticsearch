/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;


import com.carrotsearch.hppc.IntHashSet;

import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.function.LongConsumer;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * A query that wraps a {@link PostFilterableKnnQuery} and applies post-filtering with retry.
 * When post-filtering yields fewer than k results, retries with new delegates that avoid
 * re-visiting previously seen results (doc IDs for HNSW, centroid posting lists for IVF).
 * <p>
 * The retry loop runs internally (up to {@link #MAX_ROUNDS} rounds). Each round:
 * 1. Executes the delegate's search via rewrite
 * 2. Applies the filter to raw results
 * 3. Accumulates filtered results
 * 4. If not enough results, creates a retry delegate and continues
 */
public class PostFilterAwareKnnQuery extends Query implements QueryProfilerProvider {

    static final int MAX_ROUNDS = 5;

    private final PostFilterableKnnQuery delegate;
    private final Weight filterWeight;
    private final int k;
    private final IndexReader reader;
    private final LongConsumer vectorOpsCallback;
    private long totalVectorOps;

    public PostFilterAwareKnnQuery(
        PostFilterableKnnQuery delegate,
        Weight filterWeight,
        int k,
        IndexReader reader,
        LongConsumer vectorOpsCallback
    ) {
        this.delegate = delegate;
        this.filterWeight = filterWeight;
        this.k = k;
        this.reader = reader;
        this.vectorOpsCallback = vectorOpsCallback;
    }

    @Override
    public Query rewrite(IndexSearcher searcher) throws IOException {
        ScoreDoc[] accumulated = new ScoreDoc[0];
        long vectorOps = 0;
        PostFilterableKnnQuery current = delegate;

        for (int round = 0; round < MAX_ROUNDS; round++) {
            // Execute the delegate's search
            Query delegateQuery = (Query) current;
            delegateQuery.rewrite(searcher);

            TopDocs raw = current.capturedResults();
            vectorOps += current.vectorOpsCount();

            if (raw == null || raw.scoreDocs.length == 0) {
                continue;
            }

            // Post-filter the raw results
            ScoreDoc[] filtered = applyFilter(raw.scoreDocs, filterWeight, searcher);
            accumulated = mergeResults(accumulated, filtered);

            if (accumulated.length >= k) {
                break;
            }

            // Not enough results — retry with a delegate that avoids re-visiting
            current = current.createRetryQuery(searcher.getIndexReader());
        }

        // Propagate profiling info
        this.totalVectorOps = vectorOps;
        if (vectorOpsCallback != null) {
            vectorOpsCallback.accept(vectorOps);
        }

        if (accumulated.length == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }
        int count = Math.min(k, accumulated.length);
        return new KnnScoreDocQuery(Arrays.copyOf(accumulated, count), reader);
    }

    /**
     * Applies the filter to ScoreDocs with global doc IDs. Groups docs by leaf for efficient
     * filter iterator advancement, then returns passing docs sorted by score descending.
     */
    static ScoreDoc[] applyFilter(ScoreDoc[] scoreDocs, Weight filterWeight, IndexSearcher searcher) throws IOException {
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

    /**
     * Merges two score-descending arrays of ScoreDocs, deduplicating by doc ID.
     * Both inputs are already sorted by score descending, so we append newResults
     * after existing, skipping any doc IDs already seen.
     */
    static ScoreDoc[] mergeResults(ScoreDoc[] existing, ScoreDoc[] newResults) {
        if (existing.length == 0) return newResults;
        if (newResults.length == 0) return existing;

        IntHashSet seen = new IntHashSet(existing.length);
        List<ScoreDoc> merged = new ArrayList<>(existing.length + newResults.length);
        for (ScoreDoc sd : existing) {
            seen.add(sd.doc);
            merged.add(sd);
        }
        for (ScoreDoc sd : newResults) {
            if (seen.add(sd.doc)) {
                merged.add(sd);
            }
        }
        return merged.toArray
            (new ScoreDoc[0]);
    }

    @Override
    public void profile(QueryProfiler queryProfiler) {
        queryProfiler.addVectorOpsCount(totalVectorOps);
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {
        throw new UnsupportedOperationException("PostFilterAwareKnnQuery should always be rewritten before createWeight");
    }

    @Override
    public String toString(String field) {
        return "PostFilterAwareKnnQuery[k=" + k + ", delegate=" + delegate + "]";
    }

    @Override
    public void visit(QueryVisitor visitor) {
        ((Query) delegate).visit(visitor);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        PostFilterAwareKnnQuery that = (PostFilterAwareKnnQuery) o;
        return k == that.k && delegate.equals(that.delegate);
    }

    @Override
    public int hashCode() {
        return Objects.hash(classHash(), delegate, k);
    }

}
