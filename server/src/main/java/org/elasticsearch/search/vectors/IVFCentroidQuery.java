/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.AbstractKnnCollector;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A Lucene query representing a single IVF centroid's posting list.
 * Scoring is done via {@link IVFVectorsReader.PostingVisitor#visit}, which provides
 * all codec-level optimizations: filtered block skipping, sparse scoring, and
 * competitive-score block pruning.
 */
public class IVFCentroidQuery extends Query {

    public record IVFCentroidMeta(PostingMetadata postingMetadata, IVFVectorsReader.PostingVisitor postingVisitor) {}

    private final String field;
    private final IVFCentroidMeta centroidMeta;
    private final LeafReaderContext context;
    private final AtomicLong totalVectorsVisited;

    public IVFCentroidQuery(String field, IVFCentroidMeta centroid, LeafReaderContext ctx, AtomicLong totalVectorsVisited) {
        this.field = field;
        this.centroidMeta = centroid;
        this.context = ctx;
        this.totalVectorsVisited = totalVectorsVisited;
    }

    public int centroidOrdinal() {
        return centroidMeta.postingMetadata().queryCentroidOrdinal();
    }

    @Override
    public String toString(String field) {
        return this.getClass().getSimpleName() + "[" + field + ", " + centroidMeta + "]";
    }

    @Override
    public void visit(QueryVisitor visitor) {
        if (visitor.acceptField(field)) {
            visitor.visitLeaf(this);
        }
    }

    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final IVFCentroidQuery other = (IVFCentroidQuery) obj;
        return field.equals(other.field) && centroidMeta.equals(other.centroidMeta);
    }

    @Override
    public int hashCode() {
        return Objects.hash(field, centroidMeta.postingMetadata().queryCentroidOrdinal());
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) {
        return new CentroidWeight(this, boost);
    }

    private static class CentroidWeight extends Weight {

        private final IVFCentroidQuery centroidQuery;
        private final float boost;

        protected CentroidWeight(IVFCentroidQuery query, float boost) {
            super(query);
            this.centroidQuery = query;
            this.boost = boost;
        }

        @Override
        public Explanation explain(LeafReaderContext context, int doc) throws IOException {
            return Explanation.match(1f, "CentroidQuery explanation not yet implemented");
        }

        @Override
        public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
            if (context.ord != centroidQuery.context.ord || context.docBase != centroidQuery.context.docBase) {
                return null;
            }
            LeafReader leafReader = context.reader();
            if (false == leafReader instanceof CodecReader) {
                throw new IllegalStateException("Expected CodecReader but got " + leafReader.getClass());
            }

            CodecReader codecReader = (CodecReader) leafReader;
            KnnVectorsReader knnVectorsReader = codecReader.getVectorReader();
            if (knnVectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
                knnVectorsReader = fieldsReader.getFieldReader(centroidQuery.field);
            }

            if (false == knnVectorsReader instanceof IVFVectorsReader) {
                throw new IllegalStateException("Expected IVFVectorsReader but got " + knnVectorsReader.getClass());
            }

            var visitor = centroidQuery.centroidMeta.postingVisitor();
            visitor.resetPostingsScorer(centroidQuery.centroidMeta.postingMetadata());

            BufferingKnnCollector collector = new BufferingKnnCollector();
            int scored = visitor.visit(collector);

            if (centroidQuery.totalVectorsVisited != null) {
                centroidQuery.totalVectorsVisited.addAndGet(scored);
            }

            if (collector.size == 0) {
                return null;
            }

            Scorer scorer = new ArrayScorer(collector.docIds, collector.scores, collector.size, boost);
            return new DefaultScorerSupplier(scorer);
        }

        @Override
        public boolean isCacheable(LeafReaderContext ctx) {
            return true;
        }
    }

    /**
     * Scorer backed by pre-scored docId/score arrays. Docs are in ascending order
     * (as produced by {@link IVFVectorsReader.PostingVisitor#visit}).
     */
    private static class ArrayScorer extends Scorer {
        private final int[] docIds;
        private final float[] scores;
        private final int count;
        private final float boost;
        private int position = -1;

        ArrayScorer(int[] docIds, float[] scores, int count, float boost) {
            this.docIds = docIds;
            this.scores = scores;
            this.count = count;
            this.boost = boost;
        }

        @Override
        public DocIdSetIterator iterator() {
            return new DocIdSetIterator() {
                @Override
                public int docID() {
                    return currentDocID();
                }

                @Override
                public int nextDoc() {
                    position++;
                    return currentDocID();
                }

                @Override
                public int advance(int target) {
                    int start = Math.max(0, position + 1);
                    int idx = Arrays.binarySearch(docIds, start, count, target);
                    position = idx >= 0 ? idx : (-idx - 1);
                    return currentDocID();
                }

                @Override
                public long cost() {
                    return count;
                }
            };
        }

        private int currentDocID() {
            return position >= 0 && position < count ? docIds[position] : (position < 0 ? -1 : DocIdSetIterator.NO_MORE_DOCS);
        }

        @Override
        public float score() {
            return scores[position] * boost;
        }

        @Override
        public float getMaxScore(int upTo) {
            return Float.POSITIVE_INFINITY;
        }

        @Override
        public int docID() {
            return currentDocID();
        }
    }

    /**
     * KnnCollector that buffers all (docId, score) pairs into growable arrays.
     * Used to capture results from {@link IVFVectorsReader.PostingVisitor#visit}.
     */
    static class BufferingKnnCollector extends AbstractKnnCollector {
        int[] docIds;
        float[] scores;
        int size;

        BufferingKnnCollector() {
            super(Integer.MAX_VALUE, Long.MAX_VALUE, null);
            docIds = new int[64];
            scores = new float[64];
            size = 0;
        }

        @Override
        public boolean collect(int docId, float similarity) {
            if (size >= docIds.length) {
                int newLen = docIds.length * 2;
                docIds = Arrays.copyOf(docIds, newLen);
                scores = Arrays.copyOf(scores, newLen);
            }
            docIds[size] = docId;
            scores[size] = similarity;
            size++;
            return true;
        }

        @Override
        public int numCollected() {
            return size;
        }

        @Override
        public float minCompetitiveSimilarity() {
            return Float.NEGATIVE_INFINITY;
        }

        @Override
        public TopDocs topDocs() {
            return new TopDocs(new TotalHits(0, TotalHits.Relation.EQUAL_TO), new org.apache.lucene.search.ScoreDoc[0]);
        }

        @Override
        public String toString() {
            return "BufferingKnnCollector[size=" + size + "]";
        }
    }
}
