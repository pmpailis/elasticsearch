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
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.Bits;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

public class IVFCentroidQuery extends Query {

    /**
     * metadata about a centroid and its posting list.
     * Used by the new query architecture to identify which centroids to explore.
     */
    public record IVFCentroidMeta(long offset, long length, int ordinal, IVFVectorsReader.PostingVisitor postingVisitor){ }

    private final String field;
    private final float[] queryVector;
    private final IVFCentroidMeta centroidMeta;
    private final LeafReaderContext context;
    private final int maxVectorsPerCentroid;
    private final Bits parentBitSet;
    private final AtomicLong totalVectorsVisited;

    public IVFCentroidQuery(
        String field,
        float[] queryVector,
        IVFCentroidMeta centroid,
        LeafReaderContext ctx,
        int maxVectorsPerCentroid,
        Bits parentBitSet,
        AtomicLong totalVectorsVisited) {
        this.field = field;
        this.queryVector = queryVector;
        this.centroidMeta = centroid;
        this.maxVectorsPerCentroid = maxVectorsPerCentroid;
        this.context = ctx;
        this.parentBitSet = parentBitSet;
        this.totalVectorsVisited = totalVectorsVisited;
    }

    public int centroidOrdinal(){
        return centroidMeta.ordinal();
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
        return field.equals(other.field)
            && centroidMeta.equals(other.centroidMeta);
    }

    @Override
    public int hashCode() {
        return Objects.hash(field, centroidMeta.ordinal);
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
            // Check if this scorer is for the correct context
            // Each IVFCentroidQuery is tied to a specific leaf context
            if (context.ord != centroidQuery.context.ord || context.docBase != centroidQuery.context.docBase) {
                // This query is for a different segment
                return null;
            }
            // Access IVFVectorsReader through CodecReader
            LeafReader leafReader = context.reader();
            if (false == leafReader instanceof CodecReader) {
                throw new IllegalStateException("Expected CodecReader but got " + leafReader.getClass());
            }

            CodecReader codecReader = (CodecReader) leafReader;
            KnnVectorsReader knnVectorsReader = codecReader.getVectorReader();
            if (knnVectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
                knnVectorsReader = fieldsReader.getFieldReader(centroidQuery.field);
            }

            CentroidScorer scorer = getCentroidScorer(knnVectorsReader);
            return new DefaultScorerSupplier(scorer);
//            return new ScorerSupplier() {
//                @Override
//                public Scorer get(long leadCost) throws IOException {
//                    return scorer;
//                }
//
//                @Override
//                public long cost() {
//                    return centroidQuery.maxVectorsPerCentroid;
//                }
//            };
        }

        private CentroidScorer getCentroidScorer(KnnVectorsReader knnVectorsReader) throws IOException {
            if (false == knnVectorsReader instanceof IVFVectorsReader) {
                throw new IllegalStateException("Expected IVFVectorsReader but got " + knnVectorsReader.getClass());
            }
            var visitor = centroidQuery.centroidMeta.postingVisitor();
            visitor.resetPostingsScorer(0);

            // Create scorer
            return new CentroidScorer(
                visitor,
                centroidQuery.centroidMeta,
                boost,
                centroidQuery.maxVectorsPerCentroid,
                centroidQuery.parentBitSet,
                centroidQuery.totalVectorsVisited
            );
        }

        @Override
        public boolean isCacheable(LeafReaderContext ctx) {
            return true;
        }
    }

    /**
     * Scorer that iterates through a centroid's posting list and scores documents.
     * Enforces exploration limits and tracks competitive scores across leaves.
     * Supports parent/child diversification when parentBitSet is provided.
     */
    private static class CentroidScorer extends Scorer {
        private final float boost;
        private final ScoringIterator scoringIterator;

        CentroidScorer(
            IVFVectorsReader.PostingVisitor postingVisitor,
            IVFCentroidMeta centroidMeta,
            float boost,
            int maxVectorsToScore,
            Bits parentBitSet,
            AtomicLong totalVectorsVisited
        ) {
            this.boost = boost;

            // Create appropriate iterator based on whether we need diversification
            PostingVisitorIterator baseIterator = new PostingVisitorIterator(centroidMeta.ordinal, postingVisitor, totalVectorsVisited, maxVectorsToScore);
            if (parentBitSet != null) {
                // Wrap with diversifying iterator
                this.scoringIterator = new DiversifyingIterator(baseIterator, parentBitSet);
            } else {
                // Use limited iterator directly
                this.scoringIterator = baseIterator;
            }
        }

        @Override
        public DocIdSetIterator iterator() {
            return scoringIterator;
        }

        @Override
        public float getMaxScore(int upTo) throws IOException {
            return scoringIterator.maxScore(upTo);
        }

        @Override
        public float score() throws IOException {
            float rawScore = scoringIterator.scoreCurrentDoc();
            return rawScore * boost;
        }

        @Override
        public int docID() {
            return scoringIterator.docID();
        }

        /**
         * Common abstract class for scoring iterators.
         * Extends DocIdSetIterator (which is a class, not an interface) and adds scoring capability.
         */
        private abstract static class ScoringIterator extends DocIdSetIterator {
            abstract float scoreCurrentDoc() throws IOException;

            public abstract float maxScore(int upTo) throws IOException;
        }

        /**
         * Iterator that uses PostingVisitor's readDocIds() and scoreBulk() methods.
         * Maintains bulk scoring optimization by loading batches of 32 docs at a time.
         */
        private static class PostingVisitorIterator extends ScoringIterator {
            private static final int BATCH_SIZE = 16;

            private final IVFVectorsReader.PostingVisitor postingVisitor;
            private final AtomicLong totalVectorsVisited;
            private final long estimatedCost;

            private final int[] docIdsCache = new int[BATCH_SIZE];
            private final float[] scoresCache = new float[BATCH_SIZE];
            private int cacheStart = 0;
            private int cacheSize = 0;
            private int position = -1;
            private int currentDoc = -1;
            private final int ordinal;
            PostingVisitorIterator(int ordinal, IVFVectorsReader.PostingVisitor postingVisitor, AtomicLong totalVectorsVisited, long estimatedCost) {
                this.ordinal = ordinal;
                this.postingVisitor = postingVisitor;
                this.totalVectorsVisited = totalVectorsVisited;
                this.estimatedCost = estimatedCost;
            }

            @Override
            public int docID() {
                return currentDoc;
            }

            @Override
            public int nextDoc() throws IOException {
                position++;

                // Check if we need to load next batch
                if (position >= cacheStart + cacheSize) {
                    // Load next batch
                    cacheStart = position;
                    cacheSize = postingVisitor.readDocIds(BATCH_SIZE, docIdsCache);
                    if (cacheSize == 0) {
                        currentDoc = NO_MORE_DOCS;
                        return NO_MORE_DOCS;
                    }
                    // Score the batch
                    postingVisitor.scoreBulk(scoresCache);

                    // Track vectors visited for this batch
                    if (totalVectorsVisited != null) {
                        totalVectorsVisited.addAndGet(cacheSize);
                    }
                }
                currentDoc = docIdsCache[position - cacheStart];
                return currentDoc;
            }

            @Override
            public int advance(int target) throws IOException {
                int doc = docID();
                while (doc < target) {
                    doc = nextDoc();
                }
                return doc;
            }

            @Override
            public long cost() {
                return estimatedCost;
            }

            @Override
            float scoreCurrentDoc() throws IOException {
                if (position < cacheStart || position >= cacheStart + cacheSize) {
                    throw new IOException("scoreCurrentDoc called outside valid range");
                }
//                LogManager.getLogger("xoxo").error("scoring ordinal: {} at doc: " + "{}", ordinal, currentDoc);
                return scoresCache[position - cacheStart];
            }

            @Override
            public float maxScore(int upTo){
                if(upTo == -1 || upTo == NO_MORE_DOCS){
                    return Float.POSITIVE_INFINITY;
                }
                assert docIdsCache[position - cacheStart] == upTo;
                float maxScore = 0;
                for(int i=0;i<=position - cacheStart;i++){
                    if(scoresCache[i] > maxScore){
                        maxScore = scoresCache[i];
                    }
                }
                return maxScore;
            }
        }

        /**
         * Diversifying iterator that ensures only the best child per parent is returned.
         * Wraps a ScoringIterator and tracks the best score for each parent.
         */
        private static class DiversifyingIterator extends ScoringIterator {
            private final ScoringIterator delegate;
            private final Bits parentBitSet;

            // Track best child per parent
            private final java.util.HashMap<Integer, ChildScore> parentToChild = new java.util.HashMap<>();

            // Current state
            private int currentDoc = -1;
            private float currentScore;

            DiversifyingIterator(ScoringIterator delegate, Bits parentBitSet) {
                this.delegate = delegate;
                this.parentBitSet = parentBitSet;
            }
            @Override
            public float maxScore(int upTo) throws IOException {
                return delegate.maxScore(upTo);
            }

            @Override
            public int docID() {
                return currentDoc;
            }

            @Override
            public int nextDoc() throws IOException {
                while (true) {
                    int childDoc = delegate.nextDoc();
                    if (childDoc == NO_MORE_DOCS) {
                        currentDoc = NO_MORE_DOCS;
                        return NO_MORE_DOCS;
                    }

                    // Score this child
                    float score = delegate.scoreCurrentDoc();

                    // Find parent for this child
                    int parent = findParent(childDoc);

                    // Check if this is the best child for this parent
                    ChildScore existing = parentToChild.get(parent);
                    if (existing == null || score > existing.score) {
                        // This is a new or better child for this parent
                        parentToChild.put(parent, new ChildScore(childDoc, score));
                        currentDoc = childDoc;
                        currentScore = score;
                        return childDoc;
                    }
                    // Otherwise, skip this child and continue to next
                }
            }

            @Override
            public int advance(int target) throws IOException {
                int doc = docID();
                while (doc < target) {
                    doc = nextDoc();
                }
                return doc;
            }

            @Override
            public long cost() {
                return delegate.cost();
            }

            @Override
            public float scoreCurrentDoc() throws IOException {
                return currentScore;
            }

            /**
             * Finds the parent document ID for a given child document.
             * Uses the parentBitSet to identify which documents are parents.
             */
            private int findParent(int childDoc) {
                // Find the next set bit at or after childDoc
                // This gives us the parent for this child
                for (int doc = childDoc; doc >= 0; doc--) {
                    if (parentBitSet.get(doc)) {
                        return doc;
                    }
                }
                // If no parent found, child is its own parent
                return childDoc;
            }

            private static class ChildScore {
                final int childDoc;
                final float score;

                ChildScore(int childDoc, float score) {
                    this.childDoc = childDoc;
                    this.score = score;
                }
            }
        }
    }
}
