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
import java.util.Random;
import java.util.concurrent.atomic.AtomicLong;

public class IVFCentroidQuery extends Query {

    /**
     * metadata about a centroid and its posting list.
     * Used by the new query architecture to identify which centroids to explore.
     */
    public record IVFCentroidMeta(long offset, long length, int ordinal, float score, IVFVectorsReader.PostingVisitor postingVisitor){ }

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
            visitor.resetPostingsScorer(0); //centroidQuery.centroidMeta.offset);

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
        ) throws IOException {
            this.boost = boost;

            this.scoringIterator = new PostingVisitorIterator(centroidMeta.ordinal, postingVisitor, totalVectorsVisited, maxVectorsToScore, centroidMeta.score);
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

        private abstract static class ScoringIterator extends DocIdSetIterator {
            abstract float scoreCurrentDoc() throws IOException;

            public abstract float maxScore(int upTo) throws IOException;
        }

        private static class PostingVisitorIterator extends ScoringIterator {
            private static final int BATCH_SIZE = 32; // Must match ESNextOSQVectorsScorer.BULK_SIZE

            private final IVFVectorsReader.PostingVisitor postingVisitor;
            private final AtomicLong totalVectorsVisited;
            private final long estimatedCost;

            private final int[] docIdsCache = new int[BATCH_SIZE];
            private final float[] scoresCache = new float[BATCH_SIZE];
            private int cacheStart = 0;
            private int cacheSize = 0;
            private int position = -1;
            private final float centroidScore;
            private int currentDoc = -1;
            private final int ordinal;
            private int targetDoc = -1;
            private boolean currentBatchScored = false;

            PostingVisitorIterator(int ordinal, IVFVectorsReader.PostingVisitor postingVisitor, AtomicLong totalVectorsVisited, long estimatedCost, float centroidScore) throws IOException {
                this.ordinal = ordinal;
                this.postingVisitor = postingVisitor;
                this.totalVectorsVisited = totalVectorsVisited;
                this.centroidScore = centroidScore;
                this.estimatedCost = postingVisitor.cost();
                nextDoc();
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
                    boolean found = false;
                    boolean shouldSkip = false;
                    if (targetDoc >= 0) {
                        for (int i = 0; i < cacheSize; i++) {
                            if (targetDoc == docIdsCache[i]) {
                                found = true;
                                break;
                            }
                        }
                    }
                    if(docIdsCache[cacheSize - 1] < targetDoc) {
                        shouldSkip = true;
                    }
                    if (targetDoc < 0 || found || (false == shouldSkip)) {
                        postingVisitor.scoreBulk(scoresCache);
                        // otherwise skip bytes

                        // Track vectors scored for this batch
                        if (totalVectorsVisited != null) {
                            totalVectorsVisited.addAndGet(cacheSize);
                        }
                    } else {
                        postingVisitor.skipBytes(cacheSize);
                    }
                }
                currentDoc = docIdsCache[position - cacheStart];
                return currentDoc;
            }

            @Override
            public int advance(int target) throws IOException {
                targetDoc = target;
                int doc = docID();
                while (doc < target) {
                    doc = nextDoc();
                }
                targetDoc = -1;
                return doc;
            }

            @Override
            public long cost() {
                return postingVisitor.cost();
            }

            @Override
            float scoreCurrentDoc() throws IOException {
                if (position < cacheStart || position >= cacheStart + cacheSize) {
                    throw new IOException("scoreCurrentDoc called outside valid range");
                }
                return scoresCache[position - cacheStart];
            }

            @Override
            public float maxScore(int upTo) {
                if (upTo == -1 || upTo == NO_MORE_DOCS) {
                    return Float.POSITIVE_INFINITY;
                }
                assert docIdsCache[position - cacheStart] == upTo;
                float maxScore = 0;
                for (int i = 0; i <= position - cacheStart; i++) {
                    if (scoresCache[i] > maxScore) {
                        maxScore = scoresCache[i];
                    }
                }
                return maxScore;
            }
        }
    }
}
