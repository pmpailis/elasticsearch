/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BulkScorer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;

import java.io.IOException;
import java.util.Objects;

/**
 * A Lucene query that selects the highest-scoring child document for each parent block.
 * <p>
 * Children are scored using the {@code innerQuery}, and for each parent (as defined by the
 * {@code parentFilter}), the single best-scoring child is returned.
 */
public class DiversifyingParentBlockQuery extends Query {
    private final BitSetProducer parentFilter;
    private final Query innerQuery;

    public DiversifyingParentBlockQuery(BitSetProducer parentFilter, Query innerQuery) {
        this.parentFilter = Objects.requireNonNull(parentFilter);
        this.innerQuery = Objects.requireNonNull(innerQuery);
    }

    @Override
    public Query rewrite(IndexSearcher indexSearcher) throws IOException {
        Query rewritten = innerQuery.rewrite(indexSearcher);
        if (rewritten != innerQuery) {
            return new DiversifyingParentBlockQuery(parentFilter, rewritten);
        }
        return this;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
        Weight innerWeight = innerQuery.createWeight(searcher, scoreMode, boost);
        return new DiversifyingParentBlockWeight(this, innerWeight, parentFilter);
    }

    @Override
    public String toString(String field) {
        return "DiversifyingBlockQuery(inner=" + innerQuery.toString(field) + ")";
    }

    @Override
    public void visit(QueryVisitor visitor) {
        innerQuery.visit(visitor.getSubVisitor(BooleanClause.Occur.MUST, this));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        DiversifyingParentBlockQuery that = (DiversifyingParentBlockQuery) o;
        return Objects.equals(innerQuery, that.innerQuery) && parentFilter == that.parentFilter;
    }

    @Override
    public int hashCode() {
        return Objects.hash(innerQuery, parentFilter);
    }

    private static class DiversifyingParentBlockWeight extends Weight {
        private final Weight innerWeight;
        private final BitSetProducer parentFilter;

        DiversifyingParentBlockWeight(Query query, Weight innerWeight, BitSetProducer parentFilter) {
            super(query);
            this.innerWeight = innerWeight;
            this.parentFilter = parentFilter;
        }

        @Override
        public Explanation explain(LeafReaderContext context, int doc) throws IOException {
            return innerWeight.explain(context, doc);
        }

        @Override
        public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
            var innerSupplier = innerWeight.scorerSupplier(context);
            var parentBits = parentFilter.getBitSet(context);
            if (parentBits == null || innerSupplier == null) {
                return null;
            }

            return new ScorerSupplier() {
                @Override
                public Scorer get(long leadCost) throws IOException {
                    var innerScorer = innerSupplier.get(leadCost);
                    var innerIterator = innerScorer.iterator();
                    return new Scorer() {
                        int currentDoc = -1;
                        float currentScore = Float.NaN;

                        @Override
                        public int docID() {
                            return currentDoc;
                        }

                        @Override
                        public DocIdSetIterator iterator() {
                            return new DocIdSetIterator() {
                                boolean exhausted = false;

                                @Override
                                public int docID() {
                                    return currentDoc;
                                }

                                @Override
                                public int nextDoc() throws IOException {
                                    return advance(currentDoc + 1);
                                }

                                @Override
                                public int advance(int target) throws IOException {
                                    if (exhausted) {
                                        return currentDoc = NO_MORE_DOCS;
                                    }
                                    if (currentDoc == -1 || innerIterator.docID() < target) {
                                        if (innerIterator.advance(target) == NO_MORE_DOCS) {
                                            exhausted = true;
                                            return currentDoc = NO_MORE_DOCS;
                                        }
                                    }

                                    int bestChild = innerIterator.docID();
                                    float bestScore = innerScorer.score();
                                    int parent = parentBits.nextSetBit(bestChild);

                                    int innerDoc;
                                    while ((innerDoc = innerIterator.nextDoc()) < parent) {
                                        float score = innerScorer.score();
                                        if (score > bestScore) {
                                            bestChild = innerIterator.docID();
                                            bestScore = score;
                                        }
                                    }
                                    if (innerDoc == NO_MORE_DOCS) {
                                        exhausted = true;
                                    }
                                    currentScore = bestScore;
                                    return currentDoc = bestChild;
                                }

                                @Override
                                public long cost() {
                                    return innerIterator.cost();
                                }
                            };
                        }

                        @Override
                        public float score() throws IOException {
                            return currentScore;
                        }

                        @Override
                        public float getMaxScore(int upTo) throws IOException {
                            return innerScorer.getMaxScore(upTo);
                        }
                    };
                }

                @Override
                public BulkScorer bulkScorer() throws IOException {
                    BulkScorer innerBulkScorer = innerSupplier.bulkScorer();
                    if (innerBulkScorer == null) {
                        return null;
                    }
                    return new DiversifyingBulkScorer(innerBulkScorer, parentBits);
                }

                @Override
                public long cost() {
                    return innerSupplier.cost();
                }
            };
        }

        @Override
        public boolean isCacheable(LeafReaderContext ctx) {
            return false;
        }
    }

    /**
     * BulkScorer that wraps an inner BulkScorer and applies parent-block diversification.
     * This preserves the inner query's bulk scoring path (e.g. DisjunctionMaxBulkScorer),
     * which processes each sub-scorer's posting list sequentially — better I/O access patterns
     * for memory-mapped IVF posting lists than the per-doc Scorer/priority-queue merge path.
     * <p>
     * Since BulkScorer guarantees docs are collected in ascending order, we use a streaming
     * approach: a wrapping collector tracks the current parent block and emits the best-scoring
     * child when the parent boundary changes.
     */
    private static class DiversifyingBulkScorer extends BulkScorer {
        private final BulkScorer innerBulkScorer;
        private final BitSet parentBits;

        DiversifyingBulkScorer(BulkScorer innerBulkScorer, BitSet parentBits) {
            this.innerBulkScorer = innerBulkScorer;
            this.parentBits = parentBits;
        }

        @Override
        public int score(LeafCollector collector, Bits acceptDocs, int min, int max) throws IOException {
            DiversifyingLeafCollector diversifying = new DiversifyingLeafCollector(collector, parentBits);
            int next = innerBulkScorer.score(diversifying, acceptDocs, min, max);
            diversifying.flushPending();
            return next;
        }

        @Override
        public long cost() {
            return innerBulkScorer.cost();
        }
    }

    /**
     * A LeafCollector that sits between the inner BulkScorer and the outer collector,
     * performing parent-block diversification on the fly. For each parent block, only
     * the highest-scoring child document is passed to the delegate collector.
     */
    private static class DiversifyingLeafCollector implements LeafCollector {
        private final LeafCollector delegate;
        private final BitSet parentBits;
        private final ScoreHolder outerScore = new ScoreHolder();
        private Scorable innerScorable;

        private int currentParent = -1;
        private int bestChild = -1;
        private float bestScore = Float.NEGATIVE_INFINITY;

        DiversifyingLeafCollector(LeafCollector delegate, BitSet parentBits) throws IOException {
            this.delegate = delegate;
            this.parentBits = parentBits;
            delegate.setScorer(outerScore);
        }

        @Override
        public void setScorer(Scorable scorer) {
            this.innerScorable = scorer;
        }

        @Override
        public void collect(int doc) throws IOException {
            int parent = parentBits.nextSetBit(doc);
            if (parent != currentParent && bestChild != -1) {
                emitBest();
            }
            currentParent = parent;
            float score = innerScorable.score();
            if (score > bestScore) {
                bestChild = doc;
                bestScore = score;
            }
        }

        void flushPending() throws IOException {
            if (bestChild != -1) {
                emitBest();
            }
        }

        private void emitBest() throws IOException {
            outerScore.score = bestScore;
            delegate.collect(bestChild);
            bestChild = -1;
            bestScore = Float.NEGATIVE_INFINITY;
        }

        @Override
        public void finish() throws IOException {
            flushPending();
            delegate.finish();
        }
    }

    private static class ScoreHolder extends Scorable {
        float score;

        @Override
        public float score() {
            return score;
        }
    }
}
