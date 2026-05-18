/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.elasticsearch.index.codec.vectors.diskbbq.BatchedPostingVisitor;
import org.elasticsearch.index.codec.vectors.diskbbq.PostingMetadata;

/**
 * Unit of work in the parallel IVF search path: an "envelope" of one or more posting lists from
 * the same leaf, plus the per-leaf scorer that will execute them and the producer that handed
 * them out. Bundling consecutive postings into a single task amortizes the global-queue lock
 * and the per-leaf {@link LeafPostingsScorer}'s lock across multiple posting lists, while still
 * letting workers preempt between envelopes for cross-leaf parallelism.
 *
 * <p>Ordering is descending by {@code bestScore} (the highest {@link PostingMetadata#documentCentroidScore()}
 * inside the envelope) so a max-heap fed with these tasks dequeues the most promising envelope
 * across all leaves first.
 */
record PostingSearchTask(LeafPostingsScorer scorer, BatchedPostingVisitor producer, PostingMetadata[] envelope, float bestScore)
    implements
        Comparable<PostingSearchTask> {

    @Override
    public int compareTo(PostingSearchTask other) {
        return Float.compare(other.bestScore, this.bestScore);
    }
}
