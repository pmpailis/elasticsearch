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
 * Unit of work in the parallel IVF search path: one posting list to score, plus the per-leaf
 * scorer that will execute it and the producer that handed it out (so workers can ask for the
 * next task from the same leaf when this one completes).
 *
 * <p>Ordering is descending by {@link PostingMetadata#documentCentroidScore()} so a max-heap
 * fed with these tasks dequeues the most promising centroid across all leaves first.
 */
record PostingSearchTask(LeafPostingsScorer scorer, BatchedPostingVisitor producer, PostingMetadata metadata)
    implements
        Comparable<PostingSearchTask> {

    @Override
    public int compareTo(PostingSearchTask other) {
        return Float.compare(other.metadata.documentCentroidScore(), metadata.documentCentroidScore());
    }
}
