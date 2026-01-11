/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.search.KnnCollector;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;

import java.io.IOException;
import java.util.Arrays;
import java.util.Objects;

/**
 * A cursor for incrementally processing documents from an IVF posting list.
 * This class maintains state for reading a single posting list in batches,
 * allowing for efficient interleaving of multiple posting lists.
 *
 * <p>The cursor reads document IDs in batches using the {@link IVFVectorsReader.PostingVisitor}.
 * Cursors are ordered by their load order (which reflects centroid proximity to the query),
 * ensuring that posting lists from centroids closest to the query are processed first.
 * This prioritization is critical for scoring the most relevant documents early and
 * enabling efficient early termination.
 */
final class PostingListCursor implements Comparable<PostingListCursor> {
    private static final int BULK_SIZE = 16;

    private final long postingListOffset;

    // Posting visitor for batch operations
    private final IVFVectorsReader.PostingVisitor postingVisitor;

    // Filters
    private final IncrementalFilterIterator filterIterator;

    // Current batch state
    private final int[] docIdBatch;
    private int batchSize;

    // State
    private boolean initialized;
    private boolean exhausted;
    private int currentDocId;
    private final int centroidOrdinal;
    private int totalVectors; // Total number of vectors in this posting list
    private final int loadOrder; // Order in which this cursor was loaded (lower = closer to query)

    /**
     * Creates a new posting list cursor.
     *
     * @param centroidOrdinal the centroid ordinal for this posting list
     * @param postingListOffset the file offset of this posting list
     * @param postingVisitor the visitor for reading and scoring documents
     * @param filterIterator incremental filter iterator for accepting documents
     * @param loadOrder the order in which this cursor was loaded (for prioritization)
     */
    PostingListCursor(
        int centroidOrdinal,
        long postingListOffset,
        IVFVectorsReader.PostingVisitor postingVisitor,
        IncrementalFilterIterator filterIterator,
        int loadOrder
    ) {
        this.centroidOrdinal = centroidOrdinal;
        this.loadOrder = loadOrder;
        this.postingListOffset = postingListOffset;
        this.postingVisitor = postingVisitor;
        this.filterIterator = filterIterator;
        this.docIdBatch = new int[BULK_SIZE];
        this.initialized = false;
        this.exhausted = true;
        this.currentDocId = Integer.MAX_VALUE;
        this.batchSize = 0;
    }

    /**
     * Initializes this cursor by reading the posting list header and loading the first batch.
     *
     * @throws IOException if an I/O error occurs
     */
    void initialize() throws IOException {
        if (initialized) {
            return;
        }

        // Reset the posting visitor to this posting list and get total vector count
        this.totalVectors = postingVisitor.resetPostingsScorer(postingListOffset);

        if (this.totalVectors == 0) {
            exhausted = true;
            initialized = true;
            return;
        }else{
            exhausted = false;
        }
        // Load the first batch
        loadNextBatch();
        initialized = true;
    }

    /**
     * Loads the next batch of document IDs from the posting list.
     *
     * @throws IOException if an I/O error occurs
     */
    private void loadNextBatch() throws IOException {
        if (postingVisitor.hasNextBatch() == false) {
            exhausted = true;
            currentDocId = Integer.MAX_VALUE;
            batchSize = 0;
            return;
        }

        // Read the next batch of doc IDs
        batchSize = postingVisitor.readNextBatch(docIdBatch);
        if (batchSize == 0) {
            exhausted = true;
            currentDocId = Integer.MAX_VALUE;
        } else {
            currentDocId = docIdBatch[0];
        }
    }

    /**
     * Returns the current minimum document ID in this cursor.
     *
     * @return the current document ID, or Integer.MAX_VALUE if exhausted
     */
    int currentDocId() {
        return currentDocId;
    }


    /**
     * Checks if there are more documents in this posting list.
     *
     * @return true if more documents are available
     */
    boolean hasNext() {
        return false == exhausted;
    }

    /**
     * Returns the size of the current batch (number of documents loaded).
     *
     * @return the current batch size
     */
    int getCurrentBatchSize() {
        return batchSize;
    }

    /**
     * Returns the total number of vectors in this posting list.
     *
     * @return the total vector count
     */
    int getTotalVectors() {
        return totalVectors;
    }

    /**
     * Scores and collects the current document.
     * Note: This method applies the filter check using IncrementalFilterIterator.
     *
     * @param collector the collector to receive the scored document
     * @param toScore temporary array for docs to score (reused for efficiency)
     * @throws IOException if an I/O error occurs
     */
    void scoreAndCollectCurrent(KnnCollector collector, int[] toScore) throws IOException {
        if (exhausted) {
            return;
        }

        // Check if current doc passes the filter
        if (false == filterIterator.matches(currentDocId)) {
            return; // Skip this doc
        }

        // Score just the current document
        toScore[0] = currentDocId;
        postingVisitor.scoreCurrentBatch(toScore, 1, collector);
    }

    /**
     * Scores all documents in the current batch using POST-FILTERING approach.
     *
     * <p>Strategy for maximum bulk scoring efficiency:
     * <ol>
     *   <li>Score ALL docs in batch (16 docs) using bulk operations</li>
     *   <li>Apply filter + deduplication to scored results</li>
     *   <li>Collect only docs that pass both</li>
     * </ol>
     *
     * This maintains full bulk scoring efficiency (always score 16 docs at once)
     * rather than pre-filtering which would break batches into smaller pieces.
     *
     * @param collector the collector to receive scored documents
     * @param deduplicationFilter filter to check for already-seen documents
     * @return the number of documents actually scored (before filtering)
     * @throws IOException if an I/O error occurs
     */
    int scoreBatch(KnnCollector collector, IncrementalDeduplicationFilter deduplicationFilter) throws IOException {
        if (exhausted) {
            return 0;
        }

        if (batchSize == 0) {
            loadNextBatch();
            return 0;
        }

        // Score ALL docs in batch for full bulk efficiency
        // The visitor will handle post-filtering internally
        int scoredCount = postingVisitor.scoreBulkWithPostFiltering(
            docIdBatch,
            batchSize,
            filterIterator,
            deduplicationFilter,
            collector
        );

        // Load next batch for future calls
        loadNextBatch();

        return scoredCount;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        PostingListCursor that = (PostingListCursor) o;
        return batchSize == that.batchSize &&
            exhausted == that.exhausted &&
            currentDocId == that.currentDocId &&
            centroidOrdinal == that.centroidOrdinal &&
            Arrays.equals(docIdBatch, that.docIdBatch);
    }

    @Override
    public int hashCode() {
        return Objects.hash(batchSize, exhausted, currentDocId, centroidOrdinal, Arrays.hashCode(docIdBatch));
    }

    @Override
    public int compareTo(PostingListCursor other) {
        // Sort by load order first (lower = closer to query = higher priority)
        int cmp = Integer.compare(this.loadOrder, other.loadOrder);
        if (cmp != 0) {
            return cmp;
        }
        // Tie-breaker: sort by doc ID for deduplication efficiency
        return Integer.compare(this.currentDocId, other.currentDocId);
    }
}
