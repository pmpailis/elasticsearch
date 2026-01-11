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

/**
 * A cursor for incrementally processing documents from an IVF posting list.
 * This class maintains state for reading a single posting list in batches,
 * allowing for efficient interleaving of multiple posting lists.
 *
 * <p>The cursor reads document IDs in batches using the {@link IVFVectorsReader.PostingVisitor}
 * and provides access to the minimum document ID in the current batch.
 * This enables a multi-way merge strategy across multiple posting lists.
 */
final class PostingListCursor implements Comparable<PostingListCursor> {
    private static final int BULK_SIZE = 32;

    // Identity and metadata
    private final int centroidId;
    private final float centroidScore;
    private final long postingListOffset;

    // Posting visitor for batch operations
    private final IVFVectorsReader.PostingVisitor postingVisitor;

    // Filters
    private final IncrementalFilterIterator filterIterator;

    // Current batch state
    private final int[] docIdBatch;
    private int batchSize;
    private int positionInBatch;

    // State
    private boolean initialized;
    private boolean exhausted;
    private int currentDocId;

    /**
     * Creates a new posting list cursor.
     *
     * @param centroidId the centroid ID for this posting list
     * @param centroidScore the similarity score of the centroid to the query
     * @param postingListOffset the file offset of this posting list
     * @param postingVisitor the visitor for reading and scoring documents
     * @param filterIterator incremental filter iterator for accepting documents
     */
    PostingListCursor(
        int centroidId,
        float centroidScore,
        long postingListOffset,
        IVFVectorsReader.PostingVisitor postingVisitor,
        IncrementalFilterIterator filterIterator
    ) {
        this.centroidId = centroidId;
        this.centroidScore = centroidScore;
        this.postingListOffset = postingListOffset;
        this.postingVisitor = postingVisitor;
        this.filterIterator = filterIterator;
        this.docIdBatch = new int[BULK_SIZE];
        this.initialized = false;
        this.exhausted = true;
        this.currentDocId = Integer.MAX_VALUE;
        this.batchSize = 0;
        this.positionInBatch = 0;
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

        // Reset the posting visitor to this posting list
        int totalVectors = postingVisitor.resetPostingsScorer(postingListOffset);

        if (totalVectors == 0) {
            exhausted = true;
            initialized = true;
            return;
        }

        initialized = true;
        exhausted = false;

        // Load the first batch
        loadNextBatch();
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
            positionInBatch = 0;
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
     * Returns the centroid ID for this posting list.
     *
     * @return the centroid ID
     */
    int centroidId() {
        return centroidId;
    }

    /**
     * Returns the centroid score (similarity to query).
     *
     * @return the centroid score
     */
    float centroidScore() {
        return centroidScore;
    }

    /**
     * Checks if there are more documents in this posting list.
     *
     * @return true if more documents are available
     */
    boolean hasNext() {
        return !exhausted;
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
        if (!filterIterator.matches(currentDocId)) {
            return; // Skip this doc
        }

        // Score just the current document
        toScore[0] = currentDocId;
        postingVisitor.scoreCurrentBatch(toScore, 1, collector);
    }

    /**
     * Scores all remaining documents in the current batch and collects them.
     * This is more efficient than scoring documents one at a time.
     * After scoring, automatically loads the next batch.
     *
     * @param collector the collector to receive scored documents
     * @param toScore temporary array for docs to score
     * @param deduplicationFilter filter to check for already-seen documents
     * @return the number of documents scored from this batch
     * @throws IOException if an I/O error occurs
     */
    int scoreBatch(KnnCollector collector, int[] toScore, IncrementalDeduplicationFilter deduplicationFilter) throws IOException {
        if (exhausted || positionInBatch >= batchSize) {
            return 0;
        }

        int totalInBatch = batchSize - positionInBatch;

        // Collect docs that pass BOTH acceptDocs (via filterIterator) and deduplication filters
        // CRITICAL: Check filter BEFORE marking as seen to avoid losing documents
        int count = 0;
        int skippedDueToDedup = 0;
        int skippedDueToFilter = 0;
        for (int i = positionInBatch; i < batchSize; i++) {
            int docId = docIdBatch[i];

            // Check filter first using incremental iterator
            // This advances the filter iterator as needed and builds backing bitset incrementally
            boolean passesFilter = filterIterator.matches(docId);
//            System.err.println("DEBUG[PostingListCursor] filterIterator.matches(" + docId + ") = " + passesFilter +
//                " (iteratorPos=" + filterIterator.currentPosition() + ")");

            if (!passesFilter) {
                skippedDueToFilter++;
                continue;
            }

            // Check deduplication filter
            if (deduplicationFilter.alreadySeen(docId) == false) {
                toScore[count++] = docId;
                deduplicationFilter.markSeen(docId);
            } else {
                skippedDueToDedup++;
            }
        }

//        System.err.println("DEBUG[PostingListCursor.scoreBatch] centroid=" + centroidId +
//            " totalInBatch=" + totalInBatch +
//            " passedFilter=" + (totalInBatch - skippedDueToFilter) +
//            " skippedFilter=" + skippedDueToFilter +
//            " passedDedup=" + count +
//            " skippedDedup=" + skippedDueToDedup);

        int actualScored = 0;
        if (count > 0) {
            // Score all collected docs in one operation
            actualScored = postingVisitor.scoreCurrentBatch(toScore, count, collector);
//            System.err.println("DEBUG[PostingListCursor.scoreBatch] centroid=" + centroidId +
//                " actualScored=" + actualScored + " (from " + count + " passed to visitor)");
        }

        // Move to end of batch and load the next batch immediately
        positionInBatch = batchSize;
        loadNextBatch();

        return count;
    }

    /**
     * Advances to the next document in the posting list.
     *
     * @throws IOException if an I/O error occurs
     */
    void advance() throws IOException {
        if (exhausted) {
            return;
        }

        positionInBatch++;

        // Check if we need to load the next batch
        if (positionInBatch >= batchSize) {
            loadNextBatch();
        } else {
            // Update current doc ID from batch
            currentDocId = docIdBatch[positionInBatch];
        }
    }

    @Override
    public int compareTo(PostingListCursor other) {
        // First compare by current doc ID (primary sort key)
        int cmp = Integer.compare(this.currentDocId, other.currentDocId);
        if (cmp != 0) {
            return cmp;
        }

        // For stability, break ties using centroid score (higher score first)
        return Float.compare(other.centroidScore, this.centroidScore);
    }

    /**
     * Returns true if this cursor has the same minimum doc ID as another cursor.
     * This indicates a duplicate document across posting lists (overspill).
     *
     * @param other the other cursor to compare
     * @return true if both cursors have the same current doc ID
     */
    boolean hasSameDocAs(PostingListCursor other) {
        return this.currentDocId == other.currentDocId && this.currentDocId != Integer.MAX_VALUE;
    }
}
