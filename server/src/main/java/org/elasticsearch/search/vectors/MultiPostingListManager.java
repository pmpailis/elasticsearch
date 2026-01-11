/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;

import java.io.IOException;
import java.util.PriorityQueue;

/**
 * Manages parallel evaluation of multiple IVF posting lists using a multi-way merge strategy.
 * This class coordinates multiple {@link PostingListCursor}s, maintaining them in a min-heap
 * ordered by current document ID to enable efficient interleaving and deduplication.
 *
 * <p>The key idea is to process documents in global doc ID order across all posting lists,
 * which allows for:
 * <ul>
 *   <li>Efficient deduplication of overspill documents (documents in multiple centroids)</li>
 *   <li>Better cache locality by processing nearby documents together</li>
 *   <li>Early termination opportunities based on score thresholds</li>
 * </ul>
 */
public final class MultiPostingListManager {
    private static final int BULK_SIZE = 32;
    private static final int DEFAULT_MAX_ACTIVE_CURSORS = 8;

    private final PriorityQueue<PostingListCursor> activeCursors;
    private final IVFVectorsReader.CentroidIterator centroidIterator;
    private final IncrementalDeduplicationFilter deduplicationFilter;
    private final IncrementalFilterIterator filterIterator;
    private final FieldInfo fieldInfo;
    private final IndexInput postingListInput;
    private final IVFVectorsReader ivfVectorsReader;
    private final float[] target;

    private final int maxActiveCursors;
    private final int maxDoc;

    // Reusable arrays to avoid allocations
    private final int[] toScoreScratch;

    /**
     * Creates a new multi-posting list manager.
     *
     * @param centroidIterator iterator providing posting list offsets in order of centroid score
     * @param ivfVectorsReader the IVF vectors reader for creating posting visitors
     * @param fieldInfo the field info for the vector field
     * @param postingListInput the input for reading posting lists
     * @param target the query vector
     * @param acceptDocs filter for accepting documents (from query filters)
     * @param maxDoc the maximum document ID in the segment
     * @param hasOverspillAssignments whether overspill assignments exist (triggers eager dedup initialization)
     */
    public MultiPostingListManager(
        IVFVectorsReader.CentroidIterator centroidIterator,
        IVFVectorsReader ivfVectorsReader,
        FieldInfo fieldInfo,
        IndexInput postingListInput,
        float[] target,
        AcceptDocs acceptDocs,
        int maxDoc,
        boolean hasOverspillAssignments
    ) throws IOException {
        this(
            centroidIterator,
            ivfVectorsReader,
            fieldInfo,
            postingListInput,
            target,
            acceptDocs,
            maxDoc,
            hasOverspillAssignments,
            DEFAULT_MAX_ACTIVE_CURSORS
        );
    }

    /**
     * Creates a new multi-posting list manager with a custom number of active cursors.
     *
     * @param centroidIterator iterator providing posting list offsets in order of centroid score
     * @param ivfVectorsReader the IVF vectors reader for creating posting visitors
     * @param fieldInfo the field info for the vector field
     * @param postingListInput the input for reading posting lists
     * @param target the query vector
     * @param acceptDocs filter for accepting documents (from query filters)
     * @param maxDoc the maximum document ID in the segment
     * @param hasOverspillAssignments whether overspill assignments exist
     * @param maxActiveCursors maximum number of posting lists to keep active simultaneously
     */
    public MultiPostingListManager(
        IVFVectorsReader.CentroidIterator centroidIterator,
        IVFVectorsReader ivfVectorsReader,
        FieldInfo fieldInfo,
        IndexInput postingListInput,
        float[] target,
        AcceptDocs acceptDocs,
        int maxDoc,
        boolean hasOverspillAssignments,
        int maxActiveCursors
    ) throws IOException {
        this.centroidIterator = centroidIterator;
        this.ivfVectorsReader = ivfVectorsReader;
        this.fieldInfo = fieldInfo;
        this.postingListInput = postingListInput;
        this.target = target;
        this.maxDoc = maxDoc;
        this.maxActiveCursors = maxActiveCursors;
        this.activeCursors = new PriorityQueue<>(maxActiveCursors);
        this.toScoreScratch = new int[BULK_SIZE];

        // Initialize deduplication filter
        this.deduplicationFilter = new IncrementalDeduplicationFilter(maxDoc);
        if (hasOverspillAssignments) {
            // We know duplicates exist, so initialize the bitset upfront
            deduplicationFilter.initialize();
        }

        // Initialize incremental filter iterator from acceptDocs
        // Try to get an iterator for efficient filtering, but fall back to bits() if iterator is not available
        org.apache.lucene.search.DocIdSetIterator iterator = null;
        org.apache.lucene.util.Bits bits = null;
        if (acceptDocs != null) {
            try {
                iterator = acceptDocs.iterator();
            } catch (Exception e) {
                // Iterator not available or has invalid cost, fall back to bits
                bits = acceptDocs.bits();
            }
        }
        this.filterIterator = new IncrementalFilterIterator(iterator, bits, maxDoc);
    }

    /**
     * Initializes the manager by loading the first batch of posting lists.
     *
     * @throws IOException if an I/O error occurs
     */
    public void initialize() throws IOException {
        // Load initial set of posting lists
        int centroidId = 0;
        while (centroidIterator.hasNext() && activeCursors.size() < maxActiveCursors) {
            IVFVectorsReader.CentroidOffsetAndLength offsetAndLength = centroidIterator.nextPostingListOffsetAndLength();

            // Create a new cursor for this posting list
            // Note: We use a centroid score of 1.0 / (centroidId + 1) as a proxy
            // since the actual centroid score isn't readily available from the iterator
            float centroidScore = 1.0f / (centroidId + 1);

            // Create a dedicated PostingVisitor for this cursor to avoid state conflicts
            // Note: PostingVisitor is created without acceptDocs filtering since we handle that
            // at the cursor level using IncrementalFilterIterator
            IVFVectorsReader.PostingVisitor visitor = ivfVectorsReader.getPostingVisitor(
                fieldInfo,
                postingListInput,
                target,
                null // Don't apply acceptDocs in visitor - we filter at cursor level
            );

            PostingListCursor cursor = new PostingListCursor(
                centroidId,
                centroidScore,
                offsetAndLength.offset(),
                visitor,
                filterIterator
            );

            cursor.initialize();

            if (cursor.hasNext()) {
                activeCursors.offer(cursor);
            }

            centroidId++;
        }
    }

    /**
     * Checks if there are more documents to process.
     *
     * @return true if any active cursors remain
     */
    public boolean hasNext() {
        return false == activeCursors.isEmpty();
    }

    /**
     * Processes the next document by:
     * <ol>
     *   <li>Finding the cursor with the minimum doc ID</li>
     *   <li>Checking for duplicates across cursors</li>
     *   <li>Scoring and collecting if not duplicate (filter applied at cursor level)</li>
     *   <li>Advancing cursors and loading new posting lists as needed</li>
     * </ol>
     *
     * Note: This method is less efficient than processBatch() for bulk scoring.
     *
     * @param collector the collector to receive scored documents
     * @return the number of documents scored
     * @throws IOException if an I/O error occurs
     */
    public int processNext(KnnCollector collector) throws IOException {
        if (activeCursors.isEmpty()) {
            return 0;
        }

        // Get cursor with minimum doc ID
        PostingListCursor minCursor = activeCursors.poll();
        int docId = minCursor.currentDocId();

        // Note: Filtering is now handled at the cursor level via IncrementalFilterIterator
        // But deduplication across cursors still needs to be checked here

        // Check for deduplication
        if (deduplicationFilter.alreadySeen(docId)) {
            // Already scored from another posting list, skip
            minCursor.advance();
            if (minCursor.hasNext()) {
                activeCursors.offer(minCursor);
            } else {
                loadNextPostingList();
            }
            return 0;
        }

        // Mark as seen
        deduplicationFilter.markSeen(docId);

        // Score and collect this document (scoreAndCollectCurrent checks filter)
        minCursor.scoreAndCollectCurrent(collector, toScoreScratch);

        // Advance the cursor
        minCursor.advance();

        // Re-insert or replace with new posting list
        if (minCursor.hasNext()) {
            activeCursors.offer(minCursor);
        } else {
            loadNextPostingList();
        }

        return 1;
    }

    /**
     * Processes a batch of documents from the cursor with the minimum doc ID.
     * This is more efficient than processing one document at a time as it leverages
     * bulk scoring operations.
     *
     * @param collector the collector to receive scored documents
     * @return the number of documents scored
     * @throws IOException if an I/O error occurs
     */
    public int processBatch(KnnCollector collector) throws IOException {
        if (activeCursors.isEmpty()) {
            return 0;
        }

        // Get cursor with minimum doc ID
        PostingListCursor minCursor = activeCursors.poll();
        int minDocId = minCursor.currentDocId();

//        System.err.println("DEBUG[MultiPostingListManager.processBatch] Processing cursor centroid=" +
//            minCursor.centroidId() + " minDocId=" + minDocId +
//            " activeCursorsCount=" + (activeCursors.size() + 1));

        // Score the entire remaining batch from this cursor
        int scored = minCursor.scoreBatch(collector, toScoreScratch, deduplicationFilter);

//        System.err.println("DEBUG[MultiPostingListManager.processBatch] Completed cursor centroid=" +
//            minCursor.centroidId() + " scored=" + scored +
//            " hasNext=" + minCursor.hasNext() +
//            " seenCount=" + deduplicationFilter.seenCount());

        // Cursor has moved past its current batch, re-insert or replace
        if (minCursor.hasNext()) {
            activeCursors.offer(minCursor);
        } else {
            loadNextPostingList();
        }

        return scored;
    }

    /**
     * Loads the next posting list from the centroid iterator and adds it to active cursors.
     *
     * @throws IOException if an I/O error occurs
     */
    private void loadNextPostingList() throws IOException {
        if (centroidIterator.hasNext() == false) {
            return;
        }

        IVFVectorsReader.CentroidOffsetAndLength offsetAndLength = centroidIterator.nextPostingListOffsetAndLength();

        // Use a placeholder centroid ID and score
        // In a full implementation, these would come from the iterator
        int centroidId = deduplicationFilter.seenCount(); // Use seen count as a proxy for centroid ID
        float centroidScore = 1.0f / (centroidId + 1);

        // Create a dedicated PostingVisitor for this cursor to avoid state conflicts
        // Note: PostingVisitor is created without acceptDocs filtering since we handle that
        // at the cursor level using IncrementalFilterIterator
        IVFVectorsReader.PostingVisitor visitor = ivfVectorsReader.getPostingVisitor(
            fieldInfo,
            postingListInput,
            target,
            null // Don't apply acceptDocs in visitor - we filter at cursor level
        );

        PostingListCursor cursor = new PostingListCursor(
            centroidId,
            centroidScore,
            offsetAndLength.offset(),
            visitor,
            filterIterator
        );

        cursor.initialize();

        if (cursor.hasNext()) {
            activeCursors.offer(cursor);
        }
    }

    /**
     * Returns the number of documents that have been seen (and potentially deduplicated).
     *
     * @return the count of seen documents
     */
    public int seenCount() {
        return deduplicationFilter.seenCount();
    }

    /**
     * Returns the number of currently active posting list cursors.
     *
     * @return the number of active cursors
     */
    int activeCount() {
        return activeCursors.size();
    }
}
