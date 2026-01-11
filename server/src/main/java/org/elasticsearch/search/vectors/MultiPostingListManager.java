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
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;

import java.io.IOException;
import java.util.PriorityQueue;

/**
 * Manages parallel evaluation of multiple IVF posting lists.
 * This class coordinates multiple {@link PostingListCursor}s, maintaining them in a priority queue
 * ordered by centroid proximity to the query (closer centroids processed first).
 *
 * <p>The key strategy is to process posting lists in order of relevance (closest centroids first),
 * which allows for:
 * <ul>
 *   <li>Scoring the most relevant documents first (documents in closest centroids)</li>
 *   <li>Better early termination based on score thresholds</li>
 *   <li>Efficient deduplication of overspill documents using incremental deduplication filter</li>
 * </ul>
 *
 * <p>When multiple posting lists are active, they are prioritized by their load order
 * (which reflects centroid proximity from the centroid iterator). This ensures we always
 * process the most relevant centroids before moving to less relevant ones, even if those
 * less relevant centroids contain documents with lower doc IDs.
 */
public final class MultiPostingListManager {
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

    // Track vectors loaded (sum of all posting list sizes we've loaded into active cursors)
    private long loadedVectorsTotal;

    // Track vectors visited (number of vectors actually processed, even if filtered)
    private long visitedVectorsTotal;

    // Track the order in which posting lists are loaded (for prioritization)
    private int nextLoadOrder;

    /**
     * Creates a new multi-posting list manager.
     *
     * @param centroidIterator        iterator providing posting list offsets in order of centroid score
     * @param ivfVectorsReader        the IVF vectors reader for creating posting visitors
     * @param fieldInfo               the field info for the vector field
     * @param postingListInput        the input for reading posting lists
     * @param target                  the query vector
     * @param acceptDocs              filter for accepting documents (from query filters)
     * @param maxDoc                  the maximum document ID in the segment
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
     * @param centroidIterator        iterator providing posting list offsets in order of centroid score
     * @param ivfVectorsReader        the IVF vectors reader for creating posting visitors
     * @param fieldInfo               the field info for the vector field
     * @param postingListInput        the input for reading posting lists
     * @param target                  the query vector
     * @param acceptDocs              filter for accepting documents (from query filters)
     * @param maxDoc                  the maximum document ID in the segment
     * @param hasOverspillAssignments whether overspill assignments exist
     * @param maxActiveCursors        maximum number of posting lists to keep active simultaneously
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
        this.maxActiveCursors = maxActiveCursors;
        this.activeCursors = new PriorityQueue<>(maxActiveCursors);

        // Initialize deduplication filter
        this.deduplicationFilter = new IncrementalDeduplicationFilter(maxDoc);
        if (hasOverspillAssignments) {
            // We know duplicates exist, so initialize the bitset upfront
            deduplicationFilter.initialize();
        }

        // Initialize incremental filter iterator from acceptDocs
        // Try to get an iterator for efficient filtering, but fall back to bits() if iterator is not available
        DocIdSetIterator filterIterator = null;
        Bits filterBits = null;
        if (acceptDocs != null) {
            try {
                filterIterator = acceptDocs.iterator();
//                System.err.println("DEBUG[MultiPostingListManager] Created filter iterator from acceptDocs.iterator()");
            } catch (Exception e) {
                // Iterator not available or has invalid cost, fall back to bits
                filterBits = acceptDocs.bits();
//                System.err.println("DEBUG[MultiPostingListManager] Failed to create iterator (" + e.getMessage() + "), using bits fallback");
            }
        } else {
//            System.err.println("DEBUG[MultiPostingListManager] No acceptDocs filter provided");
        }
        this.filterIterator = new IncrementalFilterIterator(filterIterator, filterBits, maxDoc);
    }

    /**
     * Initializes the manager by loading the first batch of posting lists.
     *
     * @throws IOException if an I/O error occurs
     */
    public void initialize() throws IOException {
        // Load initial set of posting lists
        while (centroidIterator.hasNext() && activeCursors.size() < maxActiveCursors) {
            IVFVectorsReader.CentroidMeta centroid = centroidIterator.nextCentroidMeta();

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
                centroid.centroidOrdinal(),
                centroid.offset(),
                visitor,
                filterIterator,
                nextLoadOrder++ // Assign load order and increment
            );

            cursor.initialize();

            if (cursor.hasNext()) {
                activeCursors.offer(cursor);
                // Track total loaded vectors from this posting list
                loadedVectorsTotal += cursor.getTotalVectors();
            }
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

        // Track batch size before processing (vectors visited, not necessarily collected)
        int batchSize = minCursor.getCurrentBatchSize();

        // Score the entire batch from this cursor using post-filtering
        int scored = minCursor.scoreBatch(collector, deduplicationFilter);

        // Increment visited count by batch size (vectors processed, even if filtered)
        visitedVectorsTotal += batchSize;

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

        IVFVectorsReader.CentroidMeta centroidMeta = centroidIterator.nextCentroidMeta();

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
            centroidMeta.centroidOrdinal(),
            centroidMeta.offset(),
            visitor,
            filterIterator,
            nextLoadOrder++ // Assign load order and increment
        );

        cursor.initialize();

        if (cursor.hasNext()) {
            activeCursors.offer(cursor);
            // Track total loaded vectors from this posting list
            loadedVectorsTotal += cursor.getTotalVectors();
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
     * Returns the total vectors loaded across all posting lists that have been loaded so far.
     * This is the sum of all posting list sizes (before filtering).
     *
     * @return the total loaded vector count
     */
    public long getLoadedVectorsTotal() {
        return loadedVectorsTotal;
    }

    /**
     * Returns the total vectors visited (processed) so far.
     * This includes vectors that were scored even if they were filtered out.
     *
     * @return the total visited vector count
     */
    public long getVisitedVectorsTotal() {
        return visitedVectorsTotal;
    }

    /**
     * Drains all remaining documents from currently active cursors without loading new posting lists.
     * This ensures that all documents from posting lists that have been started are processed,
     * even if we've reached the termination condition.
     *
     * @param collector the collector to receive scored documents
     * @return the total number of documents scored during draining
     * @throws IOException if an I/O error occurs
     */
    public int drainActiveCursors(KnnCollector collector) throws IOException {
        int totalScored = 0;

        while (false == activeCursors.isEmpty()) {
            PostingListCursor minCursor = activeCursors.poll();

            // Process all remaining batches from this cursor
            while (minCursor.hasNext()) {
                int batchSize = minCursor.getCurrentBatchSize();
                int scored = minCursor.scoreBatch(collector, deduplicationFilter);
                visitedVectorsTotal += batchSize;
                totalScored += scored;
            }
        }

        return totalScored;
    }
}
