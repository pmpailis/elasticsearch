/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.util.SparseFixedBitSet;

/**
 * An incremental filter for tracking already-seen documents during vector search.
 * This filter lazily initializes its internal bitset only when duplicates are expected
 * or detected, avoiding memory allocation when not needed.
 *
 * <p>This is particularly useful for IVF vector search where posting lists may contain
 * duplicate documents due to overspill assignments (documents assigned to multiple centroids).
 */
final class IncrementalDeduplicationFilter {
    private final int maxDoc;
    private SparseFixedBitSet seenDocs;
    private boolean initialized;

    /**
     * Creates a new incremental deduplication filter.
     *
     * @param maxDoc the maximum document ID in the segment
     */
    IncrementalDeduplicationFilter(int maxDoc) {
        this.maxDoc = maxDoc;
        this.initialized = false;
        this.seenDocs = null;
    }

    /**
     * Explicitly initializes the internal bitset. This can be called upfront when
     * duplicates are known to exist (e.g., when overspill assignments are present).
     */
    void initialize() {
        if (initialized == false) {
            seenDocs = new SparseFixedBitSet(maxDoc);
            initialized = true;
        }
    }

    /**
     * Marks a document as seen. If the bitset has not been initialized yet,
     * this will trigger lazy initialization.
     *
     * @param docId the document ID to mark as seen
     */
    void markSeen(int docId) {
        if (initialized == false) {
            initialize();
        }
        seenDocs.set(docId);
    }

    /**
     * Checks if a document has already been seen.
     *
     * @param docId the document ID to check
     * @return true if the document was previously marked as seen, false otherwise
     */
    boolean alreadySeen(int docId) {
        return initialized && seenDocs.get(docId);
    }

    /**
     * Returns the number of documents marked as seen.
     *
     * @return the count of seen documents, or 0 if not initialized
     */
    int seenCount() {
        return initialized ? seenDocs.cardinality() : 0;
    }

    /**
     * Returns whether the filter has been initialized.
     *
     * @return true if the internal bitset has been allocated
     */
    boolean isInitialized() {
        return initialized;
    }
}
