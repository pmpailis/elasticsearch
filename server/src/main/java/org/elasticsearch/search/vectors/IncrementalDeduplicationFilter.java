/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import com.carrotsearch.hppc.IntHashSet;

/**
 * An incremental filter for tracking already-seen documents during vector search.
 * This filter lazily initializes its internal bitset only when duplicates are expected
 * or detected, avoiding memory allocation when not needed.
 *
 * <p>This is particularly useful for IVF vector search where posting lists may contain
 * duplicate documents due to overspill assignments (documents assigned to multiple centroids).
 */
public final class IncrementalDeduplicationFilter {

    private final IntHashSet seenDocs;

    public IncrementalDeduplicationFilter(int expectedSize) {
        this.seenDocs = new IntHashSet(expectedSize);
    }

    /**
     * Checks if a document has already been seen.
     *
     * @param docId the document ID to check
     * @return true if the document was previously marked as seen, false otherwise
     */
    public boolean add(int docId) {
        return seenDocs.add(docId);
    }
}
