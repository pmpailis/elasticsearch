/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.SparseFixedBitSet;

import java.io.IOException;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * An incremental filter that advances a DocIdSetIterator on-demand and builds
 * a backing bitset as we go. This avoids materializing the full filter upfront
 * while still supporting efficient lookups for already-scanned documents.
 *
 * <p>Strategy:
 * <ul>
 *   <li>Keep a DocIdSetIterator at current position (if available)</li>
 *   <li>Build a backing bitset incrementally as we advance the iterator</li>
 *   <li>For docId &lt; iterator position: check backing bitset (already scanned)</li>
 *   <li>For docId &gt;= iterator position: advance iterator to docId, building bitset along the way</li>
 *   <li>Fallback: If iterator not available, use Bits directly (less efficient)</li>
 * </ul>
 */
public final class IncrementalFilterIterator {
    private final DocIdSetIterator iterator;
    private final SparseFixedBitSet backingBitset;

    /**
     * Creates a new incremental filter iterator.
     *
     * @param iterator the underlying DocIdSetIterator (can be null)
     * @param maxDoc   the maximum document ID in the segment
     */
    public IncrementalFilterIterator(DocIdSetIterator iterator, int maxDoc) {
        assert iterator == null || iterator.docID() == -1;
        this.backingBitset = iterator == null ? null : new SparseFixedBitSet(maxDoc);
        this.iterator = iterator;
    }

    /**
     * Checks if the given document ID matches the filter.
     * This method advances the iterator as needed and builds the backing bitset incrementally.
     *
     * @param docId the document ID to check
     * @return true if the document passes the filter (or no filter exists)
     * @throws IOException if an I/O error occurs
     */
    public boolean matches(int docId) throws IOException {
        if (iterator == null) {
            return true;
        }

        int currentPos = iterator.docID();

        if (docId == currentPos) {
            // If docId equals current iterator position, it matches
            return true;
        } else if (docId < currentPos) {
            // If we've already scanned past this docId, check the backing bitset
            return backingBitset != null && backingBitset.get(docId);
        }

        // Advance iterator until we reach or pass docId, marking matching docs as we go
        int nextDocId;
        while ((nextDocId = iterator.nextDoc()) < docId) {
            backingBitset.set(nextDocId);
        }

        if (nextDocId != NO_MORE_DOCS) {
            backingBitset.set(nextDocId);
        }

        // Check if we landed exactly on docId
        return nextDocId == docId;
    }
}
