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
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.SparseFixedBitSet;

import java.io.IOException;

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
final class IncrementalFilterIterator {
    private final DocIdSetIterator iterator;
    private final Bits bits; // Fallback when iterator is not available
    private final int maxDoc;
    private SparseFixedBitSet backingBitset;
    private int currentIteratorDocId;
    private boolean exhausted;

    /**
     * Creates a new incremental filter iterator.
     *
     * @param iterator the underlying DocIdSetIterator (can be null)
     * @param bits fallback Bits for random access (used if iterator is null)
     * @param maxDoc the maximum document ID in the segment
     */
    IncrementalFilterIterator(DocIdSetIterator iterator, Bits bits, int maxDoc) throws IOException {
        this.iterator = iterator;
        this.bits = bits;
        this.maxDoc = maxDoc;
        this.exhausted = (iterator == null && bits == null);

        if (iterator != null) {
            // Initialize the iterator to the first matching doc
            this.currentIteratorDocId = iterator.nextDoc();
            if (currentIteratorDocId == DocIdSetIterator.NO_MORE_DOCS) {
                exhausted = true;
                currentIteratorDocId = Integer.MAX_VALUE;
            }
        } else {
            this.currentIteratorDocId = Integer.MAX_VALUE;
        }
    }

    /**
     * Checks if the given document ID matches the filter.
     * This method advances the iterator as needed and builds the backing bitset incrementally.
     *
     * @param docId the document ID to check
     * @return true if the document passes the filter (or no filter exists)
     * @throws IOException if an I/O error occurs
     */
    boolean matches(int docId) throws IOException {
        // No filter means all docs match
        if (iterator == null && bits == null) {
            return true;
        }

        // If we have bits but no iterator, use direct lookup (fallback mode)
        if (iterator == null && bits != null) {
            return bits.get(docId);
        }

        // Iterator-based filtering (preferred when available)
        assert iterator != null;

        // If we've already scanned past this docId, check the backing bitset
        if (docId < currentIteratorDocId) {
            return backingBitset != null && backingBitset.get(docId);
        }

        // If the iterator is exhausted, no more docs can match
        if (exhausted) {
            return false;
        }

        // If docId equals current iterator position, it matches
        if (docId == currentIteratorDocId) {
            // Mark it in backing bitset for future lookups
            ensureBackingBitset();
            backingBitset.set(docId);
            return true;
        }

        // Need to advance iterator to reach docId
        // Build backing bitset as we advance past docs
        ensureBackingBitset();

        while (currentIteratorDocId < docId && !exhausted) {
            // Mark current matching doc before advancing
            backingBitset.set(currentIteratorDocId);

            // Advance to next matching doc
            currentIteratorDocId = iterator.nextDoc();
            if (currentIteratorDocId == DocIdSetIterator.NO_MORE_DOCS) {
                exhausted = true;
                currentIteratorDocId = Integer.MAX_VALUE;
                return false;
            }
        }

        // Check if we landed exactly on docId
        if (currentIteratorDocId == docId) {
            backingBitset.set(docId);
            return true;
        }

        // We advanced past docId without finding it - it doesn't match
        return false;
    }

    /**
     * Lazily initializes the backing bitset when first needed.
     */
    private void ensureBackingBitset() {
        if (backingBitset == null) {
            backingBitset = new SparseFixedBitSet(maxDoc);
        }
    }

    /**
     * Returns the current position of the iterator (for debugging).
     * Returns -1 if using bits-based fallback.
     */
    int currentPosition() {
        if (iterator == null) {
            return -1; // Using bits fallback, no iterator position
        }
        return currentIteratorDocId;
    }

    /**
     * Returns true if the iterator is exhausted or we're using bits fallback.
     */
    boolean isExhausted() {
        return exhausted;
    }
}
