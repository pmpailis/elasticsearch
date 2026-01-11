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
public final class IncrementalFilterIterator {
    private final DocIdSetIterator iterator;
    private final Bits bits; // Fallback when iterator is not available
    private final int maxDoc;
    private SparseFixedBitSet backingBitset;

    /**
     * Creates a new incremental filter iterator.
     *
     * @param iterator the underlying DocIdSetIterator (can be null)
     * @param bits     fallback Bits for random access (used if iterator is null)
     * @param maxDoc   the maximum document ID in the segment
     */
    public IncrementalFilterIterator(DocIdSetIterator iterator, Bits bits, int maxDoc) throws IOException {
        if(iterator!=null) {
            assert iterator.docID() == -1;
        }
        this.iterator = iterator;
        this.bits = bits;
        this.maxDoc = maxDoc;
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
        // No filter means all docs match
        if (iterator == null && bits == null) {
//            System.err.println("DEBUG[IncrementalFilterIterator.matches(" + docId + ")] No filter, returning true");
            return true;
        }

        // If we have bits but no iterator, use direct lookup (fallback mode)
        if (iterator == null) {
            boolean result = bits.get(docId);
//            System.err.println("DEBUG[IncrementalFilterIterator.matches(" + docId + ")] Using bits fallback, result=" + result);
            return result;
        }

        int currentPos = iterator.docID();
//        System.err.println("DEBUG[IncrementalFilterIterator.matches(" + docId + ")] Iterator at position " + currentPos);

        // If we've already scanned past this docId, check the backing bitset
        if (docId < currentPos) {
            boolean result = backingBitset != null && backingBitset.get(docId);
//            System.err.println("DEBUG[IncrementalFilterIterator.matches(" + docId + ")] Already scanned past, backing bitset result=" + result);
            return result;
        }

        // If the iterator is exhausted, no more docs can match
        if (currentPos == DocIdSetIterator.NO_MORE_DOCS) {
//            System.err.println("DEBUG[IncrementalFilterIterator.matches(" + docId + ")] Iterator exhausted, returning false");
            return false;
        }

        ensureBackingBitset();

        // If docId equals current iterator position, it matches
        if (docId == currentPos) {
            backingBitset.set(docId);
//            System.err.println("DEBUG[IncrementalFilterIterator.matches(" + docId + ")] Exact match at current position, returning true");
            return true;
        }

        // Advance iterator until we reach or pass docId, marking matching docs as we go
//        System.err.println("DEBUG[IncrementalFilterIterator.matches(" + docId + ")] Advancing from " + currentPos + " to reach " + docId);
        int nextDocId;
        int marked = 0;
        while ((nextDocId = iterator.nextDoc()) < docId) {
            backingBitset.set(nextDocId);
            marked++;
//            System.err.println("DEBUG[IncrementalFilterIterator] Marked doc " + nextDocId + " while advancing");
        }

        // Only mark nextDocId if it's a valid doc (not NO_MORE_DOCS)
        if (nextDocId != DocIdSetIterator.NO_MORE_DOCS) {
            backingBitset.set(nextDocId);
//            System.err.println("DEBUG[IncrementalFilterIterator] Marked final position " + nextDocId);
        }
//        System.err.println("DEBUG[IncrementalFilterIterator] Advanced to " + nextDocId + ", marked " + marked + " docs");

        // Check if we landed exactly on docId
        if (nextDocId == docId) {
//            System.err.println("DEBUG[IncrementalFilterIterator.matches(" + docId + ")] Landed exactly on target, returning true");
            return true;
        }

        // We advanced past docId without finding it - it doesn't match
//        System.err.println("DEBUG[IncrementalFilterIterator.matches(" + docId + ")] Advanced past target to " + nextDocId + ", returning false");
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
     * Returns -1 if using bits-based fallback or iterator is unpositioned.
     */
    int currentPosition() {
        if (iterator == null) {
            return -1; // Using bits fallback
        }
        return iterator.docID();
    }
}
