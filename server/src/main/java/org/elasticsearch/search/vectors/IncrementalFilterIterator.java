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

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.RoaringDocIdSet;
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
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * An incremental filter optimized for low-level Lucene access.
 * Uses a FixedBitSet for O(1) history lookups and minimal memory footprint.
 */
public final class IncrementalFilterIterator {

    private final DocIdSetIterator iterator;
    // FixedBitSet is generally fastest for random access checks.
    // If you have >100M docs and <1% matches, consider SparseFixedBitSet.
    private final FixedBitSet visitedBits;

    /**
     * Creates a new incremental filter iterator.
     *
     * @param iterator the underlying DocIdSetIterator (can be null)
     * @param maxDoc   the maximum document ID (required for bitset sizing)
     */
    public IncrementalFilterIterator(DocIdSetIterator iterator, int maxDoc) {
        // Optimization: Use empty() to avoid null checks in hot paths
        this.iterator = iterator == null ? DocIdSetIterator.empty() : iterator;
        // Optimization: FixedBitSet is extremely fast for sequential writes and random reads
        this.visitedBits = new FixedBitSet(maxDoc);

        // Ensure iterator is at start to prevent state mismatches
        assert this.iterator.docID() == -1 : "Iterator must be positioned at -1";
    }

    /**
     * Manually advances the iterator.
     * NOTE: Using this method creates "gaps" in the cache.
     * Checks for docIds skipped by this advance will return false.
     */
    public void advance(int docId) throws IOException {
        if (docId == NO_MORE_DOCS) {
            return; // guard against overflow logic
        }

        // Only advance if the target is ahead of us
        if (iterator.docID() < docId) {
            int current = iterator.advance(docId);
            // If we land on a valid doc, mark it immediately
            if (current != NO_MORE_DOCS) {
                visitedBits.set(current);
            }
        }
    }

    /**
     * Checks if the given document ID matches, advancing the underlying iterator
     * and caching results as needed.
     */
    public boolean matches(int docId) throws IOException {
        int currentPos = iterator.docID();

        // 1. History Check: If we are past the requested docId, check the bitset.
        // This is O(1) and extremely fast.
        if (docId < currentPos) {
            return visitedBits.get(docId);
        }

        // 2. Exact Match: If we are already on the doc.
        if (docId == currentPos) {
            return true;
        }

        // 3. Forward Scan: We need to advance to docId.
        // We MUST linear scan (nextDoc) to ensure we cache the history
        // for future "backwards" lookups.
        // If we used advance(docId), we would lose the status of skipped docs.
        int nextDoc;
        while ((nextDoc = iterator.nextDoc()) < docId) {
            visitedBits.set(nextDoc);
        }

        // We either landed ON the docId, or passed it, or hit NO_MORE_DOCS.
        if (nextDoc == docId) {
            visitedBits.set(nextDoc);
            return true;
        }

        // If we passed it (nextDoc > docId) or hit NO_MORE_DOCS,
        // then docId is definitely not a match.
        // (Note: If nextDoc != NO_MORE_DOCS, we should cache it for later)
        if (nextDoc != NO_MORE_DOCS) {
            visitedBits.set(nextDoc);
        }

        return false;
    }

    /**
     * Optional: Expose the raw bitset if other low-level APIs need it.
     */
    public BitSet getBackingBitSet() {
        return visitedBits;
    }
}
