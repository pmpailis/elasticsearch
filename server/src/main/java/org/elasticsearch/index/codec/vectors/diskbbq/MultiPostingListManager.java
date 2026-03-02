/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.index.codec.vectors.diskbbq;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.elasticsearch.search.vectors.IncrementalDeduplicationFilter;

import java.io.IOException;
import java.util.NoSuchElementException;

public class MultiPostingListManager {

    private final int[][] docIDs;

    private final IVFVectorsReader.PostingVisitor[] postingVisitors;
    private final DocIdSetIterator filterIterator;
    private final BitSet filterBitSet;
    private final KnnCollector knnCollector;
    private final IncrementalDeduplicationFilter dedup;
    private int scoredDocs = 0;

    // Primitive heap to avoid object allocation
    // Encoding: docID (32 bits) | centroidIdx (16 bits) | postingIdx (16 bits)
    private final long[] heap;
    private int heapSize;

    private static final int DOC_SHIFT = 32;
    private static final int CENTROID_SHIFT = 16;
    private static final int CENTROID_MASK = 0xFFFF;
    private static final int POSTING_MASK = 0xFFFF;

    private static long encode(int docID, int centroidIdx, int postingIdx) {
        return ((long) docID << DOC_SHIFT) | ((long) centroidIdx << CENTROID_SHIFT) | postingIdx;
    }

    private static int decodeDocID(long encoded) {
        return (int) (encoded >>> DOC_SHIFT);
    }

    private static int decodeCentroidIdx(long encoded) {
        return (int) ((encoded >>> CENTROID_SHIFT) & CENTROID_MASK);
    }

    private static int decodePostingIdx(long encoded) {
        return (int) (encoded & POSTING_MASK);
    }

    public MultiPostingListManager(
        int[][] docIDs,
        IVFVectorsReader.PostingVisitor[] postingVisitors,
        DocIdSetIterator filterIterator,
        KnnCollector knnCollector,
        IncrementalDeduplicationFilter dedup
    ) throws IOException {
        if (docIDs.length != postingVisitors.length) {
            throw new IllegalArgumentException("Must provide exactly one listener per array row.");
        }

        this.docIDs = docIDs;
        this.postingVisitors = postingVisitors;
        this.filterIterator = filterIterator;
        if (filterIterator instanceof BitSetIterator bsi) {
            this.filterBitSet = bsi.getBitSet();
        } else {
            this.filterBitSet = null;
        }
        this.knnCollector = knnCollector;
        this.dedup = dedup;
        this.heap = new long[docIDs.length];
        this.heapSize = 0;

        // Initialize heap with first doc from each posting list
        for (int i = 0; i < docIDs.length; i++) {
            if (docIDs[i] != null && docIDs[i].length > 0) {
                heap[heapSize++] = encode(docIDs[i][0], i, 0);
            }
        }

        // Build heap (heapify)
        for (int i = (heapSize >>> 1) - 1; i >= 0; i--) {
            siftDown(i);
        }
    }

    int scoredDocs() {
        return scoredDocs;
    }

    public int accepts(int doc) throws IOException {
        if (dedup != null && dedup.add(doc) == false) {
            return -1;
        }

        if (filterIterator == null) return doc;

        if (filterBitSet != null) {
            return filterBitSet.get(doc) ? doc : -1;
        }

        if (filterIterator.docID() == DocIdSetIterator.NO_MORE_DOCS || filterIterator.docID() > doc) {
            return -1;
        }
        if (filterIterator.docID() == doc) {
            return doc;
        }
        if (doc == filterIterator.advance(doc)) {
            return doc;
        }
        return -1;
    }

    public boolean hasNext() {
        return heapSize > 0;
    }

    public void consumeAll() throws IOException {
        if(filterIterator!=null) {
            while (hasNext()) {
                consumeOne();
            }
        }else{
            consumeFlat();
        }
    }

    private void consumeFlat() throws IOException {
        for (int centroidIdx = 0; centroidIdx < docIDs.length; centroidIdx++) {
            if (docIDs[centroidIdx] == null || postingVisitors[centroidIdx] == null) {
                continue;
            }

            do {
                if (dedup != null) {
                    for (int i = 0; i < docIDs[centroidIdx].length; i++) {
                        int doc = docIDs[centroidIdx][i];
                        if (doc != -1 && dedup.add(doc) == false) {
                            docIDs[centroidIdx][i] = -1;
                        }
                    }
                }
                scoredDocs += postingVisitors[centroidIdx].scoreCurrentBatch(docIDs[centroidIdx], knnCollector);
                if (knnCollector.getSearchStrategy() != null) {
                    knnCollector.getSearchStrategy().nextVectorsBlock();
                }
            } while (postingVisitors[centroidIdx].readNextBatch(docIDs[centroidIdx]) > 0);

            docIDs[centroidIdx] = null;
            postingVisitors[centroidIdx] = null;
        }
    }

    private void consumeOne() throws IOException {
        if (heapSize == 0) {
            throw new NoSuchElementException("Stream is empty.");
        }

        long minEncoded = heap[0];
        int minDoc = decodeDocID(minEncoded);
        int centroidIdx = decodeCentroidIdx(minEncoded);
        int postingIdx = decodePostingIdx(minEncoded);

        docIDs[centroidIdx][postingIdx] = accepts(minDoc);

        int nextIndex = postingIdx + 1;
        if (nextIndex < docIDs[centroidIdx].length) {
            // Update top in place and sift down - no allocation
            int nextValue = docIDs[centroidIdx][nextIndex];
            heap[0] = encode(nextValue, centroidIdx, nextIndex);
            siftDown(0);
        } else {
            scoredDocs += postingVisitors[centroidIdx].scoreCurrentBatch(docIDs[centroidIdx], knnCollector);
            int read = postingVisitors[centroidIdx].readNextBatch(docIDs[centroidIdx]);
            if (knnCollector.getSearchStrategy() != null) {
                knnCollector.getSearchStrategy().nextVectorsBlock();
            }
            if (read > 0) {
                // Update top with first doc of new batch
                heap[0] = encode(docIDs[centroidIdx][0], centroidIdx, 0);
                siftDown(0);
            } else {
                // Remove this posting list from heap
                docIDs[centroidIdx] = null;
                postingVisitors[centroidIdx] = null;
                heapSize--;
                if (heapSize > 0) {
                    heap[0] = heap[heapSize];
                    siftDown(0);
                }
            }
        }
    }

    private void siftDown(int pos) {
        long val = heap[pos];
        int half = heapSize >>> 1;
        while (pos < half) {
            int child = (pos << 1) + 1;
            int right = child + 1;
            if (right < heapSize && heap[right] < heap[child]) {
                child = right;
            }
            if (val <= heap[child]) {
                break;
            }
            heap[pos] = heap[child];
            pos = child;
        }
        heap[pos] = val;
    }
}
