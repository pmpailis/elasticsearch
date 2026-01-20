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

import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.PriorityQueue;

public class MultiPostingListManager {

    private final int[][] docIDs;

    private final IVFVectorsReader.PostingVisitor[] postingVisitors;
    private final DocIdSetIterator filterIterator;
    private final PriorityQueue<PostingListNode> minHeap;
    private final KnnCollector knnCollector;
    private int scoredDocs = 0;

    private static class PostingListNode implements Comparable<PostingListNode> {
        int docID;
        int centroidIdx;
        int postingIdx;

        private PostingListNode(int docID, int centroidIdx, int postingIdx) {
            this.docID = docID;
            this.centroidIdx = centroidIdx;
            this.postingIdx = postingIdx;
        }

        @Override
        public int compareTo(PostingListNode other) {
            return Integer.compare(this.docID, other.docID);
        }
    }

    public MultiPostingListManager(int[][] docIDs, IVFVectorsReader.PostingVisitor[] postingVisitors, DocIdSetIterator filterIterator, KnnCollector knnCollector) throws IOException {
        if (docIDs.length != postingVisitors.length) {
            throw new IllegalArgumentException("Must provide exactly one listener per array row.");
        }

        this.docIDs = docIDs;
        this.postingVisitors = postingVisitors;
        this.filterIterator = filterIterator;
        this.minHeap = new PriorityQueue<>();
        this.knnCollector = knnCollector;

        // Initialize Heap
        for (int i = 0; i < docIDs.length; i++) {
            if (docIDs[i] != null && docIDs[i].length > 0) {
                minHeap.offer(new PostingListNode(docIDs[i][0], i, 0));
            }
        }
    }

    int scoredDocs(){
        return scoredDocs;
    }

    public int accepts(int doc) throws IOException {
        if (filterIterator == null) return doc;

        if (filterIterator.docID() == DocIdSetIterator.NO_MORE_DOCS || filterIterator.docID() > doc) {
            return -1;
        }
        if (filterIterator.docID() == doc) {
            return doc;
        }
        filterIterator.advance(doc);
        if (filterIterator.docID() == doc) {
            return doc;
        }
        return -1;
    }

    public boolean hasNext() {
        return false == minHeap.isEmpty();
    }

    public void consumeAll() throws IOException {
        while (hasNext()) {
            consumeOne();
        }
    }

    private void consumeOne() throws IOException {
        if (minHeap.isEmpty()) {
            throw new NoSuchElementException("Stream is empty.");
        }

        PostingListNode minNode = minHeap.poll();

        int minDoc = minNode.docID;
        int centroidIdx = minNode.centroidIdx;
        int postingIdx = minNode.postingIdx;

        docIDs[centroidIdx][postingIdx] = accepts(minDoc);

        int nextIndex = postingIdx + 1;
        if (nextIndex < docIDs[centroidIdx].length) {
            int nextValue = docIDs[centroidIdx][nextIndex];
            minHeap.offer(new PostingListNode(nextValue, centroidIdx, nextIndex));
        } else {
            scoredDocs += postingVisitors[centroidIdx].scoreCurrentBatch(docIDs[centroidIdx], knnCollector);
            int read = postingVisitors[centroidIdx].readNextBatch(docIDs[centroidIdx]);
            if (knnCollector.getSearchStrategy() != null) {
                knnCollector.getSearchStrategy().nextVectorsBlock();
            }
            if (read > 0) {
                minHeap.offer(new PostingListNode(docIDs[centroidIdx][0], centroidIdx, 0));
            } else {
                docIDs[centroidIdx] = null;
                postingVisitors[centroidIdx] = null;
            }
        }
    }
}
