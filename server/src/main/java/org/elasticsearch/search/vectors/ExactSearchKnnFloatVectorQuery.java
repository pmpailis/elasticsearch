/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the Server Side Public License, v 1; you may not use this file except
 * in compliance with, at your election, the Elastic License 2.0 or the Server
 * Side Public License, v 1.
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.HitQueue;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.Bits;
import org.elasticsearch.index.IndexVersion;
import org.elasticsearch.index.mapper.vectors.VectorEncoderDecoder;

import java.io.IOException;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class ExactSearchKnnFloatVectorQuery extends ESKnnFloatVectorQuery {

    private final VectorSimilarityFunction vectorySimilarityFunction;

    public ExactSearchKnnFloatVectorQuery(String field, float[] target, int k, Query filter, VectorSimilarityFunction similarityFunction) {
        super(field, target, k, filter);
        this.vectorySimilarityFunction = similarityFunction;
    }

    @Override
    protected TopDocs getLeafResults(LeafReaderContext ctx, Weight filterWeight) throws IOException {
        Bits liveDocs = ctx.reader().getLiveDocs();
        int maxDoc = ctx.reader().maxDoc();

        var fieldDocValues = ctx.reader().getBinaryDocValues(field);
        if (fieldDocValues == null) return TopDocsCollector.EMPTY_TOPDOCS;


        Scorer scorer = filterWeight.scorer(ctx);
        if (scorer == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }

        BitSet acceptDocs = createBitSet(scorer.iterator(), liveDocs, maxDoc);
        int cost = acceptDocs.cardinality();
        DocIdSetIterator docIdSetIterator = new BitSetIterator(acceptDocs, cost);

        HitQueue queue = new HitQueue(k, true);
        ScoreDoc topDoc = queue.top();
        int doc;
        while ((doc = docIdSetIterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            int a = fieldDocValues.advance(doc);
            if (a == NO_MORE_DOCS) break;
            float[] storedVector = new float[96];
            VectorEncoderDecoder.decodeDenseVector(IndexVersion.current(), fieldDocValues.binaryValue(), storedVector);
            float score = vectorySimilarityFunction.compare(target, storedVector);
            if (score > topDoc.score) {
                topDoc.score = score;
                topDoc.doc = doc;
                topDoc = queue.updateTop();
            }
        }

        // Remove any remaining sentinel values
        while (queue.size() > 0 && queue.top().score < 0) {
            queue.pop();
        }

        ScoreDoc[] topScoreDocs = new ScoreDoc[queue.size()];
        for (int i = topScoreDocs.length - 1; i >= 0; i--) {
            topScoreDocs[i] = queue.pop();
        }

        TotalHits totalHits = new TotalHits(topScoreDocs.length, TotalHits.Relation.EQUAL_TO);
        return new TopDocs(totalHits, topScoreDocs);

    }
}
