/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TotalHits;
import org.apache.lucene.util.ArrayUtil;
import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;

import java.io.IOException;

/**
 * Carries a separated kNN sub-search with metadata for coordinator-level k-limited merge.
 * Used when optimized rescoring separates kNN queries from regular sub-searches so that
 * the coordinator can merge kNN results globally limited to k.
 */
public record KnnScoreDocContainer(KnnScoreDocQueryBuilder knnScoreDocQueryBuilder, String queryName, Integer k, Float boost)
    implements
        Writeable {

    public KnnScoreDocContainer(StreamInput in) throws IOException {
        this(in.readNamedWriteable(KnnScoreDocQueryBuilder.class), in.readOptionalString(), in.readOptionalVInt(), in.readOptionalFloat());
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        out.writeNamedWriteable(knnScoreDocQueryBuilder);
        out.writeOptionalString(queryName);
        out.writeOptionalVInt(k);
        out.writeOptionalFloat(boost);
    }

    /**
     * Builds a {@link TopDocs} directly from this query's pre-computed docs and scores,
     * sorted by score descending then doc ascending.
     * This avoids going through the full Lucene search collector infra when the docs and
     * scores are already fully known.
     */
    public TopDocs toTopDocs() {
        var originalResults = knnScoreDocQueryBuilder.scoreDocs();
        ScoreDoc[] scoreDocs = new ScoreDoc[originalResults.length];
        for (int i = 0; i < originalResults.length; i++) {
            scoreDocs[i] = new ScoreDoc(originalResults[i].doc, originalResults[i].score);
        }
        // Sort by score descending, then by doc ascending for tiebreaking (matching Lucene's default)
        ArrayUtil.introSort(scoreDocs, (a, b) -> {
            int cmp = Float.compare(b.score, a.score);
            return cmp != 0 ? cmp : Integer.compare(a.doc, b.doc);
        });
        return new TopDocs(new TotalHits(originalResults.length, TotalHits.Relation.EQUAL_TO), scoreDocs);
    }

}
