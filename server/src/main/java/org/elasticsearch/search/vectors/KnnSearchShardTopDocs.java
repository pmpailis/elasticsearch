/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.elasticsearch.common.io.stream.StreamInput;
import org.elasticsearch.common.io.stream.StreamOutput;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.common.lucene.search.TopDocsAndMaxScore;

import java.io.IOException;

import static org.elasticsearch.common.lucene.Lucene.readTopDocs;
import static org.elasticsearch.common.lucene.Lucene.writeTopDocs;

/**
  * Per-shard kNN search TopDocs result with metadata for coordinator merge.
  * Each entry corresponds to one separated kNN sub-search, carrying the TopDocs
  * produced by that search on this shard, the original k limit, and the boost.
*/
public record KnnSearchShardTopDocs(TopDocsAndMaxScore topDocs, int k, float boost) implements Writeable {

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        writeTopDocs(out, topDocs);
        out.writeVInt(k);
        out.writeFloat(boost);
    }

    public KnnSearchShardTopDocs(StreamInput in) throws IOException {
        this(readTopDocs(in), in.readVInt(), in.readFloat());
    }
}
