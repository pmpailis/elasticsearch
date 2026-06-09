/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.util.Bits;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;

/**
 * The per-leaf product of the parallel-IVF preparation phase (Phase A).
 * <p>
 * For an IVF leaf this carries the closest posting lists selected for the query (closest-centroid first) plus
 * everything a scoring worker needs to score them without re-deriving per-leaf state: the resolved
 * {@link ESAcceptDocs} (its filter/slice caches are warmed on the preparing thread so concurrent workers only
 * read them), the materialized filter {@link Bits}, the codec reader and field layout, and the visit ratio
 * resolved exactly as the serial path would. A leaf whose vectors are not stored in IVF form (e.g. an HNSW
 * segment in a mixed index) is flagged with {@link #needsFallbackSearch()} and scored through the regular
 * per-leaf path; for such a leaf the IVF-specific fields are {@code null}/empty.
 */
record LeafSearchTask(
    LeafReaderContext ctx,
    ESAcceptDocs acceptDocs,
    Bits materializedFilter,
    IVFVectorsReader<?> reader,
    IVFVectorsReader.FieldEntry entry,
    FieldInfo fieldInfo,
    PostingSearchTask[] postings,
    float visitRatio,
    boolean needsFallbackSearch
) {

    static LeafSearchTask ivf(
        LeafReaderContext ctx,
        ESAcceptDocs acceptDocs,
        Bits materializedFilter,
        IVFVectorsReader<?> reader,
        IVFVectorsReader.FieldEntry entry,
        FieldInfo fieldInfo,
        PostingSearchTask[] postings,
        float visitRatio
    ) {
        return new LeafSearchTask(ctx, acceptDocs, materializedFilter, reader, entry, fieldInfo, postings, visitRatio, false);
    }

    static LeafSearchTask fallback(LeafReaderContext ctx, ESAcceptDocs acceptDocs, float visitRatio) {
        return new LeafSearchTask(ctx, acceptDocs, null, null, null, null, new PostingSearchTask[0], visitRatio, true);
    }
}
