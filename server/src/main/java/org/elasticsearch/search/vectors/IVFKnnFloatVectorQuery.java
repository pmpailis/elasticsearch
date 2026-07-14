/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopDocsCollector;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOSupplier;
import org.elasticsearch.common.lucene.Lucene;
import org.elasticsearch.index.codec.vectors.diskbbq.CrossSegmentPostingScheduler;
import org.elasticsearch.index.codec.vectors.diskbbq.IvfQueryConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.VectorPreconditioner;

import java.io.IOException;
import java.util.Arrays;
import java.util.function.LongSupplier;

/** A {@link IVFKnnFloatVectorQuery} that uses the IVF search strategy. */
public class IVFKnnFloatVectorQuery extends AbstractIVFKnnVectorQuery {

    protected final float[] query;
    /**
     * The query rotated by the field's {@link Preconditioner}, computed at most once (the transform is a fixed-seed
     * rotation, identical for every preconditioning segment of the field). {@link #query} itself is never mutated:
     * preconditioned segments store their IVF structures over rotated vectors while the raw float vectors (used by
     * rescoring) stay unrotated, and a shard may mix preconditioned and non-preconditioned segments — so each
     * consumer must pick the view matching the space it scores in via {@link #leafQuery}.
     */
    private volatile float[] preconditionedQuery;

    /**
     * Creates a new {@link IVFKnnFloatVectorQuery} with the given parameters.
     * @param field the field to search
     * @param query the query vector
     * @param k the number of nearest neighbors to return (possibly oversampled)
     * @param numCands the number of nearest neighbors to gather per shard
     * @param filter the filter to apply to the results
     * @param visitRatio the ratio of vectors to score for the IVF search strategy
     */
    public IVFKnnFloatVectorQuery(
        String field,
        float[] query,
        int k,
        int numCands,
        Query filter,
        float visitRatio,
        IvfQueryConfigResolver queryConfigResolver
    ) {
        super(field, visitRatio, k, numCands, filter, queryConfigResolver);
        this.query = query;
    }

    public float[] getQuery() {
        return query;
    }

    @Override
    public String toString(String field) {
        StringBuilder buffer = new StringBuilder();
        buffer.append(getClass().getSimpleName())
            .append(":")
            .append(this.field)
            .append("[")
            .append(query[0])
            .append(",...]")
            .append("[")
            .append(k)
            .append("]");
        if (this.filter != null) {
            buffer.append("[").append(this.filter).append("]");
        }
        return buffer.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (super.equals(o) == false) return false;
        IVFKnnFloatVectorQuery that = (IVFKnnFloatVectorQuery) o;
        return Arrays.equals(query, that.query);
    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        result = 31 * result + Arrays.hashCode(query);
        return result;
    }

    /**
     * Returns the query view matching the space {@code context}'s IVF structures are stored in: the rotated copy for
     * segments written with preconditioning, the original query for everything else. Safe to call concurrently from
     * leaf tasks; the rotated copy is computed at most once.
     */
    float[] leafQuery(LeafReaderContext context, boolean usePrecondition) throws IOException {
        if (usePrecondition == false) {
            return query;
        }
        SegmentReader segmentReader = Lucene.tryUnwrapSegmentReader(context.reader());
        if (segmentReader == null) {
            return query;
        }
        KnnVectorsReader fieldsReader = segmentReader.getVectorReader();
        if (fieldsReader instanceof PerFieldKnnVectorsFormat.FieldsReader perFieldReader
            && perFieldReader.getFieldReader(field) instanceof VectorPreconditioner vectorPreconditioner) {
            FieldInfo fieldInfo = segmentReader.getFieldInfos().fieldInfo(field);
            Preconditioner preconditioner = vectorPreconditioner.getPreconditioner(fieldInfo);
            if (preconditioner != null) {
                float[] preconditioned = preconditionedQuery;
                if (preconditioned == null) {
                    synchronized (this) {
                        preconditioned = preconditionedQuery;
                        if (preconditioned == null) {
                            preconditioned = new float[query.length];
                            preconditioner.applyTransform(query, preconditioned);
                            preconditionedQuery = preconditioned;
                        }
                    }
                }
                return preconditioned;
            }
        }
        return query;
    }

    @Override
    TopDocs getLeafResults(
        LeafReaderContext ctx,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        float visitRatio,
        boolean usePrecondition
    ) throws IOException {
        final LeafReader reader = ctx.reader();
        final Bits liveDocs = reader.getLiveDocs();
        final int maxDoc = reader.maxDoc();
        final float[] leafQuery = leafQuery(ctx, usePrecondition);

        if (filterWeight == null) {
            return approximateSearch(
                ctx,
                leafQuery,
                liveDocs == null ? new ESAcceptDocs.ESAcceptDocsAll() : new ESAcceptDocs.BitsAcceptDocs(liveDocs, maxDoc),
                Integer.MAX_VALUE,
                knnCollectorManager,
                visitRatio
            );
        }

        ScorerSupplier supplier = filterWeight.scorerSupplier(ctx);
        if (supplier == null) {
            return TopDocsCollector.EMPTY_TOPDOCS;
        }
        // no need to cache iterator as we are calling it once
        IOSupplier<DocIdSetIterator> docIdIteratorSupplier = () -> supplier.get(Long.MAX_VALUE).iterator();
        LongSupplier costSupplier = () -> supplier.cost();
        return approximateSearch(
            ctx,
            leafQuery,
            new ESAcceptDocs.ScorerSupplierAcceptDocs(docIdIteratorSupplier, costSupplier, liveDocs, maxDoc),
            Integer.MAX_VALUE,
            knnCollectorManager,
            visitRatio
        );
    }

    private TopDocs approximateSearch(
        LeafReaderContext context,
        float[] leafQuery,
        AcceptDocs acceptDocs,
        int visitedLimit,
        IVFCollectorManager knnCollectorManager,
        float visitRatio
    ) throws IOException {
        LeafReader reader = context.reader();
        CrossSegmentPostingScheduler crossSegmentScheduler = crossSegmentScheduler();
        // Under cross-segment scheduling, eligible leaves register with the query-level scheduler instead of
        // forking per-leaf workers; ineligible leaves (unsupported format/segment shape) fall through to the
        // serial path inside their leaf task. The two modes never run concurrently for the same leaf.
        IVFKnnSearchStrategy strategy = crossSegmentScheduler != null
            ? new IVFKnnSearchStrategy(visitRatio, numCands, k, knnCollectorManager.floors, null, crossSegmentScheduler, context.ord)
            : new IVFKnnSearchStrategy(visitRatio, numCands, k, knnCollectorManager.floors, newParallelScanContext(knnCollectorManager));
        AbstractMaxScoreKnnCollector knnCollector = knnCollectorManager.newCollector(visitedLimit, strategy, context);
        if (knnCollector == null) {
            return NO_RESULTS;
        }
        strategy.setCollector(knnCollector);
        reader.searchNearestVectors(field, leafQuery, knnCollector, acceptDocs);
        if (strategy.crossSegmentRegistered()) {
            // deferred: the scheduler scans this leaf after all leaf tasks return; null marks it for collection
            return null;
        }
        TopDocs results = knnCollector instanceof BulkKnnCollector bulkKnnCollector
            ? bulkKnnCollector.unsortedTopK()
            : knnCollector.topDocs();
        return results != null ? results : NO_RESULTS;
    }

    @Override
    Query getAutoRescoreQuery(IndexSearcher indexSearcher, TopDocs topOversampled, int effectiveK) {
        Query topDocsQuery = new KnnScoreDocQuery(topOversampled.scoreDocs, indexSearcher.getIndexReader());
        // rescoring scores the raw float vectors, which are stored unrotated even on preconditioned segments, so it
        // must always use the original query, never the preconditioned view
        return RescoreKnnVectorQuery.fromInnerQuery(field, query, k, effectiveK, topDocsQuery);
    }
}
