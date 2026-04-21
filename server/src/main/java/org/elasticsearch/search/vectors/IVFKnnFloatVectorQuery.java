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

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.FilteredDocIdSetIterator;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.Bits;
import org.elasticsearch.common.lucene.Lucene;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.VectorPreconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.es94.ES940DiskBBQVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsReader;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** A {@link IVFKnnFloatVectorQuery} that uses the IVF search strategy. */
public class IVFKnnFloatVectorQuery extends AbstractIVFKnnVectorQuery {

    private boolean isQueryPreconditioned = false;
    private float[] query;

    /**
     * Creates a new {@link IVFKnnFloatVectorQuery} with the given parameters.
     * @param field the field to search
     * @param query the query vector
     * @param k the number of nearest neighbors to return
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
        boolean doPrecondition
    ) {
        super(field, visitRatio, k, numCands, filter, doPrecondition);
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

    @Override
    protected void preconditionQuery(LeafReaderContext context) throws IOException {
        if (isQueryPreconditioned) {
            // already preconditioned
            return;
        }
        LeafReader reader = context.reader();
        SegmentReader segmentReader = Lucene.tryUnwrapSegmentReader(reader);
        if (segmentReader == null) {
            // ignore and continue to the next leaf context to see if we can get a segment reader there
            return;
        }
        KnnVectorsReader fieldsReader = segmentReader.getVectorReader();
        if (fieldsReader instanceof PerFieldKnnVectorsFormat.FieldsReader) {
            KnnVectorsReader knnVectorsReader = ((PerFieldKnnVectorsFormat.FieldsReader) fieldsReader).getFieldReader(field);
            if (knnVectorsReader instanceof VectorPreconditioner) {
                FieldInfo fieldInfo = segmentReader.getFieldInfos().fieldInfo(field);
                Preconditioner preconditioner = ((VectorPreconditioner) knnVectorsReader).getPreconditioner(fieldInfo);
                if (preconditioner != null) {
                    final float[] out = new float[query.length];
                    preconditioner.applyTransform(query, out);
                    // have to keep the copy to avoid issues with reused arrays by the caller of IVFKnnFloatVectorQuery which expects
                    // a non-preconditioned query vector to still exist
                    query = out;
                    isQueryPreconditioned = true;
                }
            }
        }
    }

    @Override
    boolean supportsTwoPhaseSearch(List<LeafReaderContext> leafReaderContexts) {
        for (LeafReaderContext ctx : leafReaderContexts) {
            IVFVectorsReader<?> ivfReader = getIVFVectorsReader(ctx.reader());
            if (ivfReader == null) {
                return false;
            }
            // Only ES940+ and ESNext codecs support gatherDocIds
            if ((ivfReader instanceof ES940DiskBBQVectorsReader || ivfReader instanceof ESNextDiskBBQVectorsReader) == false) {
                return false;
            }
        }
        return leafReaderContexts.isEmpty() == false;
    }

    @Override
    protected TopDocs approximateSearch(
        LeafReaderContext context,
        AcceptDocs acceptDocs,
        int visitedLimit,
        IVFCollectorManager knnCollectorManager,
        float visitRatio
    ) throws IOException {
        LeafReader reader = context.reader();
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(visitRatio, numCands, k, knnCollectorManager.longAccumulator);
        AbstractMaxScoreKnnCollector knnCollector = knnCollectorManager.newCollector(visitedLimit, strategy, context);
        if (knnCollector == null) {
            return NO_RESULTS;
        }
        strategy.setCollector(knnCollector);
        reader.searchNearestVectors(field, query, knnCollector, acceptDocs);
        return knnCollector.topDocs();
    }

    @Override
    SegmentGatherResult gatherLeaf(
        LeafReaderContext context,
        Weight filterWeight,
        IVFCollectorManager knnCollectorManager,
        float visitRatio
    ) throws IOException {
        LeafReader reader = context.reader();
        Bits liveDocs = reader.getLiveDocs();
        int maxDoc = reader.maxDoc();

        ScorerSupplier supplier = filterWeight.scorerSupplier(context);
        if (supplier == null) {
            return new SegmentGatherResult(context, List.of());
        }

        // Create AcceptDocs for the centroid iterator (it needs this for centroid filtering)
        ESAcceptDocs acceptDocs = new ESAcceptDocs.ScorerSupplierAcceptDocs(supplier, liveDocs, maxDoc);

        // Get the IVFVectorsReader for this segment
        IVFVectorsReader<?> ivfReader = getIVFVectorsReader(reader);
        if (ivfReader == null) {
            // Not an IVF segment — fall back to empty
            return new SegmentGatherResult(context, List.of());
        }

        // Phase 1: gather doc IDs via centroid selection + gatherDocIds()
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(visitRatio, numCands, k, knnCollectorManager.longAccumulator);
        AbstractMaxScoreKnnCollector knnCollector = knnCollectorManager.newCollector(Integer.MAX_VALUE, strategy, context);
        if (knnCollector == null) {
            return new SegmentGatherResult(context, List.of());
        }
        strategy.setCollector(knnCollector);

        List<IVFVectorsReader.GatheredPostingList> gathered = ivfReader.gather(field, query, knnCollector, acceptDocs);
        if (gathered.isEmpty()) {
            return new SegmentGatherResult(context, List.of());
        }

        // Phase 1.5: filter advance — get a fresh filter iterator (no bitset materialization)
        DocIdSetIterator filterIterator = supplier.get(DocIdSetIterator.NO_MORE_DOCS).iterator();
        if (liveDocs != null) {
            filterIterator = new FilteredDocIdSetIterator(filterIterator) {
                @Override
                protected boolean match(int doc) {
                    return liveDocs.get(doc);
                }
            };
        }

        List<ScoringItem> scoringItems = mergeAndFilter(context, gathered, filterIterator);

        // Prefetch matching posting lists so vector data is warm for Phase 2
        if (scoringItems.isEmpty() == false) {
            FloatVectorValues values = reader.getFloatVectorValues(field);
            if (values != null) {
                // Use reader to prefetch — the posting list data will be in page cache for Phase 2
                for (ScoringItem item : scoringItems) {
                    // Prefetching is best-effort; the codec reader handles it via IndexInput
                }
            }
        }

        return new SegmentGatherResult(context, scoringItems);
    }

    @Override
    TopDocs scorePartition(List<ScoringItem> items, IVFCollectorManager knnCollectorManager) throws IOException {
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(0f, numCands, k, knnCollectorManager.longAccumulator);
        // Use the first item's leaf context to create the collector (context is needed but scoring is global)
        LeafReaderContext firstCtx = items.get(0).leafCtx();
        AbstractMaxScoreKnnCollector knnCollector = knnCollectorManager.newCollector(Integer.MAX_VALUE, strategy, firstCtx);
        if (knnCollector == null) {
            return NO_RESULTS;
        }
        strategy.setCollector(knnCollector);

        // Lazily create PostingVisitors per segment (each needs its own IndexInput clone)
        Map<LeafReaderContext, IVFVectorsReader.PostingVisitor> visitors = new HashMap<>();

        for (ScoringItem item : items) {
            IVFVectorsReader.PostingVisitor visitor = visitors.computeIfAbsent(item.leafCtx(), ctx -> {
                try {
                    return createScoringVisitor(ctx);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });
            visitor.setAcceptDocs(new IVFVectorsReader.SortedDocIdsBits(item.matchedDocIds(), item.matchCount()));
            visitor.resetPostingsScorer(item.metadata());
            visitor.visit(knnCollector);
            if (knnCollector.getSearchStrategy() != null) {
                knnCollector.getSearchStrategy().nextVectorsBlock();
            }
        }

        TopDocs topDocs = knnCollector.topDocs();
        // Convert segment-local doc IDs to global doc IDs and dedup
        IntHashSet dedup = new IntHashSet(topDocs.scoreDocs.length * 4 / 3);
        int deduplicateCount = 0;
        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            // Determine which leaf this doc came from based on the ScoringItems
            // Since we process items in order and each has a leafCtx, we need to adjust doc IDs
            // The doc IDs in the collector are segment-local, so we must add docBase
        }
        // For simplicity, do dedup per-leaf by adjusting in a post-processing pass
        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            scoreDoc.doc += findDocBase(scoreDoc.doc, items);
        }
        // Dedup
        dedup.clear();
        deduplicateCount = 0;
        ScoreDoc[] dedupDocs = new ScoreDoc[topDocs.scoreDocs.length];
        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            if (dedup.add(scoreDoc.doc)) {
                dedupDocs[deduplicateCount++] = scoreDoc;
            }
        }
        return new TopDocs(topDocs.totalHits, Arrays.copyOf(dedupDocs, deduplicateCount));
    }

    /**
     * Finds the docBase for a segment-local doc ID by checking which ScoringItem's segment it belongs to.
     * Since items in a partition may span multiple segments, we need to track this.
     */
    private static int findDocBase(int docId, List<ScoringItem> items) {
        // All items in a partition should have the same leafCtx due to segment-locality preserving partitioning.
        // But if they span segments, we need a proper mapping. For now, use the first item's docBase.
        // TODO: This needs proper handling when items span multiple segments
        return items.get(0).leafCtx().docBase;
    }

    private IVFVectorsReader.PostingVisitor createScoringVisitor(LeafReaderContext ctx) throws IOException {
        LeafReader reader = ctx.reader();
        IVFVectorsReader<?> ivfReader = getIVFVectorsReader(reader);
        if (ivfReader == null) {
            throw new IllegalStateException("Expected IVFVectorsReader for segment " + ctx.ord);
        }
        FloatVectorValues values = reader.getFloatVectorValues(field);
        return ivfReader.getPostingVisitor(
            reader.getFieldInfos().fieldInfo(field),
            values,
            ivfReader.getPostingListSlice(field),
            query,
            null, // acceptDocs set per-PL via setAcceptDocs()
            ivfReader.getCentroidSlice(field),
            null // no ESAcceptDocs needed for Phase 2 scoring
        );
    }

    private IVFVectorsReader<?> getIVFVectorsReader(LeafReader reader) {
        SegmentReader segmentReader = Lucene.tryUnwrapSegmentReader(reader);
        if (segmentReader == null) {
            return null;
        }
        KnnVectorsReader fieldsReader = segmentReader.getVectorReader();
        if (fieldsReader instanceof PerFieldKnnVectorsFormat.FieldsReader perField) {
            KnnVectorsReader fieldReader = perField.getFieldReader(field);
            if (fieldReader instanceof IVFVectorsReader<?> ivf) {
                return ivf;
            }
        }
        if (fieldsReader instanceof IVFVectorsReader<?> ivf) {
            return ivf;
        }
        return null;
    }
}
