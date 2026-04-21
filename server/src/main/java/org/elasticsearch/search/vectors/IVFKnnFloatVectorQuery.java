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
import java.util.ArrayList;
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
    boolean supportsParallelScoring(List<LeafReaderContext> leafReaderContexts) {
        for (LeafReaderContext ctx : leafReaderContexts) {
            IVFVectorsReader<?> ivfReader = getIVFVectorsReader(ctx.reader());
            if (ivfReader == null) {
                return false;
            }
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
    List<ScoringItem> gatherLeaf(LeafReaderContext context, Weight filterWeight, IVFCollectorManager knnCollectorManager, float visitRatio)
        throws IOException {
        LeafReader reader = context.reader();
        Bits liveDocs = reader.getLiveDocs();
        int maxDoc = reader.maxDoc();

        // Build AcceptDocs for centroid iterator (needed for centroid filtering)
        final AcceptDocs acceptDocs;
        if (filterWeight != null) {
            ScorerSupplier supplier = filterWeight.scorerSupplier(context);
            if (supplier == null) {
                return List.of();
            }
            acceptDocs = new ESAcceptDocs.ScorerSupplierAcceptDocs(supplier, liveDocs, maxDoc);
        } else {
            acceptDocs = liveDocs == null ? new ESAcceptDocs.ESAcceptDocsAll() : new ESAcceptDocs.BitsAcceptDocs(liveDocs, maxDoc);
        }

        IVFVectorsReader<?> ivfReader = getIVFVectorsReader(reader);
        if (ivfReader == null) {
            return List.of();
        }

        // Gather doc IDs from posting lists (centroids selected, vectors skipped)
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(visitRatio, numCands, k, knnCollectorManager.longAccumulator);
        AbstractMaxScoreKnnCollector knnCollector = knnCollectorManager.newCollector(Integer.MAX_VALUE, strategy, context);
        if (knnCollector == null) {
            return List.of();
        }
        strategy.setCollector(knnCollector);

        List<IVFVectorsReader.GatheredPostingList> gathered = ivfReader.gather(field, query, knnCollector, acceptDocs);
        if (gathered.isEmpty()) {
            return List.of();
        }

        if (filterWeight != null) {
            // Filtered: merge doc IDs via per-segment heap, walk filter with advance()
            ScorerSupplier freshSupplier = filterWeight.scorerSupplier(context);
            if (freshSupplier == null) {
                return List.of();
            }
            DocIdSetIterator filterIterator = freshSupplier.get(DocIdSetIterator.NO_MORE_DOCS).iterator();
            if (liveDocs != null) {
                filterIterator = new FilteredDocIdSetIterator(filterIterator) {
                    @Override
                    protected boolean match(int doc) {
                        return liveDocs.get(doc);
                    }
                };
            }
            return gatherAndFilter(context, gathered, filterIterator, ivfReader, field);
        }

        // No filter: all PLs go to scoring, prefetch their data
        List<ScoringItem> items = new ArrayList<>(gathered.size());
        for (IVFVectorsReader.GatheredPostingList pl : gathered) {
            if (pl.numDocs() > 0) {
                items.add(new ScoringItem(context, pl.metadata(), null, pl.numDocs()));
                ivfReader.prefetchPostingList(field, pl.metadata());
            }
        }
        return items;
    }

    @Override
    TopDocs scorePartition(List<ScoringItem> items, IVFCollectorManager knnCollectorManager) throws IOException {
        IVFKnnSearchStrategy strategy = new IVFKnnSearchStrategy(0f, numCands, k, knnCollectorManager.longAccumulator);
        LeafReaderContext firstCtx = items.get(0).leafCtx();
        AbstractMaxScoreKnnCollector knnCollector = knnCollectorManager.newCollector(Integer.MAX_VALUE, strategy, firstCtx);
        if (knnCollector == null) {
            return NO_RESULTS;
        }
        strategy.setCollector(knnCollector);

        // Per-segment PostingVisitors (each thread needs its own cloned IndexInputs)
        Map<LeafReaderContext, IVFVectorsReader.PostingVisitor> visitors = new HashMap<>();

        for (ScoringItem item : items) {
            IVFVectorsReader.PostingVisitor visitor = visitors.computeIfAbsent(item.leafCtx(), ctx -> {
                try {
                    return createScoringVisitor(ctx);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });

            // Set per-PL filter: null means no filter (score all), otherwise SortedDocIdsBits
            if (item.matchedDocIds() != null) {
                visitor.setAcceptDocs(new IVFVectorsReader.SortedDocIdsBits(item.matchedDocIds(), item.numDocs()));
            } else {
                visitor.setAcceptDocs(null); // no filter — score all docs in this PL
            }

            visitor.resetPostingsScorer(item.metadata());
            visitor.visit(knnCollector);
            if (knnCollector.getSearchStrategy() != null) {
                knnCollector.getSearchStrategy().nextVectorsBlock();
            }
        }

        // Globalize doc IDs and dedup
        TopDocs topDocs = knnCollector.topDocs();
        // Since a partition may span multiple segments, we need per-doc segment lookup.
        // Build a segment boundary map for efficient lookup.
        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            // The collector stores segment-local doc IDs. We need to find which segment
            // each doc came from. Since items are ordered by segment (partitioning preserves
            // locality), and docIds within a segment are unique ranges, use the leaf's docBase.
            // For partitions spanning multiple segments, find the right leaf.
            scoreDoc.doc += findDocBase(scoreDoc.doc, items);
        }
        return deduplicateTopDocs(topDocs);
    }

    private static int findDocBase(int localDocId, List<ScoringItem> items) {
        // Partitions preserve segment locality, so most items share the same leafCtx.
        // For multi-segment partitions, find the segment whose maxDoc range contains localDocId.
        LeafReaderContext lastCtx = null;
        for (ScoringItem item : items) {
            LeafReaderContext ctx = item.leafCtx();
            if (ctx != lastCtx) {
                if (localDocId < ctx.reader().maxDoc()) {
                    return ctx.docBase;
                }
                lastCtx = ctx;
            }
        }
        // Fallback: use the last segment seen
        return items.get(items.size() - 1).leafCtx().docBase;
    }

    private static TopDocs deduplicateTopDocs(TopDocs topDocs) {
        IntHashSet dedup = new IntHashSet(topDocs.scoreDocs.length * 4 / 3);
        int count = 0;
        ScoreDoc[] dedupDocs = new ScoreDoc[topDocs.scoreDocs.length];
        for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
            if (dedup.add(scoreDoc.doc)) {
                dedupDocs[count++] = scoreDoc;
            }
        }
        return new TopDocs(topDocs.totalHits, Arrays.copyOf(dedupDocs, count));
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
            null  // no ESAcceptDocs needed for Phase 2
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
