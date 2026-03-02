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
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.Bits;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.index.codec.vectors.diskbbq.Preconditioner;
import org.elasticsearch.index.codec.vectors.diskbbq.VectorPreconditioner;
import org.elasticsearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This query finds the nearest centroids and creates a CentroidQuery for each, then
 * combines them using DisjunctionMaxQuery for scoring.
 */
public class CandidateIVFKnnFloatVectorQuery extends AbstractIVFKnnVectorQuery implements QueryProfilerProvider {
    private final String field;
    private final float[] queryVector;
    private final int k;
    private final int numCands;
    private final Query filter;
    private final float providedVisitRatio;
    private final int clusterSize;
    private final BitSetProducer parentsFilter; // null for non-diversifying
    private final AtomicLong totalVectorsVisited;

    public float[] getQueryVector() {
        return queryVector;
    }

    public long getTotalVectorsVisited() {
        return totalVectorsVisited.get();
    }

    /**
     * Creates a new IVFQuery.
     *
     * @param field         the vector field name
     * @param queryVector   the query vector
     * @param k             the number of nearest neighbors to return
     * @param numCands      the number of candidates to gather
     * @param filter        optional filter query
     * @param visitRatio    the ratio of vectors to score (0.0 for dynamic calculation)
     * @param clusterSize   the average size of each posting lists (i.e. how many vectors a cluster has)
     * @param parentsFilter optional parent filter for diversification (can be null)
     */
    public CandidateIVFKnnFloatVectorQuery(
        String field,
        float[] queryVector,
        int k,
        int numCands,
        Query filter,
        float visitRatio,
        int clusterSize,
        BitSetProducer parentsFilter
    ) {
        this(field, queryVector, k, numCands, filter, visitRatio, clusterSize, parentsFilter, new AtomicLong(0));
    }

    private CandidateIVFKnnFloatVectorQuery(
        String field,
        float[] queryVector,
        int k,
        int numCands,
        Query filter,
        float visitRatio,
        int clusterSize,
        BitSetProducer parentsFilter,
        AtomicLong totalVectorsVisited
    ) {
        super(field, visitRatio, k, k, filter);
        if (k < 1) {
            throw new IllegalArgumentException("k must be at least 1, got: " + k);
        }
        if (visitRatio < 0.0f || visitRatio > 1.0f) {
            throw new IllegalArgumentException("visitRatio must be between 0.0 and 1.0 (both inclusive), got: " + visitRatio);
        }
        if (numCands < k) {
            throw new IllegalArgumentException("numCands must be at least k, got: " + numCands);
        }
        this.field = field;
        this.queryVector = queryVector;
        this.k = k;
        this.numCands = numCands;
        this.filter = filter;
        this.providedVisitRatio = visitRatio;
        this.parentsFilter = parentsFilter;
        this.clusterSize = clusterSize;
        this.totalVectorsVisited = totalVectorsVisited;
    }

    @Override
    TopDocs approximateSearch(LeafReaderContext context, AcceptDocs acceptDocs, int visitedLimit, IVFCollectorManager knnCollectorManager, float visitRatio) {
        return null;
    }

    @Override
    public Query rewrite(IndexSearcher indexSearcher) throws IOException {
        if (filter != null) {
            BooleanQuery booleanQuery = new BooleanQuery.Builder()
                .add(filter, BooleanClause.Occur.FILTER)
                .add(new FieldExistsQuery(field), BooleanClause.Occur.FILTER)
                .build();
            Query filterRewritten = indexSearcher.rewrite(booleanQuery);
            if (false == filter.equals(filterRewritten)) {
                // Pass the same totalVectorsVisited instance to preserve count across rewrites
                return new CandidateIVFKnnFloatVectorQuery(
                    field, queryVector, k, numCands, filterRewritten, providedVisitRatio, clusterSize, parentsFilter, totalVectorsVisited
                );
            }
        }
        int totalVectors = 0;
        for (LeafReaderContext ctx : indexSearcher.getIndexReader().leaves()) {
            FloatVectorValues values = ctx.reader().getFloatVectorValues(field);
            if (values != null) {
                totalVectors += values.size();
            }
        }

        if (totalVectors == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }
        vectorOpsCount = 0;
        IndexReader reader = indexSearcher.getIndexReader();

        float effectiveVisitRatio = calculateEffectiveVisitRatio(totalVectors);
        Weight filterWeight = null;
        float filterSelectivity = 0f;
        if (filter != null) {
            filterWeight = indexSearcher.createWeight(filter, ScoreMode.COMPLETE_NO_SCORES, 1f);
            int filterCost = 0;
            for (LeafReaderContext ctx : indexSearcher.getIndexReader().leaves()) {
                ScorerSupplier ss = filterWeight.scorerSupplier(ctx);
                if (ss != null) {
                    filterCost += Math.toIntExact(ss.cost());
                }
            }
            filterSelectivity = Math.min(1f, (float) filterCost / totalVectors);
        }
        long maxVectorVisited = Math.max(100, Math.round(2.0 * effectiveVisitRatio * totalVectors));

        filterSelectivity = filter != null ? (1 + (1 - filterSelectivity)) : 1;
        int numCentroids = Math.max(1, (int) Math.ceil((double) (maxVectorVisited * filterSelectivity) / clusterSize));
        int expectedDocsPerCentroid = Math.max(1, clusterSize);
        numCentroids = Math.max(numCentroids, (int) Math.ceil((double) k / expectedDocsPerCentroid));

        List<Callable<List<IVFCentroidQuery>>> tasks = new ArrayList<>(reader.leaves().size());
        List<IVFCentroidQuery> allCentroidQueries = new ArrayList<>();
        final Weight finalFilterWeight = filterWeight;
        for (LeafReaderContext context : reader.leaves()) {
            tasks.add(() -> generateCentroidQueries(context, numCentroids, totalVectorsVisited, finalFilterWeight));
        }
        List<List<IVFCentroidQuery>> perLeafResults = indexSearcher.getTaskExecutor().invokeAll(tasks).stream().toList();
        for (List<IVFCentroidQuery> centroidQueries : perLeafResults) {
            if (centroidQueries != null && false == centroidQueries.isEmpty()) {
                allCentroidQueries.addAll(centroidQueries);
            }
        }
        if (allCentroidQueries.isEmpty()) {
            vectorOpsCount = 0;
            return Queries.NO_DOCS_INSTANCE;
        }

        Query ivfQuery = new DisjunctionMaxQuery(allCentroidQueries, 0.0f);

        if (parentsFilter != null) {
            ivfQuery = new DiversifyingParentBlockQuery(parentsFilter, ivfQuery);
        }

        if (filter != null) {
            BooleanQuery.Builder boolBuilder = new BooleanQuery.Builder();
            boolBuilder.add(ivfQuery, BooleanClause.Occur.MUST);
            boolBuilder.add(filter, BooleanClause.Occur.FILTER);
            boolBuilder.add(new FieldExistsQuery(field), BooleanClause.Occur.FILTER);
            ivfQuery = boolBuilder.build();
        }
        TopScoreDocCollectorManager manager = new TopScoreDocCollectorManager(k, (int) maxVectorVisited);
        TopDocs topDocs = indexSearcher.search(ivfQuery, manager);
        vectorOpsCount = (int) totalVectorsVisited.get();
        if (topDocs.scoreDocs.length == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }
        return new KnnScoreDocQuery(topDocs.scoreDocs, reader);
    }

    private List<IVFCentroidQuery> generateCentroidQueries(
        LeafReaderContext ctx,
        int numCentroids,
        AtomicLong totalVectorsVisited,
        Weight filterWeight
    ) throws IOException {
        List<IVFCentroidQuery.IVFCentroidMeta> topCentroids = findTopCentroids(ctx, numCentroids, filterWeight);

        if (topCentroids == null || topCentroids.isEmpty()) {
            return null;
        }

        List<IVFCentroidQuery> allCentroidQueries = new ArrayList<>();
        for (IVFCentroidQuery.IVFCentroidMeta centroid : topCentroids) {
            allCentroidQueries.add(new IVFCentroidQuery(field, centroid, ctx, totalVectorsVisited));
        }
        return allCentroidQueries;
    }

    private List<IVFCentroidQuery.IVFCentroidMeta> findTopCentroids(
        LeafReaderContext ctx,
        int maxCentroids,
        Weight filterWeight
    ) throws IOException {
        LeafReader leafReader = ctx.reader();
        FloatVectorValues vectorValues = leafReader.getFloatVectorValues(field);
        if (vectorValues == null || vectorValues.size() == 0) {
            return null;
        }

        if (false == leafReader instanceof CodecReader) {
            return null;
        }
        CodecReader codecReader = (CodecReader) leafReader;
        KnnVectorsReader knnVectorsReader = codecReader.getVectorReader();
        if (knnVectorsReader instanceof PerFieldKnnVectorsFormat.FieldsReader fieldsReader) {
            knnVectorsReader = fieldsReader.getFieldReader(field);
        }

        if (false == knnVectorsReader instanceof IVFVectorsReader) {
            return null;
        }
        IVFVectorsReader ivfReader = (IVFVectorsReader) knnVectorsReader;
        FieldInfo fieldInfo = leafReader.getFieldInfos().fieldInfo(field);

        float[] effectiveQueryVector = queryVector;
        if (knnVectorsReader instanceof VectorPreconditioner vp) {
            Preconditioner preconditioner = vp.getPreconditioner(fieldInfo);
            if (preconditioner != null) {
                effectiveQueryVector = new float[queryVector.length];
                preconditioner.applyTransform(queryVector, effectiveQueryVector);
            }
        }

        ESAcceptDocs acceptDocs;
        Bits liveDocs = leafReader.getLiveDocs();
        if (filterWeight != null) {
            ScorerSupplier ss = filterWeight.scorerSupplier(ctx);
            if (ss == null) {
                return null;
            }
            acceptDocs = new ESAcceptDocs.ScorerSupplierAcceptDocs(ss, liveDocs, leafReader.maxDoc());
        } else if (liveDocs != null) {
            acceptDocs = new ESAcceptDocs.BitsAcceptDocs(liveDocs, leafReader.maxDoc());
        } else {
            acceptDocs = ESAcceptDocs.ESAcceptDocsAll.INSTANCE;
        }

        return ivfReader.findTopCentroids(fieldInfo, effectiveQueryVector, maxCentroids, field, providedVisitRatio, acceptDocs);
    }

    @Override
    public String toString(String field) {
        StringBuilder buffer = new StringBuilder();
        buffer.append(getClass().getSimpleName())
            .append(":")
            .append(this.field)
            .append("[")
            .append(queryVector[0])
            .append(",...]")
            .append("[k=")
            .append(k)
            .append(", numCands=")
            .append(numCands)
            .append("]");
        if (filter != null) {
            buffer.append("[").append(filter).append("]");
        }
        return buffer.toString();
    }

    @Override
    public void visit(QueryVisitor visitor) {
        if (visitor.acceptField(field)) {
            visitor.visitLeaf(this);
        }
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        CandidateIVFKnnFloatVectorQuery that = (CandidateIVFKnnFloatVectorQuery) obj;
        return k == that.k
            && numCands == that.numCands
            && Float.compare(that.providedVisitRatio, providedVisitRatio) == 0
            && Objects.equals(field, that.field)
            && Objects.deepEquals(queryVector, that.queryVector)
            && Objects.equals(filter, that.filter)
            && Objects.equals(parentsFilter, that.parentsFilter);
    }

    @Override
    public int hashCode() {
        return Objects.hash(field, k, numCands, providedVisitRatio, filter, parentsFilter);
    }

    /**
     * Calculates the effective visit ratio based on total vectors and query parameters.
     * Uses dynamic calculation if providedVisitRatio is 0.0.
     */
    private float calculateEffectiveVisitRatio(int totalVectors) {
        if (providedVisitRatio > 0.0f) {
            return providedVisitRatio;
        }

        // Dynamic calculation: log10(totalVectors)² * min(10000, max(numCands, 5*k))
        float expected = (float) Math.round(
            Math.log10(totalVectors) * Math.log10(totalVectors) * (Math.min(10_000, Math.max(numCands, 5 * k)))
        );
        return Math.min(1.0f, expected / totalVectors);
    }
}
