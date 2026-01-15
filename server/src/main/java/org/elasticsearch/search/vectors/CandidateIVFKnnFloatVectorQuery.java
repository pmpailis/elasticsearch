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
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.util.Bits;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.index.codec.vectors.cluster.NeighborQueue;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.search.profile.query.QueryProfiler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicLong;

/**
 * IVF (Inverted File) vector search query that composes multiple CentroidQuery instances.
 * This query finds the nearest centroids and creates a CentroidQuery for each, then
 * combines them using DisjunctionMaxQuery for scoring.
 *
 * This is a new architecture that separates centroid iteration from document collection,
 * enabling more flexible query composition and better integration with Lucene's query infrastructure.
 */
public class CandidateIVFKnnFloatVectorQuery extends Query implements QueryProfilerProvider {
    private final String field;
    private final float[] queryVector;
    private final int k;
    private final int numCands;
    private final Query filter;
    private final float providedVisitRatio;
    private final int clusterSize;
    private final BitSetProducer parentsFilter; // null for non-diversifying

    private int vectorOpsCount = 0;

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
                return new CandidateIVFKnnFloatVectorQuery(
                    field, queryVector, k, numCands, filterRewritten, providedVisitRatio, clusterSize, parentsFilter
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
        return this;
    }

    @Override
    public Weight createWeight(IndexSearcher indexSearcher, ScoreMode scoreMode, float boost) throws IOException {
        vectorOpsCount = 0;
        IndexReader reader = indexSearcher.getIndexReader();

        int totalVectors = 0;
        for (LeafReaderContext ctx : indexSearcher.getIndexReader().leaves()) {
            FloatVectorValues values = ctx.reader().getFloatVectorValues(field);
            if (values != null) {
                totalVectors += values.size();
            }
        }

        // calculate effective visit ratio, i.e. how many vectors we're expected to visit
        float effectiveVisitRatio = calculateEffectiveVisitRatio(totalVectors);
        long maxVectorVisited = Math.round(2.0 * effectiveVisitRatio * totalVectors);

        // cross-leaf competitive scores; replacing the collector
        // might need to revisit this to check if we can still use the collector
        AtomicLong globalMinCompetitiveScore = reader.leaves().size() > 1
            ? new AtomicLong(NeighborQueue.encodeRaw(Integer.MAX_VALUE, Float.NEGATIVE_INFINITY))
            : null;

        // calculate number of centroids to initially explore
        int numCentroids = Math.max(1, (int) Math.ceil((float) maxVectorVisited / clusterSize));

        // or each leaf, find top centroids and create CentroidQueries
        List<Callable<List<IVFCentroidQuery>>> tasks = new ArrayList<>(reader.leaves().size());
        List<Query> allCentroidQueries = new ArrayList<>();
        for (LeafReaderContext context : reader.leaves()) {
            tasks.add(() -> generateCentroidQueries(context, numCentroids, maxVectorVisited, globalMinCompetitiveScore));
        }
        List<List<IVFCentroidQuery>> perLeafResults = indexSearcher.getTaskExecutor().invokeAll(tasks).stream().toList();
        for (List<IVFCentroidQuery> centroidQueries : perLeafResults) {
            if (centroidQueries != null && false == centroidQueries.isEmpty()) {
                allCentroidQueries.addAll(centroidQueries);
            }
        }

        // 7. Compose with DisMaxQuery
        Query ivfQuery = new DisjunctionMaxQuery(allCentroidQueries, 0.0f);

        // 8. Wrap with filter if needed - let BoolQuery handle filtering!
        if (filter != null) {
            BooleanQuery.Builder boolBuilder = new BooleanQuery.Builder();
            boolBuilder.add(ivfQuery, BooleanClause.Occur.SHOULD);
            boolBuilder.add(filter, BooleanClause.Occur.FILTER);
            boolBuilder.add(new FieldExistsQuery(field), BooleanClause.Occur.FILTER);
            ivfQuery = boolBuilder.build();
        }
        TopDocs topDocs = indexSearcher.search(ivfQuery, k);
        ScoreDoc[] scoreDocs = topDocs.scoreDocs;
        return new Weight(ivfQuery) {
            @Override
            public Explanation explain(LeafReaderContext context, int doc) throws IOException {
                return null;
            }

            @Override
            public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
                Scorer scorer = new Scorer() {
                    int idx = 0;

                    @Override
                    public float score() throws IOException {
                        return scoreDocs[idx].score;
                    }

                    @Override
                    public int docID() {
                        return scoreDocs[idx].doc;
                    }

                    @Override
                    public DocIdSetIterator iterator() {
                        return new DocIdSetIterator() {
                            @Override
                            public int docID() {
                                return scoreDocs[idx].doc;
                            }

                            @Override
                            public int nextDoc() throws IOException {
                                if (idx == scoreDocs.length) {
                                    return NO_MORE_DOCS;
                                }
                                return scoreDocs[idx++].doc;
                            }

                            @Override
                            public int advance(int target) throws IOException {
                                return slowAdvance(target);
                            }

                            @Override
                            public long cost() {
                                return scoreDocs.length;
                            }
                        };
                    }

                    @Override
                    public float getMaxScore(int upTo) throws IOException {
                        return scoreDocs[upTo].score;
                    }
                };
                return new DefaultScorerSupplier(scorer);
            }

            @Override
            public boolean isCacheable(LeafReaderContext ctx) {
                return false;
            }
        };
    }

    private List<IVFCentroidQuery> generateCentroidQueries(
        LeafReaderContext ctx,
        int numCentroids,
        long docsToExplore,
        AtomicLong globalMinCompetitiveScore) throws IOException {
        List<IVFCentroidMeta> topCentroids = findTopCentroids(ctx, numCentroids, docsToExplore);


        // Calculate max vectors per centroid
        int maxVectorsPerCentroid = (int) Math.max(1, docsToExplore / topCentroids.size());

        // Get parent bitset if diversifying
        Bits parentBitSet = null;
        if (parentsFilter != null) {
            parentBitSet = parentsFilter.getBitSet(ctx);
        }
        List<IVFCentroidQuery> allCentroidQueries = new ArrayList<>();
        // Create CentroidQuery for each selected centroid
        for (IVFCentroidMeta centroid : topCentroids) {
            allCentroidQueries.add(
                new IVFCentroidQuery(
                    field,
                    queryVector,
                    centroid,
                    ctx,
                    maxVectorsPerCentroid,
                    globalMinCompetitiveScore,
                    parentBitSet
                )
            );
        }
        return allCentroidQueries;
    }

    private List<IVFCentroidMeta>  findTopCentroids(LeafReaderContext ctx, int numCentroids, long docsToExplore) throws IOException {
        LeafReader leafReader = ctx.reader();
        FloatVectorValues vectorValues = leafReader.getFloatVectorValues(field);
        if (vectorValues == null || vectorValues.size() == 0) {
            return null;
        }

        // Access IVFVectorsReader through CodecReader
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
        return ivfReader.findTopCentroids(
            fieldInfo,
            queryVector,
            numCentroids
        );
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

    @Override
    public void profile(QueryProfiler queryProfiler) {
        queryProfiler.addVectorOpsCount(vectorOpsCount);
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
