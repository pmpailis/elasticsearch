/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.vectors;


import org.apache.logging.log4j.LogManager;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BulkScorer;
import org.apache.lucene.search.CollectorManager;
import org.apache.lucene.search.ConstantScoreScorer;
import org.apache.lucene.search.DisjunctionMaxQuery;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.FieldExistsQuery;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.Scorable;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.TopScoreDocCollector;
import org.apache.lucene.search.TopScoreDocCollectorManager;
import org.apache.lucene.search.Weight;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.knn.KnnCollectorManager;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.elasticsearch.common.lucene.search.Queries;
import org.elasticsearch.common.lucene.search.TopDocsAndMaxScore;
import org.elasticsearch.index.codec.vectors.cluster.NeighborQueue;
import org.elasticsearch.index.codec.vectors.diskbbq.IVFVectorsReader;
import org.elasticsearch.search.DocValueFormat;

import java.io.IOException;
import java.security.Timestamp;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAccumulator;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

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
        this.totalVectorsVisited = new AtomicLong(0);
    }

    @Override
    TopDocs approximateSearch(LeafReaderContext context, AcceptDocs acceptDocs, int visitedLimit, IVFCollectorManager knnCollectorManager, float visitRatio) throws IOException {
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
        vectorOpsCount = 0;
        IndexReader reader = indexSearcher.getIndexReader();

        // calculate effective visit ratio, i.e. how many vectors we're expected to visit
        float effectiveVisitRatio = calculateEffectiveVisitRatio(totalVectors);
        //calculate filter cost
        float filterSelectivity = 0f;
        if(filter != null) {
            int filterCost = 0;
            var filterWeight = indexSearcher.createWeight(filter, ScoreMode.COMPLETE_NO_SCORES, 1f);
            for (LeafReaderContext ctx : indexSearcher.getIndexReader().leaves()) {
                filterCost += Math.toIntExact(filterWeight.scorerSupplier(ctx).cost());
            }
            filterSelectivity = Math.min(1f, (float) filterCost / totalVectors);
        }
        long maxVectorVisited = Math.max(100, Math.round(2.0 * effectiveVisitRatio * totalVectors));

        // cross-leaf competitive scores; replacing the collector
        // might need to revisit this to check if we can still use the collector
        AtomicLong globalMinCompetitiveScore = reader.leaves().size() > 1
            ? new AtomicLong(NeighborQueue.encodeRaw(Integer.MAX_VALUE, Float.NEGATIVE_INFINITY))
            : null;

        // calculate number of centroids to initially explore
        int numCentroids = Math.max(1, (int) Math.ceil((double) (maxVectorVisited * (1 + (1 - filterSelectivity))) / clusterSize));
        LogManager.getLogger("foo").error("centroids to explore: " + numCentroids);
        // or each leaf, find top centroids and create CentroidQueries
        List<Callable<List<IVFCentroidQuery>>> tasks = new ArrayList<>(reader.leaves().size());
        List<IVFCentroidQuery> allCentroidQueries = new ArrayList<>();
        for (LeafReaderContext context : reader.leaves()) {
            tasks.add(() -> generateCentroidQueries(context, numCentroids, maxVectorVisited, globalMinCompetitiveScore, totalVectorsVisited));
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

        if (filter != null) {
            BooleanQuery.Builder boolBuilder = new BooleanQuery.Builder();
            boolBuilder.add(ivfQuery, BooleanClause.Occur.MUST);
            boolBuilder.add(filter, BooleanClause.Occur.FILTER);
            boolBuilder.add(new FieldExistsQuery(field), BooleanClause.Occur.FILTER);
            ivfQuery = boolBuilder.build();
        }
        TopScoreDocCollectorManager manager = new TopScoreDocCollectorManager(k, Integer.MAX_VALUE);
        TopDocs topDocs = indexSearcher.search(ivfQuery, manager);
        vectorOpsCount = (int) totalVectorsVisited.get();
        LogManager.getLogger("foo").error("vector ops count: " + vectorOpsCount);
        if (topDocs.scoreDocs.length == 0) {
            return Queries.NO_DOCS_INSTANCE;
        }
        return new KnnScoreDocQuery(topDocs.scoreDocs, reader);
        /**
        int[] docs = new int[scoreDocs.length];
        for(int i = 0; i < scoreDocs.length; i++){
            docs[i] = scoreDocs[i].doc;
        }
        Arrays.sort(docs);
        int[] segmentStarts = findSegmentStarts(reader, docs);
        return new Weight(ivfQuery) {
            @Override
            public Explanation explain(LeafReaderContext context, int doc) throws IOException {
                return null;
            }

            public int count(LeafReaderContext context) {
                return segmentStarts[context.ord + 1] - segmentStarts[context.ord];
            }

            @Override
            public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
                Scorer scorer = new Scorer() {
                    final int lower = segmentStarts[context.ord];
                    final int upper = segmentStarts[context.ord + 1];
                    int upTo = -1;

                    @Override
                    public float score() throws IOException {
                        return scoreDocs[upTo].score;
                    }

                    @Override
                    public int advanceShallow(int docId) {
                        int start = Math.max(upTo, lower);
                        int docIdIndex = Arrays.binarySearch(docs, start, upper, docId + context.docBase);
                        if (docIdIndex < 0) {
                            docIdIndex = -1 - docIdIndex;
                        }
                        if (docIdIndex >= upper) {
                            return NO_MORE_DOCS;
                        }
                        return scoreDocs[docIdIndex].doc;
                    }

                    @Override
                    public int docID() {
                        return currentDocId();
                    }

                    private int currentDocId() {
                        if (upTo == -1) {
                            return -1;
                        }
                        if (upTo >= upper) {
                            return NO_MORE_DOCS;
                        }
                        return docs[upTo] - context.docBase;
                    }

                    @Override
                    public DocIdSetIterator iterator() {
                        return new DocIdSetIterator() {
                            @Override
                            public int docID() {
                                return currentDocId();
                            }

                            @Override
                            public int nextDoc() {
                                if (upTo == -1) {
                                    upTo = lower;
                                } else {
                                    ++upTo;
                                }
                                return currentDocId();
                            }

                            @Override
                            public int advance(int target) throws IOException {
                                return slowAdvance(target);
                            }

                            @Override
                            public long cost() {
                                return upper - lower;
                            }
                        };
                    }

                    @Override
                    public float getMaxScore(int doc) throws IOException {
                        // NO_MORE_DOCS indicates the maximum score for all docs in this segment
                        // Anything less than must be accounted for via the docBase.
                        if (doc != NO_MORE_DOCS) {
                            doc += context.docBase;
                        }
                        float maxScore = 0;
                        for (int idx = Math.max(lower, upTo); idx < upper && scoreDocs[idx].doc <= doc; idx++) {
                            maxScore = Math.max(maxScore, scoreDocs[idx].score);
                        }
                        return maxScore;
                    }
                };
                return new DefaultScorerSupplier(scorer);
            }

            @Override
            public boolean isCacheable(LeafReaderContext ctx) {
                return false;
            }
        };
         */
    }

    private static int[] findSegmentStarts(IndexReader reader, int[] docs){
        int[] starts = new int[reader.leaves().size() + 1];
        starts[starts.length - 1] = docs.length;
        if (starts.length == 2) {
            return starts;
        }
        int resultIndex = 0;
        for (int i = 1; i < starts.length - 1; i++) {
            int upper = reader.leaves().get(i).docBase;
            resultIndex = Arrays.binarySearch(docs, resultIndex, docs.length, upper);
            if (resultIndex < 0) {
                resultIndex = -1 - resultIndex;
            }
            starts[i] = resultIndex;
        }
        return starts;
    }

    private List<IVFCentroidQuery> generateCentroidQueries(
        LeafReaderContext ctx,
        int numCentroids,
        long docsToExplore,
        AtomicLong globalMinCompetitiveScore,
        AtomicLong totalVectorsVisited) throws IOException {
//        var startTime = System.nanoTime();
        List<IVFCentroidQuery.IVFCentroidMeta> topCentroids = findTopCentroids(ctx, numCentroids, docsToExplore);
//        LogManager.getLogger("foo").error("topCentroids took: {} nanoseconds", System.nanoTime() - startTime);

        // Calculate max vectors per centroid
       if(topCentroids == null || topCentroids.isEmpty()) {
           return null;
       }
        int maxVectorsPerCentroid = (int) Math.max(1, docsToExplore / topCentroids.size());

        // Get parent bitset if diversifying
        Bits parentBitSet = null;
        if (parentsFilter != null) {
            parentBitSet = parentsFilter.getBitSet(ctx);
        }
        List<IVFCentroidQuery> allCentroidQueries = new ArrayList<>();
        // Create CentroidQuery for each selected centroid
        for (IVFCentroidQuery.IVFCentroidMeta centroid : topCentroids) {
            allCentroidQueries.add(
                new IVFCentroidQuery(
                    field,
                    queryVector,
                    centroid,
                    ctx,
                    maxVectorsPerCentroid,
                    parentBitSet,
                    totalVectorsVisited
                )
            );
        }
        return allCentroidQueries;
    }

    private List<IVFCentroidQuery.IVFCentroidMeta> findTopCentroids(LeafReaderContext ctx, int maxCentroids, long docsToExplore) throws IOException {
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
            maxCentroids,
            field,
            providedVisitRatio,
            docsToExplore
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
