/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */
package org.elasticsearch.search.vectors;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.join.BitSetProducer;
import org.apache.lucene.search.join.CheckJoinIndex;
import org.apache.lucene.search.join.QueryBitSetProducer;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.elasticsearch.common.logging.LogConfigurator;
import org.elasticsearch.index.codec.vectors.cluster.NeighborQueue;
import org.elasticsearch.index.codec.vectors.diskbbq.TestIvfQueryConfigResolver;
import org.elasticsearch.index.codec.vectors.diskbbq.next.ESNextDiskBBQVectorsFormat;
import org.junit.After;
import org.junit.Before;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.LongAccumulator;

import static com.carrotsearch.randomizedtesting.RandomizedTest.randomFloat;
import static com.carrotsearch.randomizedtesting.RandomizedTest.randomIntBetween;
import static org.elasticsearch.search.vectors.AbstractMaxScoreKnnCollector.LEAST_COMPETITIVE;

/**
 * Verifies query-level cross-leaf score-floor sharing: with sharing enabled, every leaf's collectors publish their
 * (gated) min-competitive floors to one query-level accumulator and prune against the best of all leaves. Sharing is
 * lossless by construction — a published floor certifies at least {@code mergeK} distinct docs above it, so a doc
 * pruned below the floor can never make the merged top-{@code mergeK} — hence results must be <i>equal</i> to the
 * no-sharing baseline (modulo exactly-tied scores), at full and at partial visit ratios alike.
 */
public class IVFCrossLeafFloorSharingTests extends LuceneTestCase {

    static {
        LogConfigurator.configureESLogging(); // native access requires logging to be initialized
    }

    private static final String FIELD = "field";

    private ExecutorService executor;
    private TaskExecutor taskExecutor;
    private int workers;

    @Before
    public void setUpExecutor() {
        workers = randomIntBetween(2, 4);
        executor = Executors.newFixedThreadPool(workers);
        taskExecutor = new TaskExecutor(executor);
    }

    @After
    public void tearDownExecutor() {
        executor.shutdownNow();
    }

    /** The publish gate blocks the global floor; consuming it is never gated. */
    public void testScoreFloorsPublishGate() {
        LongAccumulator global = new LongAccumulator(Long::max, LEAST_COMPETITIVE);
        ScoreFloors gated = new ScoreFloors(global, false);
        ScoreFloors publishing = new ScoreFloors(global, true);

        gated.publish(NeighborQueue.encodeRaw(1, 0.9f));
        assertEquals("a gated leaf must not export its floor", LEAST_COMPETITIVE, global.get());
        assertEquals("the local floor stays visible to its own leaf", NeighborQueue.encodeRaw(1, 0.9f), gated.floor());

        publishing.publish(NeighborQueue.encodeRaw(2, 0.5f));
        assertEquals(NeighborQueue.encodeRaw(2, 0.5f), global.get());
        // consuming is ungated: the gated leaf folds the global floor in, keeping the max
        assertEquals(NeighborQueue.encodeRaw(1, 0.9f), gated.floor());
        publishing.publish(NeighborQueue.encodeRaw(3, 1.5f));
        assertEquals(NeighborQueue.encodeRaw(3, 1.5f), gated.floor());
        assertEquals(NeighborQueue.encodeRaw(3, 1.5f), publishing.floor());
    }

    public void testMultiSegmentEqualityWithSharing() throws IOException {
        int dims = randomIntBetween(16, 48);
        int segments = randomIntBetween(2, 4);
        int perSegment = randomIntBetween(1000, 1500);
        try (Directory dir = newDirectory()) {
            indexSegments(dir, segments, perSegment, dims);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                assertEquals(segments, reader.leaves().size());
                IndexSearcher searcher = new IndexSearcher(reader);
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    float visitRatio = i % 2 == 0 ? 1.0f : randomFloat() * 0.15f + 0.05f;
                    TopDocs baseline = searcher.search(plainQuery(query, k, visitRatio, false, false), k);
                    TopDocs sharing = searcher.search(plainQuery(query, k, visitRatio, true, false), k);
                    IVFIntraSegmentParallelScanTests.assertSameTopDocs(baseline, sharing);
                }
            }
        }
    }

    public void testSharingCombinedWithIntraSegmentWorkers() throws IOException {
        int dims = randomIntBetween(16, 48);
        int segments = randomIntBetween(2, 3);
        int perSegment = randomIntBetween(1000, 1500);
        try (Directory dir = newDirectory()) {
            indexSegments(dir, segments, perSegment, dims);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 20);
                    // visit ratio 1.0: serial-no-sharing and parallel-with-sharing must agree exactly (modulo ties)
                    TopDocs baseline = searcher.search(plainQuery(query, k, 1.0f, false, false), k);
                    TopDocs combined = searcher.search(plainQuery(query, k, 1.0f, true, true), k);
                    IVFIntraSegmentParallelScanTests.assertSameTopDocs(baseline, combined);
                }
            }
        }
    }

    public void testDiversifiedMultiSegmentEqualityWithSharing() throws IOException {
        int dims = randomIntBetween(16, 32);
        int segments = 2;
        int parentsPerSegment = randomIntBetween(120, 180);
        try (Directory dir = newDirectory()) {
            indexParentChildSegments(dir, segments, parentsPerSegment, dims);
            try (DirectoryReader reader = DirectoryReader.open(dir)) {
                IndexSearcher searcher = new IndexSearcher(reader);
                BitSetProducer parentsFilter = new QueryBitSetProducer(new TermQuery(new Term("docType", "_parent")));
                CheckJoinIndex.check(reader, parentsFilter);
                for (int i = 0; i < 5; i++) {
                    float[] query = randomVector(dims);
                    int k = randomIntBetween(5, 15);
                    TopDocs baseline = searcher.search(diversifiedQuery(query, k, parentsFilter, false), k);
                    TopDocs sharing = searcher.search(diversifiedQuery(query, k, parentsFilter, true), k);
                    IVFIntraSegmentParallelScanTests.assertSameTopDocs(baseline, sharing);
                }
            }
        }
    }

    private IVFKnnFloatVectorQuery plainQuery(float[] vector, int k, float visitRatio, boolean sharing, boolean intraWorkers) {
        return new IVFKnnFloatVectorQuery(FIELD, vector, k, k, null, visitRatio, testResolver()) {
            @Override
            boolean crossLeafFloorSharing() {
                return sharing;
            }

            @Override
            IVFParallelismConfig intraSegmentParallelism() {
                return intraWorkers ? new IVFParallelismConfig(taskExecutor, workers, () -> {}) : null;
            }
        };
    }

    private DiversifyingChildrenIVFKnnFloatVectorQuery diversifiedQuery(float[] vector, int k, BitSetProducer parents, boolean sharing) {
        return new DiversifyingChildrenIVFKnnFloatVectorQuery(FIELD, vector, k, k, null, parents, 1.0f, testResolver()) {
            @Override
            boolean crossLeafFloorSharing() {
                return sharing;
            }

            @Override
            IVFParallelismConfig intraSegmentParallelism() {
                return null;
            }
        };
    }

    private static TestIvfQueryConfigResolver testResolver() {
        return new TestIvfQueryConfigResolver(
            ESNextDiskBBQVectorsFormat.CentroidIndexFormat.FLAT,
            ESNextDiskBBQVectorsFormat.QuantEncoding.ONE_BIT_4BIT_QUERY,
            false,
            1.0f
        );
    }

    private static void indexSegments(Directory dir, int segments, int perSegment, int dims) throws IOException {
        IndexWriterConfig config = new IndexWriterConfig().setCodec(
            TestUtil.alwaysKnnVectorsFormat(new ESNextDiskBBQVectorsFormat(64, 16, null))
        )
            .setMaxBufferedDocs(perSegment * 2)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);
        try (IndexWriter writer = new IndexWriter(dir, config)) {
            for (int segment = 0; segment < segments; segment++) {
                for (int i = 0; i < perSegment; i++) {
                    Document doc = new Document();
                    doc.add(new KnnFloatVectorField(FIELD, randomVector(dims), VectorSimilarityFunction.EUCLIDEAN));
                    writer.addDocument(doc);
                }
                writer.commit();
            }
        }
    }

    private static void indexParentChildSegments(Directory dir, int segments, int parentsPerSegment, int dims) throws IOException {
        IndexWriterConfig config = new IndexWriterConfig().setCodec(
            TestUtil.alwaysKnnVectorsFormat(new ESNextDiskBBQVectorsFormat(64, 16, null))
        )
            .setMaxBufferedDocs(parentsPerSegment * 20)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH)
            .setMergePolicy(NoMergePolicy.INSTANCE);
        try (IndexWriter writer = new IndexWriter(dir, config)) {
            for (int segment = 0; segment < segments; segment++) {
                for (int parent = 0; parent < parentsPerSegment; parent++) {
                    List<Document> block = new ArrayList<>();
                    int children = randomIntBetween(1, 8);
                    for (int child = 0; child < children; child++) {
                        Document childDoc = new Document();
                        childDoc.add(new KnnFloatVectorField(FIELD, randomVector(dims), VectorSimilarityFunction.EUCLIDEAN));
                        block.add(childDoc);
                    }
                    Document parentDoc = new Document();
                    parentDoc.add(new StringField("docType", "_parent", Field.Store.NO));
                    block.add(parentDoc);
                    writer.addDocuments(block);
                }
                writer.commit();
            }
        }
    }

    private static float[] randomVector(int dims) {
        float[] vector = new float[dims];
        for (int i = 0; i < dims; i++) {
            vector[i] = randomFloat() * 2 - 1;
        }
        return vector;
    }
}
